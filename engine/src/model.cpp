#include "../include/model.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// The TinyLlama forward pass and the mmap'd weight loading that feeds it. The model
// math (RMSNorm, RoPE, GQA attention, SwiGLU) lives here, built from the tensor.h
// primitives plus a `linear` helper — tensor.cpp stays memory + core ops only.
// Everything below matches docs/03-model-weights.md; the reference oracle is
// reference/logits.npy (tolerance-checked in Step 4, top-1 sanity-checked here).

namespace atlas {

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "model: %s\n", msg.c_str());
    std::exit(1);
}

// y = x @ Wᵀ. x is [m, in], w keeps the PyTorch nn.Linear [out, in] layout (no
// transpose at convert time), out is [m, out]. Both w rows and x rows are contiguous,
// so the inner dot product walks both buffers sequentially.
void linear(const Tensor& x, const Tensor& w, Tensor& out) {
    assert(x.shape.size() == 2 && w.shape.size() == 2 && out.shape.size() == 2);
    const int64_t m = x.shape[0];
    const int64_t in = x.shape[1];
    const int64_t out_f = w.shape[0];
    assert(w.shape[1] == in && "linear: weight in_features mismatch");
    assert(out.shape[0] == m && out.shape[1] == out_f && "linear: out shape mismatch");
    for (int64_t i = 0; i < m; ++i) {
        const float* xrow = x.data + i * in;
        for (int64_t o = 0; o < out_f; ++o) {
            const float* wrow = w.data + o * in;
            float acc = 0.0f;
            for (int64_t p = 0; p < in; ++p) acc += xrow[p] * wrow[p];
            out.data[i * out_f + o] = acc;
        }
    }
}

// RMSNorm per row: x_i / sqrt(mean(x²) + eps) * w_i. No mean subtraction (this is
// RMSNorm, not LayerNorm); the variance accumulates in FP32.
void rmsnorm(const Tensor& x, const Tensor& w, float eps, Tensor& out) {
    assert(x.shape == out.shape && x.shape.size() == 2);
    const int64_t m = x.shape[0];
    const int64_t n = x.shape[1];
    assert(w.numel() == n && "rmsnorm: weight size mismatch");
    for (int64_t i = 0; i < m; ++i) {
        const float* row = x.data + i * n;
        float ss = 0.0f;
        for (int64_t j = 0; j < n; ++j) ss += row[j] * row[j];
        const float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
        for (int64_t j = 0; j < n; ++j) {
            out.data[i * n + j] = row[j] * scale * w.data[j];
        }
    }
}

// RoPE, HF Llama's half-split ("NeoX") rotation — NOT interleaved adjacent pairs.
// x is [seq, n_heads*head_dim] viewed per head; cos/sin are [seq, head_dim] built by
// concatenating the head_dim/2 angles with themselves. Applied in place:
//   x_rot = x * cos + rotate_half(x) * sin,  rotate_half(x) = cat(-x[d/2:], x[:d/2])
void rope(Tensor& x, const Tensor& cos, const Tensor& sin, int n_heads, int head_dim) {
    const int64_t seq = x.shape[0];
    const int half = head_dim / 2;
    assert(x.shape[1] == static_cast<int64_t>(n_heads) * head_dim);
    for (int64_t pos = 0; pos < seq; ++pos) {
        const float* c = cos.data + pos * head_dim;
        const float* s = sin.data + pos * head_dim;
        for (int h = 0; h < n_heads; ++h) {
            float* v = x.data + pos * x.shape[1] + static_cast<int64_t>(h) * head_dim;
            for (int i = 0; i < half; ++i) {
                const float x0 = v[i];
                const float x1 = v[i + half];
                v[i] = x0 * c[i] - x1 * s[i];
                v[i + half] = x1 * c[i + half] + x0 * s[i + half];
            }
        }
    }
}

}  // namespace

// --- WeightStore: mmap the FP32 blob, hand out non-owning views by manifest name ---

WeightStore WeightStore::load(const std::string& bin_path, const std::string& manifest_path) {
    WeightStore ws;

#ifdef _WIN32
    HANDLE file = CreateFileA(bin_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) die("cannot open " + bin_path);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file, &sz)) die("GetFileSizeEx failed for " + bin_path);
    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) die("CreateFileMapping failed for " + bin_path);
    void* base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!base) die("MapViewOfFile failed for " + bin_path);
    ws.file_handle_ = file;
    ws.mapping_handle_ = mapping;
    ws.base_ = static_cast<float*>(base);
    ws.size_bytes_ = static_cast<size_t>(sz.QuadPart);
#else
    int fd = ::open(bin_path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + bin_path);
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat failed for " + bin_path);
    void* base = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) die("mmap failed for " + bin_path);
    ws.file_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
    ws.base_ = static_cast<float*>(base);
    ws.size_bytes_ = static_cast<size_t>(st.st_size);
#endif

    // Manifest: one tensor per line, "name byte_offset ndim d0 d1 ...".
    std::ifstream mf(manifest_path);
    if (!mf) die("cannot open " + manifest_path);
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string name;
        uint64_t offset = 0;
        int ndim = 0;
        if (!(ls >> name >> offset >> ndim)) die("malformed manifest line: " + line);
        std::vector<int64_t> shape(static_cast<size_t>(ndim));
        int64_t n_elem = 1;
        for (int d = 0; d < ndim; ++d) {
            if (!(ls >> shape[static_cast<size_t>(d)])) die("malformed manifest line: " + line);
            n_elem *= shape[static_cast<size_t>(d)];
        }
        if (offset % sizeof(float) != 0) die("misaligned offset for " + name);
        if (offset + static_cast<uint64_t>(n_elem) * sizeof(float) > ws.size_bytes_) {
            die("tensor extends past blob end: " + name);
        }
        ws.tensors_.emplace(name, Tensor::view(ws.base_ + offset / sizeof(float), std::move(shape)));
    }
    if (ws.tensors_.empty()) die("empty manifest: " + manifest_path);
    return ws;
}

WeightStore::~WeightStore() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (mapping_handle_) CloseHandle(mapping_handle_);
    if (file_handle_) CloseHandle(file_handle_);
#else
    if (base_) ::munmap(base_, size_bytes_);
    if (file_handle_) ::close(static_cast<int>(reinterpret_cast<intptr_t>(file_handle_)));
#endif
}

WeightStore::WeightStore(WeightStore&& other) noexcept
    : file_handle_(other.file_handle_),
      mapping_handle_(other.mapping_handle_),
      base_(other.base_),
      size_bytes_(other.size_bytes_),
      tensors_(std::move(other.tensors_)) {
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
    other.base_ = nullptr;
    other.size_bytes_ = 0;
}

WeightStore& WeightStore::operator=(WeightStore&& other) noexcept {
    if (this != &other) {
        this->~WeightStore();
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        base_ = other.base_;
        size_bytes_ = other.size_bytes_;
        tensors_ = std::move(other.tensors_);
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
        other.base_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

const Tensor& WeightStore::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) die("weight not found: " + name);
    return it->second;
}

bool WeightStore::has(const std::string& name) const {
    return tensors_.count(name) != 0;
}

// --- Model ---

Model Model::load(const std::string& bin_path, const std::string& manifest_path,
                  const Config& cfg) {
    Model m;
    m.config = cfg;
    m.weights = WeightStore::load(bin_path, manifest_path);
    return m;
}

Tensor Model::forward(const std::vector<int>& token_ids) const {
    const Config& c = config;
    const int64_t seq = static_cast<int64_t>(token_ids.size());
    const int64_t H = c.hidden_size;
    const int64_t KV = c.kv_dim();
    const int64_t I = c.intermediate_size;
    const int hd = c.head_dim;
    const int half = hd / 2;

    // 1. Embed: gather rows of the embedding table.
    const Tensor& embed = weights.get("model.embed_tokens.weight");
    Tensor h = Tensor::zeros({seq, H});
    for (int64_t i = 0; i < seq; ++i) {
        const int id = token_ids[static_cast<size_t>(i)];
        assert(id >= 0 && id < c.vocab_size && "token id out of range");
        const float* src = embed.data + static_cast<int64_t>(id) * H;
        for (int64_t j = 0; j < H; ++j) h.data[i * H + j] = src[j];
    }

    // Precompute the RoPE tables once for all layers: inv_freq[i] = 1/theta^(2i/hd) for
    // i in [0, hd/2); cos/sin rows are the hd/2 angles concatenated with themselves
    // (the half-split layout rope() expects).
    Tensor rope_cos = Tensor::zeros({seq, hd});
    Tensor rope_sin = Tensor::zeros({seq, hd});
    for (int64_t pos = 0; pos < seq; ++pos) {
        for (int i = 0; i < half; ++i) {
            const float inv_freq =
                1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / static_cast<float>(hd));
            const float angle = static_cast<float>(pos) * inv_freq;
            const float cv = std::cos(angle);
            const float sv = std::sin(angle);
            rope_cos.data[pos * hd + i] = cv;
            rope_cos.data[pos * hd + i + half] = cv;
            rope_sin.data[pos * hd + i] = sv;
            rope_sin.data[pos * hd + i + half] = sv;
        }
    }

    // Scratch buffers reused across the 22 layers (allocations stay explicit and bounded).
    Tensor x = Tensor::zeros({seq, H});      // RMSNorm output
    Tensor q = Tensor::zeros({seq, H});      // [seq, 32*64]
    Tensor k = Tensor::zeros({seq, KV});     // [seq, 4*64] — the GQA asymmetry
    Tensor v = Tensor::zeros({seq, KV});
    Tensor ctx = Tensor::zeros({seq, H});    // attention context, pre-o_proj
    Tensor attn = Tensor::zeros({seq, H});
    Tensor gate = Tensor::zeros({seq, I});
    Tensor up = Tensor::zeros({seq, I});
    Tensor mlp = Tensor::zeros({seq, H});
    std::vector<float> scores(static_cast<size_t>(seq));  // one query row's attention probs

    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));

    // 2. The 22 pre-norm residual blocks.
    for (int layer = 0; layer < c.num_layers; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";

        // -- attention sublayer --
        rmsnorm(h, weights.get(p + "input_layernorm.weight"), c.rms_norm_eps, x);
        linear(x, weights.get(p + "self_attn.q_proj.weight"), q);
        linear(x, weights.get(p + "self_attn.k_proj.weight"), k);
        linear(x, weights.get(p + "self_attn.v_proj.weight"), v);
        rope(q, rope_cos, rope_sin, c.num_heads, hd);
        rope(k, rope_cos, rope_sin, c.num_kv_heads, hd);

        // Causal scaled-dot-product attention, one query head at a time. GQA: query
        // head hq reads kv head hq / q_per_kv (HF repeat_kv expands kv heads in
        // contiguous blocks of 8). Softmax in FP32 with max subtraction.
        for (int hq = 0; hq < c.num_heads; ++hq) {
            const int hkv = hq / c.q_per_kv();
            const int64_t q_off = static_cast<int64_t>(hq) * hd;
            const int64_t kv_off = static_cast<int64_t>(hkv) * hd;
            for (int64_t i = 0; i < seq; ++i) {
                const float* qi = q.data + i * H + q_off;
                float max_score = -INFINITY;
                for (int64_t j = 0; j <= i; ++j) {  // causal: keys j > i masked out
                    const float* kj = k.data + j * KV + kv_off;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; ++d) dot += qi[d] * kj[d];
                    const float sc = dot * inv_sqrt_hd;
                    scores[static_cast<size_t>(j)] = sc;
                    if (sc > max_score) max_score = sc;
                }
                float denom = 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    const float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                    scores[static_cast<size_t>(j)] = e;
                    denom += e;
                }
                float* out_row = ctx.data + i * H + q_off;
                for (int d = 0; d < hd; ++d) out_row[d] = 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    const float w = scores[static_cast<size_t>(j)] / denom;
                    const float* vj = v.data + j * KV + kv_off;
                    for (int d = 0; d < hd; ++d) out_row[d] += w * vj[d];
                }
            }
        }
        linear(ctx, weights.get(p + "self_attn.o_proj.weight"), attn);
        add(h, attn, h);  // residual

        // -- SwiGLU MLP sublayer --
        rmsnorm(h, weights.get(p + "post_attention_layernorm.weight"), c.rms_norm_eps, x);
        linear(x, weights.get(p + "mlp.gate_proj.weight"), gate);
        linear(x, weights.get(p + "mlp.up_proj.weight"), up);
        // m = SiLU(gate) * up, in place in `gate`; SiLU(z) = z * sigmoid(z).
        {
            const int64_t total = gate.numel();
            for (int64_t e = 0; e < total; ++e) {
                const float z = gate.data[e];
                gate.data[e] = z / (1.0f + std::exp(-z)) * up.data[e];
            }
        }
        linear(gate, weights.get(p + "mlp.down_proj.weight"), mlp);
        add(h, mlp, h);  // residual
    }

    // 3. Final RMSNorm, 4. LM head -> [seq, vocab] logits.
    rmsnorm(h, weights.get("model.norm.weight"), c.rms_norm_eps, x);
    Tensor logits = Tensor::zeros({seq, c.vocab_size});
    linear(x, weights.get("lm_head.weight"), logits);
    return logits;
}

}  // namespace atlas
