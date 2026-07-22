#include "forward.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "attention.h"
#include "cuda_check.h"
#include "kernels.h"
#include "matmul.h"
#include "rmsnorm.h"

// Step 6 — the composition (docs/12-cuda-forward.md). No new kernel lives here: this file
// is the plumbing that wiring needs (blob upload, weight views, id buffer, scratch) plus a
// launch sequence held line-for-line to Model::forward()'s op order in model.cpp.
//
// Deliberately unoptimized, per the spec's scope boundary: every launcher keeps its
// CUDA_CHECK_KERNEL() device sync (~230 syncs per call over 22 layers) and scratch is
// allocated per call. A first end-to-end bring-up wants every failure to surface at the
// offending launch; dropping the syncs and preallocating scratch are named perf follow-ups,
// to be held to this step's pinned tolerance.

namespace atlas {

GpuModel GpuModel::create(const Model& model) {
    const WeightStore& w = model.weights;
    assert(w.base() != nullptr && w.size_bytes() > 0 && "GpuModel::create: empty WeightStore");

    GpuModel g;
    g.model_ = &model;
    g.blob_bytes_ = w.size_bytes();
    CUDA_CHECK(cudaMalloc(&g.blob_, g.blob_bytes_));
    // One H2D of the whole mapping. Touching all 4.4 GB here also faults the blob's pages
    // in on the host, which is why this cost belongs at create() and not in forward().
    CUDA_CHECK(cudaMemcpy(g.blob_, w.base(), g.blob_bytes_, cudaMemcpyHostToDevice));
    return g;
}

GpuModel::~GpuModel() {
    if (blob_) cudaFree(blob_);
}

GpuModel::GpuModel(GpuModel&& other) noexcept
    : model_(other.model_), blob_(other.blob_), blob_bytes_(other.blob_bytes_) {
    other.model_ = nullptr;
    other.blob_ = nullptr;
    other.blob_bytes_ = 0;
}

GpuModel& GpuModel::operator=(GpuModel&& other) noexcept {
    if (this != &other) {
        if (blob_) cudaFree(blob_);
        model_ = other.model_;
        blob_ = other.blob_;
        blob_bytes_ = other.blob_bytes_;
        other.model_ = nullptr;
        other.blob_ = nullptr;
        other.blob_bytes_ = 0;
    }
    return *this;
}

// Zero-copy device view of a named weight: same shape as the host view, at the same
// element offset from the blob base. No per-tensor allocation and no name->device map —
// WeightStore remains the single source of truth (docs/12-cuda-forward.md).
DeviceTensor GpuModel::weight(const std::string& name) const {
    assert(model_ != nullptr && "GpuModel::weight on a moved-from model");
    const Tensor& host = model_->weights.get(name);
    const std::ptrdiff_t offset = host.data - model_->weights.base();
    assert(offset >= 0 &&
           static_cast<size_t>(offset) * sizeof(float) < blob_bytes_ &&
           "GpuModel::weight: host view outside the mapped blob");
    return DeviceTensor::view(blob_ + offset, host.shape);
}

Tensor GpuModel::forward(const std::vector<int>& token_ids) const {
    assert(model_ != nullptr && "GpuModel::forward on a moved-from model");
    const Config& c = model_->config;
    const int64_t seq = static_cast<int64_t>(token_ids.size());
    const int64_t H = c.hidden_size;
    const int64_t KV = c.kv_dim();
    const int64_t I = c.intermediate_size;
    const int hd = c.head_dim;
    assert(seq > 0 && "GpuModel::forward: empty prompt");

    // Token ids: launch_embed takes a raw device int32 array (DeviceTensor is float-typed),
    // so this one buffer is malloc'd/copied/freed by hand rather than earning an abstraction.
    for (const int id : token_ids) {
        assert(id >= 0 && id < c.vocab_size && "token id out of range");
        (void)id;
    }
    int* ids_d = nullptr;
    const size_t ids_bytes = static_cast<size_t>(seq) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&ids_d, ids_bytes));
    CUDA_CHECK(cudaMemcpy(ids_d, token_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

    // RoPE tables are computed on the host by the shared rope_tables() helper (cheap:
    // [seq, 64] x2) and uploaded once per call — one copy of the table math for both paths.
    Tensor host_cos;
    Tensor host_sin;
    rope_tables(seq, c, host_cos, host_sin);
    DeviceTensor rope_cos = DeviceTensor::alloc({seq, hd});
    DeviceTensor rope_sin = DeviceTensor::alloc({seq, hd});
    to_device(host_cos, rope_cos);
    to_device(host_sin, rope_sin);

    // Scratch, mirroring Model::forward()'s nine buffers; reused across the 22 layers.
    DeviceTensor h = DeviceTensor::alloc({seq, H});
    DeviceTensor x = DeviceTensor::alloc({seq, H});     // RMSNorm output
    DeviceTensor q = DeviceTensor::alloc({seq, H});     // [seq, 32*64]
    DeviceTensor k = DeviceTensor::alloc({seq, KV});    // [seq, 4*64] — the GQA asymmetry
    DeviceTensor v = DeviceTensor::alloc({seq, KV});
    DeviceTensor ctx = DeviceTensor::alloc({seq, H});   // attention context, pre-o_proj
    DeviceTensor attn = DeviceTensor::alloc({seq, H});
    DeviceTensor gate = DeviceTensor::alloc({seq, I});
    DeviceTensor up = DeviceTensor::alloc({seq, I});
    DeviceTensor mlp = DeviceTensor::alloc({seq, H});

    // 1. Embed: gather rows of the embedding table.
    const DeviceTensor embed = weight("model.embed_tokens.weight");
    launch_embed(embed, ids_d, seq, h);

    // 2. The 22 pre-norm residual blocks. FP32 throughout — qweights is ignored here, so
    // there is no linear dispatch: every projection is launch_matmul.
    for (int layer = 0; layer < c.num_layers; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";

        // -- attention sublayer --
        launch_rmsnorm(h, weight(p + "input_layernorm.weight"), c.rms_norm_eps, x);
        launch_matmul(x, weight(p + "self_attn.q_proj.weight"), q);
        launch_matmul(x, weight(p + "self_attn.k_proj.weight"), k);
        launch_matmul(x, weight(p + "self_attn.v_proj.weight"), v);
        launch_rope(q, rope_cos, rope_sin, c.num_heads, hd);
        launch_rope(k, rope_cos, rope_sin, c.num_kv_heads, hd);

        launch_attention(q, k, v, c.num_heads, c.num_kv_heads, hd, ctx);
        launch_matmul(ctx, weight(p + "self_attn.o_proj.weight"), attn);
        launch_add(h, attn);  // residual

        // -- SwiGLU MLP sublayer --
        launch_rmsnorm(h, weight(p + "post_attention_layernorm.weight"), c.rms_norm_eps, x);
        launch_matmul(x, weight(p + "mlp.gate_proj.weight"), gate);
        launch_matmul(x, weight(p + "mlp.up_proj.weight"), up);
        launch_swiglu(gate, up);
        launch_matmul(gate, weight(p + "mlp.down_proj.weight"), mlp);
        launch_add(h, mlp);  // residual
    }

    // 3. Final RMSNorm, 4. LM head -> [seq, vocab] logits, then the single D2H.
    launch_rmsnorm(h, weight("model.norm.weight"), c.rms_norm_eps, x);
    DeviceTensor logits_d = DeviceTensor::alloc({seq, c.vocab_size});
    launch_matmul(x, weight("lm_head.weight"), logits_d);

    Tensor logits = Tensor::zeros({seq, c.vocab_size});
    to_host(logits_d, logits);

    CUDA_CHECK(cudaFree(ids_d));
    return logits;
}

}  // namespace atlas
