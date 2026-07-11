// Utility-kernel validation — Phase 2 · Step 4 (docs/10-cuda-kernels.md).
//
// All four kernels are element-wise or gather — no reduction, so unlike matmul/RMSNorm
// there is no reordering diff. embed (integer-indexed row copy) and add (straight FP32
// sum) are expected BIT-EXACT: pinned at 0, and a nonzero diff is a bug, not a tolerance
// question. swiglu differs only by __expf vs std::exp rounding; rope by FMA contraction
// of the same table-lookup arithmetic — both measured then pinned.
//
// Oracles: atlas::add() (tensor.h), atlas::swiglu()/rope() (model.h) — all validated
// against HuggingFace in Phase 1 — plus the inline embed row-gather from Model::forward()
// re-stated here (three lines; model.cpp:310). The rope cos/sin tables mirror forward()'s
// preamble construction (model.cpp:319) so the kernel sees real half-split-layout angles.
//
// The in-place oracles (add's accumulator, swiglu's gate, rope's x) mutate their host
// input, so each case uploads to the device BEFORE running the CPU oracle. Blob-free,
// rides the Step-1 infra + shared harness; a failure here is unambiguously a kernel bug.

#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "../cuda_check.h"
#include "../device_tensor.h"
#include "../kernels.h"
#include "../../include/tensor.h"
#include "../../include/model.h"  // atlas::swiglu / atlas::rope oracles
#include "test_harness.h"         // CHECK + g_failures, Diff/compare, fill_prng

// TinyLlama's real RoPE base (Config::rope_theta).
constexpr float kRopeTheta = 10000.0f;

// embed and add: integer gather / unreordered FP32 sum — bit-exact by design (doc-10),
// pinned at 0. Confirmed 0 across all shapes on the first green run (A6000, CUDA 12.6).
constexpr double kExactTol = 0.0;

// swiglu: the only GPU↔CPU difference is __expf vs std::exp rounding, one ulp-scale error
// per element on ~[-0.5,0.5) inputs. Measured worst case 1.49e-08 max-abs (prefill and
// decode [_,5632]) on the A6000 (CUDA 12.6); pinned at 1e-6 — ~67x headroom, still ~7
// orders below a real-bug signal (a wrong gate is O(0.1) on these inputs).
constexpr double kSwigluTol = 1e-6;

// rope: same table-lookup arithmetic on both sides; the only difference is nvcc's FMA
// contraction of the rotate pair. Measured worst case 5.96e-08 max-abs (= 2^-24, one
// half-ulp; q-prefill and k-gqa) on the A6000 (CUDA 12.6); q-decode is bit-exact (seq=1
// means pos=0 only: cos=1/sin=0 is an identity rotation). Pinned at 1e-6 — ~17x headroom,
// and a wrong rotation (bad angle/pairing) is O(1), not 1e-8.
constexpr double kRopeTol = 1e-6;

// --- launch_embed vs the inline row-gather in Model::forward() ---
static void run_embed(const char* name, int64_t vocab, int64_t H, int64_t seq,
                      uint32_t seed) {
    atlas::Tensor embed = atlas::Tensor::zeros({vocab, H});
    fill_prng(embed, seed);

    // Deterministic ids in [0, vocab), same LCG family as fill_prng.
    std::vector<int> ids(seq);
    uint32_t s = seed ^ 0xBEEFu;
    for (int64_t i = 0; i < seq; ++i) {
        s = s * 1664525u + 1013904223u;
        ids[i] = static_cast<int>(s % static_cast<uint32_t>(vocab));
    }

    // CPU oracle: the forward() gather, h[i, :] = embed[ids[i], :].
    atlas::Tensor ref = atlas::Tensor::zeros({seq, H});
    for (int64_t i = 0; i < seq; ++i) {
        const float* src = embed.data + static_cast<int64_t>(ids[i]) * H;
        for (int64_t j = 0; j < H; ++j) ref.data[i * H + j] = src[j];
    }

    // GPU. DeviceTensor is float-typed, so the id buffer is a raw cudaMalloc (doc-10).
    atlas::DeviceTensor dembed = atlas::DeviceTensor::alloc({vocab, H});
    atlas::DeviceTensor dout = atlas::DeviceTensor::alloc({seq, H});
    atlas::to_device(embed, dembed);
    int* ids_d = nullptr;
    CUDA_CHECK(cudaMalloc(&ids_d, seq * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ids_d, ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice));
    atlas::launch_embed(dembed, ids_d, seq, dout);
    CUDA_CHECK(cudaFree(ids_d));

    atlas::Tensor got = atlas::Tensor::zeros({seq, H});
    atlas::to_host(dout, got);

    const Diff d = compare(got, ref);
    std::printf("  embed  %-10s [vocab=%lld H=%lld seq=%lld]  max-abs=%.3g mean-abs=%.3g (tol=%g)\n",
                name, static_cast<long long>(vocab), static_cast<long long>(H),
                static_cast<long long>(seq), d.max_abs, d.mean_abs, kExactTol);
    CHECK(d.max_abs <= kExactTol);
}

// --- launch_add vs atlas::add(), the in-place residual accumulate add(h, x, h) ---
static void run_add(const char* name, int64_t m, int64_t n, uint32_t seed) {
    atlas::Tensor a = atlas::Tensor::zeros({m, n});
    atlas::Tensor b = atlas::Tensor::zeros({m, n});
    fill_prng(a, seed);
    fill_prng(b, seed ^ 0x9E3779B9u);

    // Upload before the oracle: the kernel is in-place in a, like the forward pass.
    atlas::DeviceTensor da = atlas::DeviceTensor::alloc({m, n});
    atlas::DeviceTensor db = atlas::DeviceTensor::alloc({m, n});
    atlas::to_device(a, da);
    atlas::to_device(b, db);

    atlas::Tensor ref = atlas::Tensor::zeros({m, n});
    atlas::add(a, b, ref);

    atlas::launch_add(da, db);
    atlas::Tensor got = atlas::Tensor::zeros({m, n});
    atlas::to_host(da, got);

    const Diff d = compare(got, ref);
    std::printf("  add    %-10s [m=%lld n=%lld]  max-abs=%.3g mean-abs=%.3g (tol=%g)\n",
                name, static_cast<long long>(m), static_cast<long long>(n),
                d.max_abs, d.mean_abs, kExactTol);
    CHECK(d.max_abs <= kExactTol);
}

// --- launch_swiglu vs atlas::swiglu(), in place in gate ---
static void run_swiglu(const char* name, int64_t m, int64_t n, uint32_t seed) {
    atlas::Tensor gate = atlas::Tensor::zeros({m, n});
    atlas::Tensor up = atlas::Tensor::zeros({m, n});
    fill_prng(gate, seed);
    fill_prng(up, seed ^ 0x9E3779B9u);

    atlas::DeviceTensor dgate = atlas::DeviceTensor::alloc({m, n});
    atlas::DeviceTensor dup = atlas::DeviceTensor::alloc({m, n});
    atlas::to_device(gate, dgate);
    atlas::to_device(up, dup);

    atlas::swiglu(gate, up);  // gate becomes the reference

    atlas::launch_swiglu(dgate, dup);
    atlas::Tensor got = atlas::Tensor::zeros({m, n});
    atlas::to_host(dgate, got);

    const Diff d = compare(got, gate);
    std::printf("  swiglu %-10s [m=%lld n=%lld]  max-abs=%.3g mean-abs=%.3g (tol=%.1g)\n",
                name, static_cast<long long>(m), static_cast<long long>(n),
                d.max_abs, d.mean_abs, kSwigluTol);
    CHECK(d.max_abs <= kSwigluTol);
}

// --- launch_rope vs atlas::rope(), in place in x, real forward() angle tables ---
static void run_rope(const char* name, int64_t seq, int n_heads, int head_dim,
                     uint32_t seed) {
    const int half = head_dim / 2;
    atlas::Tensor x = atlas::Tensor::zeros({seq, static_cast<int64_t>(n_heads) * head_dim});
    fill_prng(x, seed);

    // cos/sin tables exactly as forward()'s preamble builds them (model.cpp:319): hd/2
    // angles concatenated with themselves — the half-split layout rope() expects.
    atlas::Tensor cos_t = atlas::Tensor::zeros({seq, head_dim});
    atlas::Tensor sin_t = atlas::Tensor::zeros({seq, head_dim});
    for (int64_t pos = 0; pos < seq; ++pos) {
        for (int i = 0; i < half; ++i) {
            const float inv_freq = 1.0f / std::pow(
                kRopeTheta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(pos) * inv_freq;
            cos_t.data[pos * head_dim + i] = std::cos(angle);
            cos_t.data[pos * head_dim + i + half] = std::cos(angle);
            sin_t.data[pos * head_dim + i] = std::sin(angle);
            sin_t.data[pos * head_dim + i + half] = std::sin(angle);
        }
    }

    atlas::DeviceTensor dx = atlas::DeviceTensor::alloc(x.shape);
    atlas::DeviceTensor dcos = atlas::DeviceTensor::alloc({seq, head_dim});
    atlas::DeviceTensor dsin = atlas::DeviceTensor::alloc({seq, head_dim});
    atlas::to_device(x, dx);
    atlas::to_device(cos_t, dcos);
    atlas::to_device(sin_t, dsin);

    atlas::rope(x, cos_t, sin_t, n_heads, head_dim);  // x becomes the reference

    atlas::launch_rope(dx, dcos, dsin, n_heads, head_dim);
    atlas::Tensor got = atlas::Tensor::zeros(x.shape);
    atlas::to_host(dx, got);

    const Diff d = compare(got, x);
    std::printf("  rope   %-10s [seq=%lld heads=%d hd=%d]  max-abs=%.3g mean-abs=%.3g (tol=%.1g)\n",
                name, static_cast<long long>(seq), n_heads, head_dim,
                d.max_abs, d.mean_abs, kRopeTol);
    CHECK(d.max_abs <= kRopeTol);
}

int main() {
    std::printf("test_kernels: utility kernels vs their CPU oracles (docs/10)\n");

    // embed — bit-exact row gather.
    run_embed("small", 100, 64, 4, 0x01u);
    run_embed("tinyllama", 32000, 2048, 8, 0x02u);

    // add — bit-exact in-place residual.
    run_add("small", 4, 64, 0x03u);
    run_add("hidden", 8, 2048, 0x04u);

    // swiglu — __expf rounding only.
    run_swiglu("small", 4, 64, 0x05u);
    run_swiglu("prefill", 8, 5632, 0x06u);
    run_swiglu("decode", 1, 5632, 0x07u);

    // rope — q-shaped (32 heads), decode, and k-shaped (4 kv heads, the GQA asymmetry).
    run_rope("q-prefill", 4, 32, 64, 0x08u);
    run_rope("q-decode", 1, 32, 64, 0x09u);
    run_rope("k-gqa", 8, 4, 64, 0x0Au);

    if (g_failures == 0) {
        std::printf("test_kernels: all checks passed\n");
        return 0;
    }
    std::printf("test_kernels: %d check(s) FAILED\n", g_failures);
    return 1;
}
