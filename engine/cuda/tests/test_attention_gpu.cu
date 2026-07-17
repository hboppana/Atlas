// Fused attention kernel validation — Phase 2 · Step 5 (docs/11-cuda-attention.md).
//
// The last GPU primitive, and the first with two reductions per query row. The GPU↔CPU
// diff sources are, by construction: FMA contraction of the score dots, __expf vs
// std::exp, and the denom-sum reduction reorder (one ≤seq-term sum — the rmsnorm-class
// diff). Phase 3's weighted-V accumulation preserves the oracle's j-order, so it adds
// only FMA contraction. Same measure-then-pin discipline as Steps 2–4.
//
// Oracle is atlas::attention() (model.cpp), itself validated against HuggingFace in
// Phase 1 (test_attention), so checking the kernel against it inherits that trust. Flow
// per shape:
//   fill_prng q,k,v -> attention() on CPU -> to_device x3 / launch_attention / to_host
//   -> compare().
// ctx is an out-param on both sides, so there is no upload-before-oracle ordering
// concern (unlike Step 4's in-place kernels).
//
// Named test_attention_gpu — Phase 1 already owns the CTest name test_attention
// (engine/CMakeLists.txt), and build-cuda/ builds both suites. Blob-free: random inputs.

#include <cstdio>

#include <cuda_runtime.h>

#include "../cuda_check.h"
#include "../device_tensor.h"
#include "../attention.h"
#include "../../include/tensor.h"
#include "../../include/model.h"  // atlas::attention oracle
#include "test_harness.h"         // CHECK + g_failures, Diff/compare, fill_prng

// Tolerance MEASURED on the first green run then PINNED. Worst measured cases: prefill
// and long-seq, both max-abs=8.94e-08 on the A6000 (CUDA 12.6) — softmax probabilities
// are ≤ 1 and the values ~[-0.5, 0.5), so the absolute error sits below even rmsnorm's
// 5.36e-07 despite the extra __expf. Decode measured exactly 0: a single-key softmax is
// exactly 1.0 (__expf(0)/1), making ctx a bit-exact copy of v's row on both paths.
// Pinned at 1e-6 — ~11x headroom over the measured worst, and still ~five orders of
// magnitude below a real-bug signal (a wrong mask/head mapping shifts outputs by
// O(0.1), not 1e-7).
constexpr double kMaxAbsTol = 1e-6;

// Run one shape against the attention() oracle and report max-abs/mean-abs.
static void run_case(const char* name, int64_t seq, int n_heads, int n_kv_heads,
                     int head_dim, uint32_t seed) {
    const int64_t qd = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t kvd = static_cast<int64_t>(n_kv_heads) * head_dim;
    atlas::Tensor q = atlas::Tensor::zeros({seq, qd});
    atlas::Tensor k = atlas::Tensor::zeros({seq, kvd});
    atlas::Tensor v = atlas::Tensor::zeros({seq, kvd});
    fill_prng(q, seed);
    fill_prng(k, seed ^ 0x9E3779B9u);
    fill_prng(v, seed ^ 0x85EBCA6Bu);

    // CPU oracle.
    atlas::Tensor ref = atlas::Tensor::zeros({seq, qd});
    atlas::attention(q, k, v, n_heads, n_kv_heads, head_dim, ref);

    // GPU.
    atlas::DeviceTensor dq = atlas::DeviceTensor::alloc({seq, qd});
    atlas::DeviceTensor dk = atlas::DeviceTensor::alloc({seq, kvd});
    atlas::DeviceTensor dv = atlas::DeviceTensor::alloc({seq, kvd});
    atlas::DeviceTensor dctx = atlas::DeviceTensor::alloc({seq, qd});
    atlas::to_device(q, dq);
    atlas::to_device(k, dk);
    atlas::to_device(v, dv);
    atlas::launch_attention(dq, dk, dv, n_heads, n_kv_heads, head_dim, dctx);

    atlas::Tensor got = atlas::Tensor::zeros({seq, qd});
    atlas::to_host(dctx, got);

    const Diff d = compare(got, ref);
    std::printf("  %-10s [seq=%lld heads=%d/%d hd=%d]  max-abs=%.3g mean-abs=%.3g (tol=%.1g)\n",
                name, static_cast<long long>(seq), n_heads, n_kv_heads, head_dim,
                d.max_abs, d.mean_abs, kMaxAbsTol);
    CHECK(d.max_abs <= kMaxAbsTol);
}

int main() {
    std::printf("test_attention_gpu: fused GPU attention vs atlas::attention() oracle\n");

    // Tiny GQA (q_per_kv=2), hand-checkable dims.
    run_case("small-gqa", 4, 4, 2, 16, 0x01u);
    // q_per_kv=1 — the no-GQA (MHA) edge.
    run_case("mha-edge", 4, 2, 2, 16, 0x02u);
    // TinyLlama real dims, the 8:1 asymmetry.
    run_case("prefill", 8, 32, 4, 64, 0x03u);
    // Single query row, one key — softmax of one element.
    run_case("decode", 1, 32, 4, 64, 0x04u);
    // Mask variety + score rows spanning beyond one BLOCK of keys.
    run_case("long-seq", 128, 32, 4, 64, 0x10u);

    if (g_failures == 0) {
        std::printf("test_attention_gpu: all checks passed\n");
        return 0;
    }
    std::printf("test_attention_gpu: %d check(s) FAILED\n", g_failures);
    return 1;
}
