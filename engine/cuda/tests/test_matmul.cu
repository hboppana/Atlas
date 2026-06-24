// Matmul kernel validation — Phase 2 · Step 2 (docs/08-cuda-matmul.md).
//
// The first kernel with REAL arithmetic, so the first whose GPU result is not bit-exact to
// the CPU oracle: tiled partial-sum accumulation reorders the FP32 additions relative to
// linear()'s sequential dot product, so the diff is small-but-nonzero. This is where the
// Phase 1 measure-then-pin discipline first does real work.
//
// Oracle is atlas::linear() (model.cpp), itself validated against HuggingFace in Phase 1, so
// checking the kernel against it inherits that trust — no re-derivation. Flow per shape:
//   fill_prng x,w -> linear(x,w,ref) on CPU -> to_device/launch_matmul/to_host -> compare().
//
// Blob-free: random inputs, no 4.4 GB weight blob. Rides the Step-1 infra + the shared
// harness (test_harness.h), so a failure here is unambiguously a kernel bug.

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "../cuda_check.h"
#include "../device_tensor.h"
#include "../matmul.h"
#include "../../include/tensor.h"
#include "../../include/model.h"  // atlas::linear oracle
#include "test_harness.h"         // CHECK + g_failures, Diff/compare, fill_prng

// Tolerance, MEASURED on the first green run then PINNED. FP32 reordered summation grows with
// the contraction length `in`; the worst measured case is mlp_down (in=5632) at max-abs
// 1.14e-5 on the A6000 — the widest contraction TinyLlama has, so essentially the ceiling.
// Pinned at 1e-4: ~9x headroom over the measured worst, yet ~four orders of magnitude under a
// real-bug signal (a wrong sum is off by O(output magnitude) ~ O(1), not 1e-4).
constexpr double kMaxAbsTol = 1e-4;

// Run one shape against the linear() oracle and report max-abs/mean-abs.
static void run_case(const char* name, int64_t m, int64_t in, int64_t out_f, uint32_t seed) {
    atlas::Tensor x = atlas::Tensor::zeros({m, in});
    atlas::Tensor w = atlas::Tensor::zeros({out_f, in});
    fill_prng(x, seed);
    fill_prng(w, seed ^ 0x9E3779B9u);

    // CPU oracle.
    atlas::Tensor ref = atlas::Tensor::zeros({m, out_f});
    atlas::linear(x, w, ref);

    // GPU.
    atlas::DeviceTensor dx = atlas::DeviceTensor::alloc({m, in});
    atlas::DeviceTensor dw = atlas::DeviceTensor::alloc({out_f, in});
    atlas::DeviceTensor dout = atlas::DeviceTensor::alloc({m, out_f});
    atlas::to_device(x, dx);
    atlas::to_device(w, dw);
    atlas::launch_matmul(dx, dw, dout);

    atlas::Tensor got = atlas::Tensor::zeros({m, out_f});
    atlas::to_host(dout, got);

    const Diff d = compare(got, ref);
    std::printf("  %-12s [m=%lld in=%lld out=%lld]  max-abs=%.3g mean-abs=%.3g (tol=%.1g)\n",
                name, static_cast<long long>(m), static_cast<long long>(in),
                static_cast<long long>(out_f), d.max_abs, d.mean_abs, kMaxAbsTol);
    CHECK(d.max_abs <= kMaxAbsTol);
}

int main() {
    std::printf("test_matmul: tiled GPU matmul vs atlas::linear() oracle\n");

    // Tile-aligned small case — simplest path, no masking.
    run_case("aligned", 32, 32, 32, 0x01u);
    // Non-aligned on all three dims — exercises boundary masking.
    run_case("non-aligned", 3, 50, 17, 0x02u);
    // Decode shape m=1 — the single-row degenerate grid.
    run_case("decode-m1", 1, 2048, 2048, 0x03u);

    // Representative TinyLlama shapes (modest m to stay fast, blob-free).
    run_case("q_proj", 8, 2048, 2048, 0x10u);   // square
    run_case("kv_proj", 8, 2048, 256, 0x11u);   // GQA kv_dim
    run_case("mlp_gate", 8, 2048, 5632, 0x12u); // SwiGLU inner width
    run_case("mlp_down", 8, 5632, 2048, 0x13u); // wide contraction
    run_case("lm_head", 1, 2048, 32000, 0x14u); // huge out

    if (g_failures == 0) {
        std::printf("test_matmul: all checks passed\n");
        return 0;
    }
    std::printf("test_matmul: %d check(s) FAILED\n", g_failures);
    return 1;
}
