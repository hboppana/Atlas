// RMSNorm kernel validation — Phase 2 · Step 3 (docs/09-cuda-rmsnorm.md).
//
// The second kernel with real arithmetic, so its GPU result is not bit-exact to the CPU
// oracle: the parallel sum-of-squares reorders the FP32 additions relative to rmsnorm()'s
// sequential accumulation (plus rsqrtf vs 1/sqrt), so the diff is small-but-nonzero. Same
// measure-then-pin discipline as the matmul, but the diff's source is a *reduction reorder*
// — a single n-term sum per row, so smaller than the matmul's up-to-5632-term contraction.
//
// Oracle is atlas::rmsnorm() (model.cpp), itself validated against HuggingFace in Phase 1, so
// checking the kernel against it inherits that trust — no re-derivation. Flow per shape:
//   fill_prng x,w -> rmsnorm(x,w,eps,ref) on CPU -> to_device/launch_rmsnorm/to_host -> compare().
//
// Blob-free: random inputs, no 4.4 GB weight blob. Rides the Step-1 infra + the shared
// harness (test_harness.h), so a failure here is unambiguously a kernel bug.

#include <cstdio>

#include <cuda_runtime.h>

#include "../cuda_check.h"
#include "../device_tensor.h"
#include "../rmsnorm.h"
#include "../../include/tensor.h"
#include "../../include/model.h"  // atlas::rmsnorm oracle
#include "test_harness.h"         // CHECK + g_failures, Diff/compare, fill_prng

// TinyLlama's real RMSNorm epsilon (Config::rms_norm_eps, model.h).
constexpr float kEps = 1e-5f;

// Tolerance MEASURED on the first green run then PINNED. Worst measured case: prefill
// m=8 n=2048 at max-abs=5.36e-07 on the A6000 (CUDA 12.6). A single 2048-term reduction
// reorders far less than the matmul's up-to-5632-term contraction, so the measured number
// sits ~187x below the matmul's 1e-4 pin — and ~six orders of magnitude below a real-bug
// signal (a wrong scale/reduction is off by O(1), not 1e-6). Pinned at 1e-5 to give ~19x
// headroom over the measured worst while remaining tight against any algorithmic error.
constexpr double kMaxAbsTol = 1e-5;

// Run one shape against the rmsnorm() oracle and report max-abs/mean-abs.
static void run_case(const char* name, int64_t m, int64_t n, uint32_t seed) {
    atlas::Tensor x = atlas::Tensor::zeros({m, n});
    atlas::Tensor w = atlas::Tensor::zeros({n});
    fill_prng(x, seed);
    fill_prng(w, seed ^ 0x9E3779B9u);

    // CPU oracle.
    atlas::Tensor ref = atlas::Tensor::zeros({m, n});
    atlas::rmsnorm(x, w, kEps, ref);

    // GPU.
    atlas::DeviceTensor dx = atlas::DeviceTensor::alloc({m, n});
    atlas::DeviceTensor dw = atlas::DeviceTensor::alloc({n});
    atlas::DeviceTensor dout = atlas::DeviceTensor::alloc({m, n});
    atlas::to_device(x, dx);
    atlas::to_device(w, dw);
    atlas::launch_rmsnorm(dx, dw, kEps, dout);

    atlas::Tensor got = atlas::Tensor::zeros({m, n});
    atlas::to_host(dout, got);

    const Diff d = compare(got, ref);
    std::printf("  %-12s [m=%lld n=%lld]  max-abs=%.3g mean-abs=%.3g (tol=%.1g)\n",
                name, static_cast<long long>(m), static_cast<long long>(n),
                d.max_abs, d.mean_abs, kMaxAbsTol);
    CHECK(d.max_abs <= kMaxAbsTol);
}

int main() {
    std::printf("test_rmsnorm: fused GPU RMSNorm vs atlas::rmsnorm() oracle\n");

    // Aligned small — n a clean block multiple, no reduction tail.
    run_case("aligned", 4, 64, 0x01u);
    // Non-multiple n — exercises the reduction tail masking.
    run_case("non-mult-n", 3, 100, 0x02u);
    // Decode shape m=1 — single row, single block.
    run_case("decode-m1", 1, 2048, 0x03u);
    // Prefill shape — the real TinyLlama hidden width across a multi-row grid.
    run_case("prefill", 8, 2048, 0x10u);

    if (g_failures == 0) {
        std::printf("test_rmsnorm: all checks passed\n");
        return 0;
    }
    std::printf("test_rmsnorm: %d check(s) FAILED\n", g_failures);
    return 1;
}
