#include "rmsnorm.h"

#include <cassert>

#include <cuda_runtime.h>

#include "cuda_check.h"

namespace atlas {

// Fused block-per-row RMSNorm, computing exactly what atlas::rmsnorm() does (model.cpp):
// out[i][j] = x[i][j] · rsqrt(mean_j(x²) + eps) · w[j], no mean subtraction.
//
// One block owns one row: threads stride over the row accumulating a partial sum-of-squares
// in a register (coalesced loads — consecutive threads read consecutive j), then a two-level
// reduction (warp-shuffle -> shared -> first warp) collapses them to the row's `ss`. FP32
// accumulation matches rmsnorm()'s `float ss`, so the only GPU↔CPU difference is the
// reduction *order* (plus rsqrtf vs 1/sqrt). Step 3 is the correct fused *baseline*; the perf
// levers (block-size sweep, float4 loads, norm-matmul fusion, TF32) are the named follow-up
// in docs/09-cuda-rmsnorm.md.

namespace {

constexpr int BLOCK = 256;
constexpr int WARP = 32;

// Reduce `val` across the whole block to a single sum, returned to every thread. Warp-level
// shuffle first, one partial per warp into shared, then the first warp reduces those.
__device__ float block_reduce_sum(float val, float* warp_sums) {
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;

    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);

    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();

    // First warp reduces the per-warp partials. BLOCK/WARP warps (=8 for BLOCK=256), so the
    // first warp's lanes cover them; lanes past the warp count read 0.
    constexpr int NWARPS = BLOCK / WARP;
    float total = (threadIdx.x < NWARPS) ? warp_sums[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    return total;  // valid in lane 0 of warp 0; broadcast via shared below.
}

// out[i][j] = x[i][j] · rsqrt(mean(x²)+eps) · w[j]. grid.x = m (one block per row).
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ out,
                               int64_t n, float eps) {
    __shared__ float warp_sums[BLOCK / WARP];
    __shared__ float scale_bc;  // row scale, broadcast to every thread

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const float* xrow = x + row * n;
    float* orow = out + row * n;

    // Partial sum-of-squares over this thread's strided elements. j >= n contributes 0
    // (tail masking), which is what makes a non-BLOCK-multiple n correct.
    float ss = 0.0f;
    for (int64_t j = threadIdx.x; j < n; j += BLOCK) {
        const float v = xrow[j];
        ss += v * v;
    }

    ss = block_reduce_sum(ss, warp_sums);

    if (threadIdx.x == 0)
        scale_bc = rsqrtf(ss / static_cast<float>(n) + eps);
    __syncthreads();

    const float scale = scale_bc;
    for (int64_t j = threadIdx.x; j < n; j += BLOCK)
        orow[j] = xrow[j] * scale * w[j];
}

}  // namespace

void launch_rmsnorm(const DeviceTensor& x, const DeviceTensor& w,
                    float eps, DeviceTensor& out) {
    // Same shape contract as atlas::rmsnorm().
    assert(x.shape.size() == 2 && out.shape.size() == 2);
    assert(x.shape == out.shape && "launch_rmsnorm: out shape mismatch");
    const int64_t m = x.shape[0];
    const int64_t n = x.shape[1];
    assert(w.numel() == n && "launch_rmsnorm: weight size mismatch");

    rmsnorm_kernel<<<static_cast<unsigned>(m), BLOCK>>>(x.data, w.data, out.data, n, eps);
    CUDA_CHECK_KERNEL();
}

}  // namespace atlas
