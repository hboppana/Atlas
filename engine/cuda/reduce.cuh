#pragma once

// Shared warp-shuffle block reductions (Phase 2 · Step 5, docs/11-cuda-attention.md): the
// two-level sum reduction lifted from rmsnorm.cu's anonymous namespace, plus the max
// variant attention needs — same shape, fmaxf / -INFINITY in place of + / 0. One copy
// instead of two divergent ones; test_rmsnorm's pinned tolerance guards the extraction as
// behavior-identical.
//
// Device-only header (.cuh) — included by .cu translation units, never by host code.
// Templated on the block size so rmsnorm's BLOCK=256 and attention's BLOCK=128 share one
// definition. `warp_partials` is caller-provided shared memory of >= BLOCK/32 floats; the
// result is valid in lane 0 of warp 0 — callers broadcast via shared memory. Both entry
// points contain a __syncthreads(), so every thread of the block must reach the call.

#include <cmath>

namespace atlas {

// Reduce `val` across the whole block to a single sum. Warp-level shuffle first, one
// partial per warp into shared, then the first warp reduces those.
template <int BLOCK>
__device__ float block_reduce_sum(float val, float* warp_partials) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;

    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);

    if (lane == 0) warp_partials[warp] = val;
    __syncthreads();

    // First warp reduces the per-warp partials; lanes past the warp count read the
    // identity element.
    constexpr int NWARPS = BLOCK / WARP;
    float total = (threadIdx.x < NWARPS) ? warp_partials[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    return total;
}

// Same two-level shape with fmaxf / -INFINITY. Exact regardless of reduction order — max
// is order-insensitive — so this reduction contributes no GPU↔CPU diff.
template <int BLOCK>
__device__ float block_reduce_max(float val, float* warp_partials) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;

    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));

    if (lane == 0) warp_partials[warp] = val;
    __syncthreads();

    constexpr int NWARPS = BLOCK / WARP;
    float total = (threadIdx.x < NWARPS) ? warp_partials[threadIdx.x] : -INFINITY;
    if (warp == 0) {
        #pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1)
            total = fmaxf(total, __shfl_down_sync(0xffffffffu, total, offset));
    }
    return total;
}

}  // namespace atlas
