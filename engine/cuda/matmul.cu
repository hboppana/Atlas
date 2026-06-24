#include "matmul.h"

#include <cassert>

#include <cuda_runtime.h>

#include "cuda_check.h"

namespace atlas {

// Tiled shared-memory NT matmul: y = x @ Wᵀ, computing exactly what atlas::linear() does
// (model.cpp:47). The contraction `p` walks the contiguous last dim of both x ([m, in]) and
// w ([out, in]), so we consume the nn.Linear weight layout directly — no Wᵀ materialized,
// and global tile loads are coalesced (threadIdx.x indexes contiguous `p`).
//
// Step 2 is the correct tiled *baseline* (TILE=16, one output element per thread); the perf
// levers (register-blocking, float4 loads, TILE=32, TF32) are the named follow-up in
// docs/08-cuda-matmul.md. FP32 accumulation matches linear()'s `float acc`, so the only
// GPU↔CPU difference is summation order.

namespace {

constexpr int TILE = 16;

// out[i][o] = Σ_p x[i][p] · w[o][p].
//   x : [m, in]    row i contiguous in p
//   w : [out, in]  row o contiguous in p   (PyTorch nn.Linear layout, NOT transposed)
//   out : [m, out] row i contiguous in o
// Block owns a TILE×TILE tile of `out`: threadIdx.y -> row i, threadIdx.x -> col o.
__global__ void matmul_kernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ out,
                              int64_t m, int64_t in, int64_t out_f) {
    // Shared staging tiles, both indexed [row-within-tile][p-within-tile] so the inner dot
    // reads them with the same `k`, contiguous in p.
    __shared__ float xs[TILE][TILE];  // xs[ty][k] = x[i][p0+k]
    __shared__ float ws[TILE][TILE];  // ws[tx][k] = w[o][p0+k]

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int64_t i = static_cast<int64_t>(blockIdx.y) * TILE + ty;  // output row
    const int64_t o = static_cast<int64_t>(blockIdx.x) * TILE + tx;  // output col

    float acc = 0.0f;
    for (int64_t p0 = 0; p0 < in; p0 += TILE) {
        // Stage x: thread (ty,tx) loads x[i][p0+tx]. Coalesced across tx. Mask OOB -> 0.
        const int64_t xp = p0 + tx;
        xs[ty][tx] = (i < m && xp < in) ? x[i * in + xp] : 0.0f;

        // Stage w: thread (ty,tx) loads w[(blockIdx.x*TILE+ty)][p0+tx], staged at ws[ty][tx]
        // so the dot below reads ws[<output col within tile>][k]. Coalesced across tx.
        const int64_t wo = static_cast<int64_t>(blockIdx.x) * TILE + ty;
        const int64_t wp = p0 + tx;
        ws[ty][tx] = (wo < out_f && wp < in) ? w[wo * in + wp] : 0.0f;

        __syncthreads();

        // Partial dot over this tile's slice of p. The trailing (in - p0) < TILE remainder is
        // covered by the zero-padding above, so no extra bound check here.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += xs[ty][k] * ws[tx][k];

        __syncthreads();
    }

    if (i < m && o < out_f) out[i * out_f + o] = acc;
}

}  // namespace

void launch_matmul(const DeviceTensor& x, const DeviceTensor& w, DeviceTensor& out) {
    // Same shape contract as atlas::linear().
    assert(x.shape.size() == 2 && w.shape.size() == 2 && out.shape.size() == 2);
    const int64_t m = x.shape[0];
    const int64_t in = x.shape[1];
    const int64_t out_f = w.shape[0];
    assert(w.shape[1] == in && "launch_matmul: weight in_features mismatch");
    assert(out.shape[0] == m && out.shape[1] == out_f && "launch_matmul: out shape mismatch");

    const dim3 threads(TILE, TILE);
    const dim3 blocks(static_cast<unsigned>((out_f + TILE - 1) / TILE),
                      static_cast<unsigned>((m + TILE - 1) / TILE));
    matmul_kernel<<<blocks, threads>>>(x.data, w.data, out.data, m, in, out_f);
    CUDA_CHECK_KERNEL();
}

}  // namespace atlas
