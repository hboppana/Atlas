#include "attention.h"

#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

#include "cuda_check.h"
#include "reduce.cuh"

namespace atlas {

// Fused causal GQA attention, one block per (query position i, query head hq) pair —
// grid (seq, n_heads). Three phases over the shared-memory score row, __syncthreads()
// between them (docs/11-cuda-attention.md):
//
//   1. Scores — thread-per-key: thread t computes the full head_dim-term dot
//      q[i,hq,:]·k[j,hkv,:]/√hd for its keys j = t, t+BLOCK, … ≤ i. Causality is by
//      construction (j > i never computed); the dot's d-loop is sequential per thread,
//      same order as the oracle.
//   2. Softmax — block max reduction (exact: max is order-insensitive), __expf, block
//      sum reduction → denom. The sum reorder is the rmsnorm-class diff.
//   3. Weighted V — thread-per-d, each accumulating Σ_j (scores[j]/denom)·v[j,hkv,d] in
//      a register, deliberately *not* a parallel reduction: the same j-order as the
//      oracle's loop, so the only phase-3 diff is FMA contraction.
//
// Both block reductions come from the shared reduce.cuh (the sum lifted from rmsnorm.cu,
// the max variant added for attention) — one copy, guarded by test_rmsnorm's pinned
// tolerance.

namespace {

constexpr int BLOCK = 128;
constexpr int WARP = 32;
constexpr int NWARPS = BLOCK / WARP;

// ctx[i,hq,:] = softmax(q[i,hq,:]·kᵀ/√hd, causal) · v. grid = (seq, n_heads); dynamic
// shared memory holds the score row (seq floats; only [0, i] is used).
__global__ void attention_kernel(const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 float* __restrict__ ctx,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 float inv_sqrt_hd) {
    extern __shared__ float scores[];
    __shared__ float warp_partials[NWARPS];
    __shared__ float bc;  // broadcast slot: row max, then denom

    const int64_t i = static_cast<int64_t>(blockIdx.x);  // query position
    const int hq = static_cast<int>(blockIdx.y);         // query head
    const int hkv = hq / (n_heads / n_kv_heads);
    const int64_t qd = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t kvd = static_cast<int64_t>(n_kv_heads) * head_dim;
    const float* qi = q + i * qd + static_cast<int64_t>(hq) * head_dim;
    const int64_t kv_off = static_cast<int64_t>(hkv) * head_dim;

    // Phase 1: scores. Threads with no key (threadIdx.x > i) skip the loop and carry the
    // -INFINITY / 0 identities into the reductions — tail masking.
    float local_max = -INFINITY;
    for (int64_t j = threadIdx.x; j <= i; j += BLOCK) {
        const float* kj = k + j * kvd + kv_off;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
        const float sc = dot * inv_sqrt_hd;
        scores[j] = sc;
        local_max = fmaxf(local_max, sc);
    }

    const float m = block_reduce_max<BLOCK>(local_max, warp_partials);
    if (threadIdx.x == 0) bc = m;
    __syncthreads();  // broadcasts bc; also fences warp_partials for reuse below
    const float row_max = bc;

    // Phase 2: exponentiate in place (each thread revisits its own keys) and sum.
    float local_sum = 0.0f;
    for (int64_t j = threadIdx.x; j <= i; j += BLOCK) {
        const float e = __expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }

    const float s = block_reduce_sum<BLOCK>(local_sum, warp_partials);
    if (threadIdx.x == 0) bc = s;
    __syncthreads();  // broadcasts denom; fences every scores[j] write for phase 3
    const float denom = bc;

    // Phase 3: weighted V, one thread per output element d (head_dim ≤ BLOCK, asserted
    // by the launcher). Sequential over j, same order as the oracle's accumulation loop.
    if (threadIdx.x < head_dim) {
        const int d = threadIdx.x;
        float acc = 0.0f;
        for (int64_t j = 0; j <= i; ++j) {
            const float w = scores[j] / denom;
            acc += w * v[j * kvd + kv_off + d];
        }
        ctx[i * qd + static_cast<int64_t>(hq) * head_dim + d] = acc;
    }
}

}  // namespace

void launch_attention(const DeviceTensor& q, const DeviceTensor& k, const DeviceTensor& v,
                      int n_heads, int n_kv_heads, int head_dim, DeviceTensor& ctx) {
    // Same shape contract as atlas::attention().
    assert(q.shape.size() == 2 && k.shape.size() == 2 && v.shape.size() == 2 &&
           ctx.shape.size() == 2);
    const int64_t seq = q.shape[0];
    const int64_t qd = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t kvd = static_cast<int64_t>(n_kv_heads) * head_dim;
    assert(q.shape[1] == qd && ctx.shape[1] == qd && "launch_attention: q/ctx width mismatch");
    assert(k.shape[1] == kvd && v.shape[1] == kvd && "launch_attention: k/v width mismatch");
    assert(k.shape[0] == seq && v.shape[0] == seq && ctx.shape[0] == seq);
    assert(n_kv_heads > 0 && n_heads % n_kv_heads == 0 && "launch_attention: GQA ratio");
    // Baseline's own bounds: the score row (seq floats) must fit the 48 KB default
    // dynamic-shared-memory limit, and phase 3 needs one thread per output element.
    assert(seq <= 12288 && "launch_attention: seq exceeds shared-memory score row");
    assert(head_dim <= BLOCK && "launch_attention: head_dim exceeds block size");

    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const dim3 grid(static_cast<unsigned>(seq), static_cast<unsigned>(n_heads));
    const size_t smem = static_cast<size_t>(seq) * sizeof(float);
    attention_kernel<<<grid, BLOCK, smem>>>(q.data, k.data, v.data, ctx.data,
                                            n_heads, n_kv_heads, head_dim, inv_sqrt_hd);
    CUDA_CHECK_KERNEL();
}

}  // namespace atlas
