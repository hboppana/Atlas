#pragma once

#include "device_tensor.h"

namespace atlas {

// The last missing GPU primitive (Phase 2 · Step 5, docs/11-cuda-attention.md): fused
// causal GQA attention computing exactly what the proven CPU atlas::attention() computes
// (model.cpp), in a single launch:
//
//   hkv = hq / (n_heads / n_kv_heads)             (GQA: contiguous-block kv mapping)
//   score[j] = (q[i,hq,:] · k[j,hkv,:]) / √head_dim   for j ≤ i (causal: j > i masked)
//   prob     = softmax(score)                     (FP32, max-subtracted)
//   ctx[i,hq,:] = Σ_j prob[j] · v[j,hkv,:]
//
//   q, ctx : [seq, n_heads·head_dim]     row-major; ctx pre-allocated by the caller
//   k, v   : [seq, n_kv_heads·head_dim]  the 8:1 width asymmetry on TinyLlama
//
// This is the correctness-first fused baseline, not FlashAttention: one block owns one
// (query position, query head) pair, the full score row lives in dynamic shared memory
// (seq floats → seq ≤ 12288 under the 48 KB default; TinyLlama's 2048 is 6x under),
// softmax runs over it, then the weighted-V accumulation preserves the oracle's j-order.
// The GPU↔CPU diff sources are, by construction: FMA contraction of the dots, __expf vs
// std::exp, and the denom-sum reduction reorder — validated against attention() within a
// measured-then-pinned tolerance (tests/test_attention_gpu.cu). The online-softmax /
// K-V-tiled rewrite is the named perf follow-up, held to the same oracle.
//
// Same shape contract / asserts as atlas::attention(): q/ctx width n_heads·head_dim,
// k/v width n_kv_heads·head_dim, equal seq everywhere; plus the baseline's own shared-
// memory bound. FP32-in / FP32-out, no KV-cache decode path (Step 6+).
//
// Host-includable (declaration only, like device_tensor.h / matmul.h / rmsnorm.h); the
// kernel and <cuda_runtime.h> stay in attention.cu.
void launch_attention(const DeviceTensor& q, const DeviceTensor& k, const DeviceTensor& v,
                      int n_heads, int n_kv_heads, int head_dim, DeviceTensor& ctx);

}  // namespace atlas
