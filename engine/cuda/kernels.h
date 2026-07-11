#pragma once

#include "device_tensor.h"

namespace atlas {

// The four remaining utility kernels (Phase 2 · Step 4, docs/10-cuda-kernels.md): the
// element-wise and gather primitives the TinyLlama forward pass needs beyond matmul and
// RMSNorm. All four are FP32-in / FP32-out, unfused, and validated against their proven
// CPU oracles (tests/test_kernels.cu) — after this step the only missing GPU primitive
// is attention (Step 5).
//
// Host-includable (declaration only, like device_tensor.h / matmul.h / rmsnorm.h); the
// __global__ kernels and <cuda_runtime.h> stay in kernels.cu.

// Token embedding gather — the inline row-gather in Model::forward() (model.cpp):
//
//   out[i, :] = embed[ids[i], :]   for i in [0, seq)
//
//   embed : [vocab, H]  pre-uploaded weight, row-major
//   ids_d : device int32 array of length seq (pre-uploaded by the caller — DeviceTensor
//           is float-typed, so the id buffer is a raw cudaMalloc; Step 6 wraps this)
//   out   : [seq, H]    pre-allocated
//
// Integer-indexed row copy, no arithmetic — bit-exact vs the CPU path (pinned max-abs = 0).
void launch_embed(const DeviceTensor& embed, const int* ids_d,
                  int64_t seq, DeviceTensor& out);

// Element-wise in-place residual add — atlas::add() (tensor.cpp) as the forward pass
// always calls it, add(h, other, h):
//
//   a[i] += b[i]   (same shape, flat)
//
// Straight FP32 addition with no reordering — bit-exact vs the CPU path (pinned max-abs = 0).
void launch_add(DeviceTensor& a, const DeviceTensor& b);

// SwiGLU gating in place — atlas::swiglu() (model.cpp):
//
//   gate[i] = SiLU(gate[i]) * up[i]   where SiLU(z) = z / (1 + exp(-z))
//
// `gate` is read-write, `up` read-only, same shape. Uses __expf on the device, so the
// GPU↔CPU diff is __expf vs std::exp rounding — measured then pinned.
void launch_swiglu(DeviceTensor& gate, const DeviceTensor& up);

// RoPE in place — atlas::rope() (model.cpp), HF Llama's NeoX half-split rotation (NOT
// interleaved adjacent pairs). For each (pos, head), with half = head_dim/2:
//
//   x[i]        = x0·cos[pos,i]        - x1·sin[pos,i]          x0 = x[i]
//   x[i + half] = x1·cos[pos,i + half] + x0·sin[pos,i + half]   x1 = x[i + half]
//
//   x        : [seq, n_heads·head_dim]  q or k (GQA: n_heads differs, 32 vs 4)
//   cos, sin : [seq, head_dim]          precomputed by the forward() preamble, read-only;
//              angles duplicated across halves, so no in-kernel trig — pure table lookup.
void launch_rope(DeviceTensor& x, const DeviceTensor& cos, const DeviceTensor& sin,
                 int n_heads, int head_dim);

}  // namespace atlas
