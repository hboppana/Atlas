#pragma once

#include "device_tensor.h"

namespace atlas {

// The second real CUDA kernel (Phase 2 · Step 3, docs/09-cuda-rmsnorm.md): a fused
// single-pass block-per-row reduction RMSNorm that computes exactly what the proven CPU
// atlas::rmsnorm() computes (model.cpp), on the device.
//
//   out[i][j] = x[i][j] · rsqrt( mean_j(x[i][j]²) + eps ) · w[j]
//
//   x   : [m, n]  row-major, each row contiguous in n
//   w   : [n]     per-feature gain, shared across all rows
//   out : [m, n]  same shape as x — pre-allocated by the caller.
//
// No mean subtraction (this is RMSNorm, not LayerNorm). One thread block per row reduces
// that row's sum-of-squares via a warp-shuffle + shared two-level reduction; FP32
// accumulation matches rmsnorm()'s `float ss`. The only GPU↔CPU difference is reduction
// order (plus rsqrtf vs 1/sqrt), so the diff is small, nonzero, and explainable —
// validated against rmsnorm() within a measured-then-pinned tolerance (tests/test_rmsnorm.cu).
//
// Same shape contract / asserts as atlas::rmsnorm(): x.shape == out.shape, 2-D,
// w.numel() == n. Tail masking inside the kernel handles a non-BLOCK-multiple n (e.g.
// TinyLlama's n=2048) and the m=1 decode case.
//
// Host-includable (declaration only, like device_tensor.h / matmul.h); the kernel and
// <cuda_runtime.h> stay in rmsnorm.cu.
void launch_rmsnorm(const DeviceTensor& x, const DeviceTensor& w,
                    float eps, DeviceTensor& out);

}  // namespace atlas
