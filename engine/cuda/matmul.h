#pragma once

#include "device_tensor.h"

namespace atlas {

// The first real CUDA kernel (Phase 2 · Step 2, docs/08-cuda-matmul.md): a tiled
// shared-memory matmul that computes exactly what the proven CPU atlas::linear() computes,
// on the device.
//
//   y = x @ Wᵀ   x:[m, in]   w:[out, in] (PyTorch nn.Linear layout, NO transpose)
//                out:[m, out] — pre-allocated by the caller.
//
// The contraction walks the contiguous last dim of both operands (an NT GEMM), so Wᵀ is
// never materialized and tile loads stay coalesced. FP32 accumulation, same numeric type
// as linear(): the only GPU↔CPU difference is summation order, so the diff is small,
// nonzero, and explainable — validated against linear() within a measured-then-pinned
// tolerance (see tests/test_matmul.cu).
//
// Same shape contract / asserts as atlas::linear(): w.shape[1] == x.shape[1] (in),
// out.shape == {x.shape[0], w.shape[0]}. Boundary masking inside the kernel handles
// TinyLlama's non-tile-aligned dims (256, 2048, 5632, 32000) and the m=1 decode case.
//
// Host-includable (declaration only, like device_tensor.h); the kernel and
// <cuda_runtime.h> stay in matmul.cu.
void launch_matmul(const DeviceTensor& x, const DeviceTensor& w, DeviceTensor& out);

}  // namespace atlas
