#pragma once

#include <string>
#include <vector>

#include "tensor.h"

namespace atlas {

// Zero-dependency NPY v1.0 reader — the measuring stick for validation against the
// reference/ oracles (Step 4) and every later accuracy comparison (Step 5 quantize
// delta, Phase 2 CUDA kernels).
//
// Supports exactly what the oracles need and fails loudly on anything else:
// version 1.0 only, little-endian '<f4' / '<i4', C-order. Read-only — no writer
// until something needs one.

// .npy of descr '<f4' -> owning FP32 Tensor with the file's shape.
Tensor load_npy_f32(const std::string& path);

// .npy of descr '<i4' -> ids, flattened in C-order.
std::vector<int> load_npy_i32(const std::string& path);

}  // namespace atlas
