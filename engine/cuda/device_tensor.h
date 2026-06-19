#pragma once

#include <cstdint>
#include <vector>

#include "../include/tensor.h"

namespace atlas {

// The device-memory mirror of tensor.h — the Phase 2 foundation every CUDA kernel rides
// on. It is to GPU memory what Tensor is to host memory: a row-major, FP32, contiguous
// buffer with the same owns-or-views, move-only discipline, so a cudaMalloc can't silently
// leak or double-free any more than a host new[] could.
//
// Step 1 deliberately ships no compute kernel (see docs/07-cuda-build-matmul.md). This
// primitive plus the copy helpers and CUDA_CHECK are the scaffolding; they are proven with
// a no-math round-trip before the matmul (Step 2) rides on them, so a Step 2 failure is
// unambiguously a kernel bug, not a build/copy mystery.
//
// The host Tensor stays the single source of truth for shape/stride — DeviceTensor mirrors
// them but never reinterprets layout. Only the buffer lives on the device; everything else
// (the shape/strides vectors) lives on the host, exactly like Tensor.
struct DeviceTensor {
    float* data = nullptr;          // device pointer (cudaMalloc'd if owns, else a view)
    std::vector<int64_t> shape;     // logical dimensions, row-major (host-side bookkeeping)
    std::vector<int64_t> strides;   // element strides per dim (host-side bookkeeping)
    bool owns = false;              // true => cudaFree(data) in destructor

    DeviceTensor() = default;
    ~DeviceTensor();

    // Move-only: the device pointer is unique, so copying must never duplicate a cudaMalloc
    // (the same discipline Tensor and WeightStore enforce for their owned handles).
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;
    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&& other) noexcept;

    // Owned device buffer of the given shape (cudaMalloc; contents uninitialized).
    static DeviceTensor alloc(std::vector<int64_t> shape);
    // Non-owning window over an existing device buffer (caller keeps it alive). For the
    // eventual slice of the mmap'd weights' device copy, mirroring Tensor::view.
    static DeviceTensor view(float* dev_ptr, std::vector<int64_t> shape);

    int64_t numel() const;
};

// Host<->device copies, out-param style like the tensor.h ops. The destination must be
// pre-sized (alloc / zeros) by the caller; numel must match — these move bytes, never
// reshape. The host Tensor is the source of truth for shape/stride.
void to_device(const Tensor& host, DeviceTensor& dev);  // H2D cudaMemcpy
void to_host(const DeviceTensor& dev, Tensor& host);    // D2H cudaMemcpy

// No-math payload kernel launcher (Step 1 only): d[i] *= k over numel elements. k == 1.0f
// is a bit-exact identity (proves the round-trip), k != 1.0f gives the harness a non-zero
// expected output to diff. Carries no algorithmic risk — a failure means infrastructure,
// not a kernel. Replaced by real kernels (matmul, ...) from Step 2 on.
void launch_scale(DeviceTensor& dev, float k);

}  // namespace atlas
