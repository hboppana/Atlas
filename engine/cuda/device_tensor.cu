#include "device_tensor.h"

#include <cassert>
#include <utility>

#include <cuda_runtime.h>

#include "cuda_check.h"

namespace atlas {

namespace {

// Row-major contiguous strides for a shape: strides[i] = product of dims after i.
// Mirrors the tensor.cpp-local helper (kept local there, replicated here) so device
// buffers carry the same layout bookkeeping as their host counterparts.
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t acc = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = acc;
        acc *= shape[static_cast<size_t>(i)];
    }
    return strides;
}

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

}  // namespace

DeviceTensor::~DeviceTensor() {
    // cudaFree(nullptr) is a no-op, but the owns guard keeps views from freeing memory they
    // don't own — the tensor.cpp discipline, on the device.
    if (owns) CUDA_CHECK(cudaFree(data));
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept
    : data(other.data),
      shape(std::move(other.shape)),
      strides(std::move(other.strides)),
      owns(other.owns) {
    other.data = nullptr;
    other.owns = false;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
    if (this != &other) {
        if (owns) CUDA_CHECK(cudaFree(data));
        data = other.data;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        owns = other.owns;
        other.data = nullptr;
        other.owns = false;
    }
    return *this;
}

DeviceTensor DeviceTensor::alloc(std::vector<int64_t> shape) {
    DeviceTensor t;
    t.shape = std::move(shape);
    t.strides = contiguous_strides(t.shape);
    t.owns = true;
    const int64_t n = numel_of(t.shape);
    CUDA_CHECK(cudaMalloc(&t.data, static_cast<size_t>(n) * sizeof(float)));
    return t;
}

DeviceTensor DeviceTensor::view(float* dev_ptr, std::vector<int64_t> shape) {
    DeviceTensor t;
    t.shape = std::move(shape);
    t.strides = contiguous_strides(t.shape);
    t.owns = false;
    t.data = dev_ptr;
    return t;
}

int64_t DeviceTensor::numel() const {
    return numel_of(shape);
}

void to_device(const Tensor& host, DeviceTensor& dev) {
    assert(host.numel() == dev.numel() && "to_device: numel mismatch");
    CUDA_CHECK(cudaMemcpy(dev.data, host.data,
                          static_cast<size_t>(host.numel()) * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void to_host(const DeviceTensor& dev, Tensor& host) {
    assert(host.numel() == dev.numel() && "to_host: numel mismatch");
    CUDA_CHECK(cudaMemcpy(host.data, dev.data,
                          static_cast<size_t>(dev.numel()) * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// --- No-math payload (Step 1 only) ----------------------------------------------------
// A grid-stride loop touches every element, so the launch exercises the full buffer
// regardless of block/grid sizing. Pure memory traffic + one multiply; the matmul (Step 2)
// is the first kernel with real arithmetic to validate.
namespace {
__global__ void scale_kernel(float* d, int64_t n, float k) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        d[i] *= k;
    }
}
}  // namespace

void launch_scale(DeviceTensor& dev, float k) {
    const int64_t n = dev.numel();
    const int threads = 256;
    const int64_t blocks64 = (n + threads - 1) / threads;
    // Cap grid width; the grid-stride loop covers the remainder.
    const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
    scale_kernel<<<blocks, threads>>>(dev.data, n, k);
    CUDA_CHECK_KERNEL();
}

}  // namespace atlas
