#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace atlas {

// A lightweight, FP32, row-major tensor — the foundation the whole TinyLlama
// forward pass is built on.
//
// Memory model: a Tensor either *owns* its buffer (allocated activations, freed in
// the destructor) or *views* a buffer owned by someone else (e.g. a non-owning window
// over mmap'd weights, added in a later step). This is what lets weight loading avoid
// copying ~2 GB. Tensors are move-only so ownership is never silently duplicated.
//
// Layout is row-major and contiguous with explicit strides, so reshape/slice are free
// (no copy). Model-specific math (RMSNorm, RoPE, attention, SwiGLU) is NOT here — it is
// built from these primitives in model.cpp later. This stays memory + core ops only.
struct Tensor {
    float* data = nullptr;          // first element (owned or viewed)
    std::vector<int64_t> shape;     // logical dimensions, row-major
    std::vector<int64_t> strides;   // element strides per dim
    bool owns = false;              // true => delete[] data in destructor

    Tensor() = default;
    ~Tensor();

    // Move-only: no accidental deep copies of activation buffers.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Owned, zero-initialized buffer of the given shape.
    static Tensor zeros(std::vector<int64_t> shape);
    // Non-owning window over an existing buffer (caller keeps it alive).
    static Tensor view(float* ptr, std::vector<int64_t> shape);

    int64_t numel() const;

    // A non-owning view over the same buffer with a new contiguous shape. numel must
    // be preserved (no copy). The owning Tensor must outlive the returned view.
    Tensor reshape(std::vector<int64_t> new_shape) const;

    // Element access by multi-dim index. Rank/bounds checked in debug builds.
    float& at(std::initializer_list<int64_t> idx);
    const float& at(std::initializer_list<int64_t> idx) const;
};

// Core ops as free functions, out-param style so allocations are explicit.
// `out` must be pre-sized by the caller (e.g. via Tensor::zeros).
void matmul(const Tensor& a, const Tensor& b, Tensor& out);  // [m,k] x [k,n] -> [m,n]
void add(const Tensor& a, const Tensor& b, Tensor& out);     // elementwise (+ row broadcast)
void mul(const Tensor& a, const Tensor& b, Tensor& out);     // elementwise

}  // namespace atlas
