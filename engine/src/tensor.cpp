#include "../include/tensor.h"

#include <cassert>
#include <cstring>
#include <utility>

namespace atlas {

namespace {

// Row-major contiguous strides for a shape: strides[i] = product of dims after i.
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

// Flat offset into `data` for a multi-dim index, honoring strides (so it is correct
// for views as well as contiguous buffers). Rank/bounds checked in debug builds.
int64_t flat_offset(const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& strides,
                    std::initializer_list<int64_t> idx) {
    assert(idx.size() == strides.size() && "index rank mismatch");
    int64_t offset = 0;
    size_t i = 0;
    for (int64_t v : idx) {
        assert(v >= 0 && v < shape[i] && "index out of bounds");
        offset += v * strides[i];
        ++i;
    }
    (void)shape;  // only read inside asserts
    return offset;
}

// a is [m,n]; b is a row vector broadcastable across a's rows ([n] or [1,n]).
bool is_row_broadcast(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2) return false;
    const int64_t n = a.shape[1];
    if (b.shape.size() == 1) return b.shape[0] == n;
    if (b.shape.size() == 2) return b.shape[0] == 1 && b.shape[1] == n;
    return false;
}

// Elementwise apply with optional row broadcast. Assumes contiguous a/b/out, which
// holds for owned tensors; strided broadcasting beyond this arrives if a later step
// needs it.
template <typename Op>
void elementwise(const Tensor& a, const Tensor& b, Tensor& out, Op op) {
    assert(a.shape == out.shape && "elementwise: out shape must match a");
    if (a.shape == b.shape) {
        const int64_t total = a.numel();
        for (int64_t i = 0; i < total; ++i) out.data[i] = op(a.data[i], b.data[i]);
    } else if (is_row_broadcast(a, b)) {
        const int64_t m = a.shape[0];
        const int64_t n = a.shape[1];
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                out.data[i * n + j] = op(a.data[i * n + j], b.data[j]);
            }
        }
    } else {
        assert(false && "elementwise: unsupported operand shapes");
    }
}

}  // namespace

Tensor::~Tensor() {
    if (owns) delete[] data;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data),
      shape(std::move(other.shape)),
      strides(std::move(other.strides)),
      owns(other.owns) {
    other.data = nullptr;
    other.owns = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns) delete[] data;
        data = other.data;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        owns = other.owns;
        other.data = nullptr;
        other.owns = false;
    }
    return *this;
}

Tensor Tensor::zeros(std::vector<int64_t> shape) {
    Tensor t;
    t.shape = std::move(shape);
    t.strides = contiguous_strides(t.shape);
    t.owns = true;
    const int64_t n = numel_of(t.shape);
    t.data = new float[static_cast<size_t>(n)];
    std::memset(t.data, 0, static_cast<size_t>(n) * sizeof(float));
    return t;
}

Tensor Tensor::view(float* ptr, std::vector<int64_t> shape) {
    Tensor t;
    t.shape = std::move(shape);
    t.strides = contiguous_strides(t.shape);
    t.owns = false;
    t.data = ptr;
    return t;
}

int64_t Tensor::numel() const {
    return numel_of(shape);
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    assert(numel_of(new_shape) == numel() && "reshape must preserve numel");
    return Tensor::view(data, std::move(new_shape));
}

float& Tensor::at(std::initializer_list<int64_t> idx) {
    return data[flat_offset(shape, strides, idx)];
}

const float& Tensor::at(std::initializer_list<int64_t> idx) const {
    return data[flat_offset(shape, strides, idx)];
}

void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    assert(a.shape.size() == 2 && b.shape.size() == 2 && out.shape.size() == 2 &&
           "matmul operands must be 2-D");
    const int64_t m = a.shape[0];
    const int64_t k = a.shape[1];
    const int64_t n = b.shape[1];
    assert(b.shape[0] == k && "matmul inner dim mismatch");
    assert(out.shape[0] == m && out.shape[1] == n && "matmul out shape mismatch");
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                acc += a.data[i * a.strides[0] + p * a.strides[1]] *
                       b.data[p * b.strides[0] + j * b.strides[1]];
            }
            out.data[i * out.strides[0] + j * out.strides[1]] = acc;
        }
    }
}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    elementwise(a, b, out, [](float x, float y) { return x + y; });
}

void mul(const Tensor& a, const Tensor& b, Tensor& out) {
    elementwise(a, b, out, [](float x, float y) { return x * y; });
}

}  // namespace atlas
