#include "kernels.h"

#include <cassert>

#include <cuda_runtime.h>

#include "cuda_check.h"

namespace atlas {

// The four utility kernels (docs/10-cuda-kernels.md): embedding gather, residual add,
// SwiGLU gating, and RoPE. Each computes exactly what its CPU oracle computes — no shared
// memory, no reductions, so unlike matmul/RMSNorm there is no reordering: embed and add
// are bit-exact, swiglu/rope differ only by device-intrinsic rounding (__expf). One
// translation unit for all four: each is 10–30 lines of device code, and splitting them
// adds link targets without benefit. Fusion with neighbors is a named perf follow-up.

namespace {

constexpr int BLOCK = 256;

// out[row, :] = embed[ids[row], :]. grid.x = seq (one block per token); threads stride
// over the H elements of the row — consecutive threads read/write consecutive elements,
// coalesced on both sides.
__global__ void embed_kernel(const float* __restrict__ embed,
                             const int* __restrict__ ids,
                             float* __restrict__ out, int64_t H) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const float* src = embed + static_cast<int64_t>(ids[row]) * H;
    float* dst = out + row * H;
    for (int64_t j = threadIdx.x; j < H; j += BLOCK)
        dst[j] = src[j];
}

// a[i] += b[i]. 1-D grid, one element per thread, coalesced.
__global__ void add_kernel(float* __restrict__ a, const float* __restrict__ b,
                           int64_t numel) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i < numel) a[i] += b[i];
}

// gate[i] = SiLU(gate[i]) * up[i] = gate[i] / (1 + exp(-gate[i])) * up[i]. Same launch
// shape as add. __expf is the FP32 device intrinsic — matches std::exp up to rounding.
__global__ void swiglu_kernel(float* __restrict__ gate, const float* __restrict__ up,
                              int64_t numel) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i < numel) {
        const float z = gate[i];
        gate[i] = z / (1.0f + __expf(-z)) * up[i];
    }
}

// NeoX half-split rotation, in place. grid.x = seq * n_heads — each block owns one
// (pos, head) pair; blockDim.x = head_dim/2 (= 32 for TinyLlama, exactly one warp), so
// thread i rotates the (i, i + half) pair. Each element is read once and written once by
// the same thread, so in-place is race-free.
__global__ void rope_kernel(float* __restrict__ x,
                            const float* __restrict__ cos_t,
                            const float* __restrict__ sin_t,
                            int n_heads, int head_dim) {
    const int half = head_dim / 2;
    const int64_t pos = static_cast<int64_t>(blockIdx.x) / n_heads;
    const int head = blockIdx.x % n_heads;

    const float* c = cos_t + pos * head_dim;
    const float* s = sin_t + pos * head_dim;
    float* v = x + pos * static_cast<int64_t>(n_heads) * head_dim
                 + static_cast<int64_t>(head) * head_dim;

    const int i = threadIdx.x;
    const float x0 = v[i];
    const float x1 = v[i + half];
    v[i] = x0 * c[i] - x1 * s[i];
    v[i + half] = x1 * c[i + half] + x0 * s[i + half];
}

// 1-D launch config for the flat element-wise kernels.
unsigned blocks_for(int64_t numel) {
    return static_cast<unsigned>((numel + BLOCK - 1) / BLOCK);
}

}  // namespace

void launch_embed(const DeviceTensor& embed, const int* ids_d,
                  int64_t seq, DeviceTensor& out) {
    assert(embed.shape.size() == 2 && out.shape.size() == 2);
    assert(out.shape[0] == seq && "launch_embed: out rows != seq");
    assert(out.shape[1] == embed.shape[1] && "launch_embed: hidden size mismatch");
    const int64_t H = embed.shape[1];

    embed_kernel<<<static_cast<unsigned>(seq), BLOCK>>>(embed.data, ids_d, out.data, H);
    CUDA_CHECK_KERNEL();
}

void launch_add(DeviceTensor& a, const DeviceTensor& b) {
    // Same shape contract as atlas::add() with out == a.
    assert(a.shape == b.shape && "launch_add: shape mismatch");
    const int64_t numel = a.numel();

    add_kernel<<<blocks_for(numel), BLOCK>>>(a.data, b.data, numel);
    CUDA_CHECK_KERNEL();
}

void launch_swiglu(DeviceTensor& gate, const DeviceTensor& up) {
    // Same shape contract as atlas::swiglu().
    assert(gate.shape == up.shape && "launch_swiglu: gate/up shape mismatch");
    const int64_t numel = gate.numel();

    swiglu_kernel<<<blocks_for(numel), BLOCK>>>(gate.data, up.data, numel);
    CUDA_CHECK_KERNEL();
}

void launch_rope(DeviceTensor& x, const DeviceTensor& cos, const DeviceTensor& sin,
                 int n_heads, int head_dim) {
    // Same shape contract as atlas::rope(), plus the cos/sin table dims it relies on
    // implicitly: at least seq rows of head_dim angles each.
    assert(x.shape.size() == 2);
    assert(x.shape[1] == static_cast<int64_t>(n_heads) * head_dim &&
           "launch_rope: x width != n_heads * head_dim");
    assert(head_dim % 2 == 0 && "launch_rope: head_dim must be even");
    const int64_t seq = x.shape[0];
    assert(cos.shape.size() == 2 && cos.shape[1] == head_dim && cos.shape[0] >= seq &&
           "launch_rope: cos table too small");
    assert(sin.shape == cos.shape && "launch_rope: sin/cos shape mismatch");

    const unsigned grid = static_cast<unsigned>(seq * n_heads);
    rope_kernel<<<grid, head_dim / 2>>>(x.data, cos.data, sin.data, n_heads, head_dim);
    CUDA_CHECK_KERNEL();
}

}  // namespace atlas
