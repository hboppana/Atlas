#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Assert-don't-handle on the CUDA boundary — the Phase 1 die() ethos carried to the GPU.
// A failed cudaMalloc, a bad cudaMemcpy, or a kernel launch error is a bug, not a
// recoverable condition: report file:line + the offending call + cudaGetErrorString, then
// abort. Wrap every CUDA runtime call: CUDA_CHECK(cudaMalloc(...)).
//
// CUDA-only (pulls in <cuda_runtime.h>), so this lives apart from device_tensor.h, which
// stays host-includable. Only .cu translation units include it.
#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t err__ = (expr);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA_CHECK failed %s:%d  %s\n  -> %s\n",       \
                         __FILE__, __LINE__, #expr, cudaGetErrorString(err__));  \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

// Kernel-launch errors are asynchronous: the launch itself returns void, so the only way
// to surface a bad launch (bad config, etc.) is to poll cudaGetLastError right after.
// Use immediately after a <<<>>> launch. cudaDeviceSynchronize forces completion so an
// async fault is attributed here rather than to the next CUDA call.
#define CUDA_CHECK_KERNEL()                  \
    do {                                     \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
