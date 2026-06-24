#pragma once

// The reusable CUDA-test harness — "the pattern every later kernel reuses" (doc-07), made
// concrete in Step 2 (docs/08-cuda-matmul.md). Lifted verbatim from test_device.cu so both
// it and test_matmul.cu share one copy: the CHECK macro + failure counter, the diff harness
// (compare() -> max-abs/mean-abs), and the deterministic PRNG fill. No behavior change.
//
// The Phase 1 method these encode: MEASURE the diff against the CPU oracle on the first green
// run, then PIN it as the gate — never guess a tolerance up front. Step 1's payloads were
// bit-exact (pinned at 0); the matmul is the first kernel with a small-but-nonzero diff.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../../include/tensor.h"

// Each test translation unit owns the failure counter (declared here, defined once per exe).
static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Element-wise diff between a GPU result and its CPU oracle, max-abs / mean-abs.
struct Diff {
    double max_abs = 0.0;
    double mean_abs = 0.0;
};

inline Diff compare(const atlas::Tensor& got, const atlas::Tensor& ref) {
    const int64_t n = got.numel();
    Diff d;
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(got.data[i]) -
                                   static_cast<double>(ref.data[i]));
        if (e > d.max_abs) d.max_abs = e;
        sum += e;
    }
    d.mean_abs = n > 0 ? sum / static_cast<double>(n) : 0.0;
    return d;
}

// Deterministic fill so runs are reproducible (Phase 1 PRNG convention). A plain LCG — no
// <random> engine dependence across platforms. Values ~[-0.5, 0.5).
inline void fill_prng(atlas::Tensor& t, uint32_t seed) {
    uint32_t s = seed;
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        t.data[i] = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) - 0.5f;
    }
}
