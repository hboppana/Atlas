// Device bring-up test — Phase 2 · Step 1 (docs/07-cuda-build-matmul.md).
//
// Proves the CUDA infrastructure with ZERO compute kernel: the nvcc build works on the
// A6000, cudaMalloc/cudaMemcpy round-trips bit-exactly, a kernel can be launched, the diff
// harness reports max-abs/mean-abs, and CUDA_CHECK aborts on an injected failure. Because
// it carries no math, any failure here is the infrastructure — not a kernel. Step 2's
// matmul then rides this proven stack, so a matmul failure is unambiguously a kernel bug.
//
// Blob-free: runs on any GPU node without the 4.4 GB weight blob. Mirrors the Phase 1
// zero-dependency CHECK harness (test_model_math) and self-subprocess death pattern
// (test_weightstore).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../cuda_check.h"
#include "../device_tensor.h"
#include "../../include/tensor.h"

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// The reusable diff harness — the pattern every later kernel reuses. Reports max-abs and
// mean-abs element-wise diff between a GPU result and its CPU oracle. The Phase 1 method:
// MEASURE the diff on the first green run, then PIN it as the gate (never guess a tolerance
// up front). Step 2 drops in linear()'s output as `ref` with no changes here.
struct Diff {
    double max_abs = 0.0;
    double mean_abs = 0.0;
};

static Diff compare(const atlas::Tensor& got, const atlas::Tensor& ref) {
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

// A deterministic fill so runs are reproducible (Phase 1 PRNG convention). A plain LCG —
// no <random> engine dependence across platforms.
static void fill_prng(atlas::Tensor& t, uint32_t seed) {
    uint32_t s = seed;
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        t.data[i] = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) - 0.5f;  // ~[-0.5,0.5)
    }
}

// Round-trip / identity: H2D -> scale(1.0) -> D2H must be BIT-EXACT to the input. A
// non-tile-aligned, multi-dim shape exercises the grid-stride loop's remainder handling.
static void test_round_trip() {
    std::printf("test_device: round-trip identity (H2D -> scale(1.0) -> D2H, bit-exact)\n");
    atlas::Tensor x = atlas::Tensor::zeros({3, 50});  // 150 elems, not warp/block aligned
    fill_prng(x, 0xC0FFEEu);

    atlas::DeviceTensor d = atlas::DeviceTensor::alloc({3, 50});
    atlas::to_device(x, d);
    atlas::launch_scale(d, 1.0f);

    atlas::Tensor y = atlas::Tensor::zeros({3, 50});
    atlas::to_host(d, y);

    int64_t exact = 0;
    for (int64_t i = 0; i < x.numel(); ++i) exact += (y.data[i] == x.data[i]);
    CHECK(exact == x.numel());  // bit-exact, every element

    const Diff diff = compare(y, x);
    std::printf("  round-trip diff: max-abs=%.3g mean-abs=%.3g (expect 0)\n",
                diff.max_abs, diff.mean_abs);
    CHECK(diff.max_abs == 0.0);
}

// Scale by a constant: a non-trivial expected output for the diff harness. CPU oracle is
// x * k. FP32 * FP32 by the same constant is exact, so this too pins at 0.
static void test_scale_diff() {
    std::printf("test_device: scale(2.0) vs CPU oracle (diff harness)\n");
    const float k = 2.0f;
    atlas::Tensor x = atlas::Tensor::zeros({128, 17});
    fill_prng(x, 0x1234u);

    atlas::Tensor ref = atlas::Tensor::zeros({128, 17});
    for (int64_t i = 0; i < x.numel(); ++i) ref.data[i] = x.data[i] * k;

    atlas::DeviceTensor d = atlas::DeviceTensor::alloc({128, 17});
    atlas::to_device(x, d);
    atlas::launch_scale(d, k);
    atlas::Tensor got = atlas::Tensor::zeros({128, 17});
    atlas::to_host(d, got);

    const Diff diff = compare(got, ref);
    std::printf("  scale diff: max-abs=%.3g mean-abs=%.3g (PIN on first green run)\n",
                diff.max_abs, diff.mean_abs);
    CHECK(diff.max_abs == 0.0);  // exact for multiply-by-constant; tighten/pin per kernel
}

// --- death case (child mode) ----------------------------------------------------------
// Proves CUDA_CHECK actually aborts on a CUDA error. Re-invokes this exe with an argument;
// the child triggers a guaranteed-failing CUDA call (D2H copy from a deliberately bad
// device pointer) wrapped in CUDA_CHECK, which must abort with non-zero exit.
static int run_death_case(const std::string& which) {
    if (which == "bad-memcpy") {
        float host = 0.0f;
        // Copy from an obviously invalid device address -> cudaErrorInvalidValue / illegal
        // access; CUDA_CHECK must fire and abort.
        CUDA_CHECK(cudaMemcpy(&host, reinterpret_cast<float*>(0x1),
                              sizeof(float), cudaMemcpyDeviceToHost));
        return 0;  // CUDA_CHECK failed to fire — parent flags this
    }
    std::fprintf(stderr, "unknown death case: %s\n", which.c_str());
    return 2;
}

static void expect_death(const char* exe, const std::string& which) {
    const std::string cmd = "\"" + std::string(exe) + "\" " + which;
    const int rc = std::system(cmd.c_str());
    std::printf("test_device: death case '%s' -> child exit %d\n", which.c_str(), rc);
    CHECK(rc != 0);  // CUDA_CHECK aborted as intended
}

int main(int argc, char** argv) {
    if (argc > 1) return run_death_case(argv[1]);  // child mode

    test_round_trip();
    test_scale_diff();
    expect_death(argv[0], "bad-memcpy");

    if (g_failures == 0) {
        std::printf("test_device: all checks passed\n");
        return 0;
    }
    std::printf("test_device: %d check(s) FAILED\n", g_failures);
    return 1;
}
