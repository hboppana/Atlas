// Component tests for the model math building blocks — Phase 1 test hardening.
//
// linear, rmsnorm, swiglu vs hand-computed oracles on tiny tensors. Blob-free: these
// give blob-less machines (CI) real coverage of the model math that test_forward can
// only exercise when the 4.4 GB local blob is present. RoPE and attention have their
// own targets (test_rope, test_attention) — they carry the architecture's riskiest
// details and deserve isolated failure reports.

#include <cmath>
#include <cstdio>

#include "../include/model.h"
#include "../include/tensor.h"

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

static bool near(float a, double b, double tol = 1e-6) {
    return std::fabs(static_cast<double>(a) - b) <= tol;
}

// y = x @ Wᵀ with small integers — exactly representable, so equality is exact.
static void test_linear() {
    std::printf("test_model_math: linear (y = x @ W^T) vs hand-computed\n");
    atlas::Tensor x = atlas::Tensor::zeros({2, 3});
    atlas::Tensor w = atlas::Tensor::zeros({2, 3});  // [out, in], PyTorch layout
    const float xv[2][3] = {{1, 2, 3}, {4, 5, 6}};
    const float wv[2][3] = {{1, 0, -1}, {2, 1, 0}};
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t j = 0; j < 3; ++j) {
            x.at({i, j}) = xv[i][j];
            w.at({i, j}) = wv[i][j];
        }
    atlas::Tensor y = atlas::Tensor::zeros({2, 2});
    atlas::linear(x, w, y);
    // Row 0: [1,2,3]·[1,0,-1] = -2,  [1,2,3]·[2,1,0] = 4
    // Row 1: [4,5,6]·[1,0,-1] = -2,  [4,5,6]·[2,1,0] = 13
    CHECK(y.at({0, 0}) == -2.0f);
    CHECK(y.at({0, 1}) == 4.0f);
    CHECK(y.at({1, 0}) == -2.0f);
    CHECK(y.at({1, 1}) == 13.0f);
}

static void test_rmsnorm() {
    std::printf("test_model_math: rmsnorm vs hand-computed\n");
    atlas::Tensor x = atlas::Tensor::zeros({2, 4});
    atlas::Tensor w = atlas::Tensor::zeros({4});
    // Row 0: [3,4,0,0] -> mean(x^2) = 25/4 = 6.25, rms = 2.5 (exact), scale = 0.4.
    const float r0[4] = {3, 4, 0, 0};
    for (int64_t j = 0; j < 4; ++j) {
        x.at({0, j}) = r0[j];
        w.at({j}) = static_cast<float>(j + 1);  // w = [1,2,3,4]
    }
    // Row 1: all zeros — with eps > 0 this must be well-defined (scale = 1/sqrt(eps)),
    // and 0 * anything = 0. No NaN.
    atlas::Tensor out = atlas::Tensor::zeros({2, 4});
    atlas::rmsnorm(x, w, 0.0f, out);
    CHECK(near(out.at({0, 0}), 3 * 0.4 * 1));  // 1.2
    CHECK(near(out.at({0, 1}), 4 * 0.4 * 2));  // 3.2
    CHECK(near(out.at({0, 2}), 0.0));
    CHECK(near(out.at({0, 3}), 0.0));

    atlas::rmsnorm(x, w, 1.0f, out);  // eps dominates the zero row: scale = 1, out = 0
    for (int64_t j = 0; j < 4; ++j) CHECK(out.at({1, j}) == 0.0f);
    // And eps shifts the non-zero row the documented way: scale = 1/sqrt(6.25 + 1).
    CHECK(near(out.at({0, 0}), 3.0 / std::sqrt(7.25) * 1.0, 1e-6));

    // The RMSNorm-not-LayerNorm discriminator: a constant row is NOT zeroed (no mean
    // subtraction). x = [c,c,c,c] -> rms = |c|, out = sign(c) * w.
    atlas::Tensor xc = atlas::Tensor::zeros({1, 4});
    for (int64_t j = 0; j < 4; ++j) xc.at({0, j}) = 5.0f;
    atlas::Tensor outc = atlas::Tensor::zeros({1, 4});
    atlas::rmsnorm(xc, w, 0.0f, outc);
    for (int64_t j = 0; j < 4; ++j) CHECK(near(outc.at({0, j}), static_cast<double>(j + 1)));
}

static void test_swiglu() {
    std::printf("test_model_math: swiglu (SiLU(gate) * up, in place) vs closed form\n");
    atlas::Tensor gate = atlas::Tensor::zeros({1, 4});
    atlas::Tensor up = atlas::Tensor::zeros({1, 4});
    const float gv[4] = {0.0f, 1.0f, -1.0f, 2.0f};
    for (int64_t j = 0; j < 4; ++j) {
        gate.at({0, j}) = gv[j];
        up.at({0, j}) = 3.0f;
    }
    atlas::swiglu(gate, up);
    for (int64_t j = 0; j < 4; ++j) {
        const double z = gv[j];
        const double expect = z / (1.0 + std::exp(-z)) * 3.0;  // SiLU(z) * up
        CHECK(near(gate.at({0, j}), expect));
        CHECK(up.at({0, j}) == 3.0f);  // up is read-only; gate is the in-place output
    }
    // SiLU(0) = 0 exactly — the gate kills the channel regardless of up.
    CHECK(gate.at({0, 0}) == 0.0f);
}

int main() {
    test_linear();
    test_rmsnorm();
    test_swiglu();

    if (g_failures == 0) {
        std::printf("test_model_math: all checks passed\n");
        return 0;
    }
    std::printf("test_model_math: %d check(s) FAILED\n", g_failures);
    return 1;
}
