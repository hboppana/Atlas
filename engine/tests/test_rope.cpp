// RoPE component test — Phase 1 test hardening.
//
// docs/03 calls the half-split ("NeoX") rotation "the detail most likely to be wrong":
// an interleaved-adjacent-pairs implementation produces silently diverging logits. This
// target pins the rotation against the closed form, computed independently in double —
// any layout mistake (interleave, swapped sign, wrong frequency) fails here with a
// per-element report instead of an opaque e2e diff. Blob-free.

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

static bool near(float a, double b, double tol = 1e-5) {
    return std::fabs(static_cast<double>(a) - b) <= tol;
}

// The cos/sin tables exactly as Model::forward builds them: inv_freq[i] = theta^(-2i/hd)
// for i in [0, hd/2), each row the hd/2 angles concatenated with themselves.
static void build_tables(int64_t seq, int hd, float theta, atlas::Tensor& cos_t,
                         atlas::Tensor& sin_t) {
    const int half = hd / 2;
    for (int64_t pos = 0; pos < seq; ++pos) {
        for (int i = 0; i < half; ++i) {
            const float inv_freq =
                1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(hd));
            const float angle = static_cast<float>(pos) * inv_freq;
            cos_t.at({pos, i}) = std::cos(angle);
            cos_t.at({pos, i + half}) = std::cos(angle);
            sin_t.at({pos, i}) = std::sin(angle);
            sin_t.at({pos, i + half}) = std::sin(angle);
        }
    }
}

int main() {
    const int hd = 4;       // head_dim 4 -> half = 2, two frequencies: 1 and theta^-1/2
    const float theta = 10000.0f;
    const int64_t seq = 3;

    atlas::Tensor cos_t = atlas::Tensor::zeros({seq, hd});
    atlas::Tensor sin_t = atlas::Tensor::zeros({seq, hd});
    build_tables(seq, hd, theta, cos_t, sin_t);

    // Two heads with different data — heads must rotate independently with the same
    // tables, and the per-head offset must be applied (a wrong offset mixes heads).
    const int n_heads = 2;
    const float head0[3][4] = {{0.3f, -1.2f, 2.5f, 0.7f},
                               {1.0f, 2.0f, 3.0f, 4.0f},
                               {-0.5f, 0.25f, -2.0f, 1.5f}};
    const float head1[3][4] = {{4.0f, -3.0f, 2.0f, -1.0f},
                               {0.1f, 0.2f, 0.3f, 0.4f},
                               {2.0f, 2.0f, 2.0f, 2.0f}};
    atlas::Tensor x = atlas::Tensor::zeros({seq, n_heads * hd});
    for (int64_t p = 0; p < seq; ++p)
        for (int i = 0; i < hd; ++i) {
            x.at({p, i}) = head0[p][i];
            x.at({p, hd + i}) = head1[p][i];
        }

    std::printf("test_rope: half-split rotation vs closed form (seq=3, 2 heads, hd=4)\n");
    atlas::rope(x, cos_t, sin_t, n_heads, hd);

    // Position 0 is the identity (cos = 1, sin = 0) — bit-exact.
    for (int i = 0; i < hd; ++i) {
        CHECK(x.at({0, i}) == head0[0][i]);
        CHECK(x.at({0, hd + i}) == head1[0][i]);
    }

    // Positions 1..2: the closed form, computed independently in double. Half-split:
    //   v0' = v0*cos(a0) - v2*sin(a0)     v2' = v2*cos(a0) + v0*sin(a0)
    //   v1' = v1*cos(a1) - v3*sin(a1)     v3' = v3*cos(a1) + v1*sin(a1)
    // with a_i = pos * theta^(-2i/hd). An interleaved implementation pairs (v0,v1) /
    // (v2,v3) instead and fails every element below at pos > 0.
    for (int64_t pos = 1; pos < seq; ++pos) {
        const double a0 = static_cast<double>(pos);  // theta^0 = 1
        const double a1 = static_cast<double>(pos) *
                          (1.0 / std::pow(static_cast<double>(theta), 2.0 / hd));
        const float (*heads[2])[4] = {&head0[pos], &head1[pos]};
        for (int h = 0; h < n_heads; ++h) {
            const float* v = *heads[h];
            const double e0 = v[0] * std::cos(a0) - v[2] * std::sin(a0);
            const double e1 = v[1] * std::cos(a1) - v[3] * std::sin(a1);
            const double e2 = v[2] * std::cos(a0) + v[0] * std::sin(a0);
            const double e3 = v[3] * std::cos(a1) + v[1] * std::sin(a1);
            const int64_t off = static_cast<int64_t>(h) * hd;
            CHECK(near(x.at({pos, off + 0}), e0));
            CHECK(near(x.at({pos, off + 1}), e1));
            CHECK(near(x.at({pos, off + 2}), e2));
            CHECK(near(x.at({pos, off + 3}), e3));
        }
    }

    // Rotation invariant: RoPE preserves the L2 norm of every (i, i+half) pair, so the
    // per-head norm is preserved at every position. Catches scale/sign slips generically.
    for (int64_t pos = 0; pos < seq; ++pos) {
        for (int h = 0; h < n_heads; ++h) {
            const float* orig = (h == 0) ? head0[pos] : head1[pos];
            double before = 0.0, after = 0.0;
            for (int i = 0; i < hd; ++i) {
                before += static_cast<double>(orig[i]) * orig[i];
                const float r = x.at({pos, static_cast<int64_t>(h) * hd + i});
                after += static_cast<double>(r) * r;
            }
            CHECK(std::fabs(before - after) <= 1e-5);
        }
    }

    if (g_failures == 0) {
        std::printf("test_rope: all checks passed\n");
        return 0;
    }
    std::printf("test_rope: %d check(s) FAILED\n", g_failures);
    return 1;
}
