// GQA causal attention component test — Phase 1 test hardening.
//
// Tiny hand-checkable shapes isolate each contract of atlas::attention: causal
// masking, the GQA head mapping (query head hq reads kv head hq / q_per_kv — the 8:1
// asymmetry is what docs/03 flags as the likeliest attention bug), softmax weighting,
// and the seq=1 degenerate case. Blob-free.

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

// seq = 1: softmax over a single (unmasked) key is 1 regardless of scores, so every
// query head returns its kv head's v row verbatim.
static void test_seq1_returns_v() {
    std::printf("test_attention: seq=1 returns v for every query head\n");
    atlas::Tensor q = atlas::Tensor::zeros({1, 4});  // 2 heads x hd 2
    atlas::Tensor k = atlas::Tensor::zeros({1, 2});  // 1 kv head
    atlas::Tensor v = atlas::Tensor::zeros({1, 2});
    q.at({0, 0}) = 3.0f; q.at({0, 1}) = -1.0f; q.at({0, 2}) = 0.5f; q.at({0, 3}) = 9.0f;
    k.at({0, 0}) = -2.0f; k.at({0, 1}) = 4.0f;
    v.at({0, 0}) = 5.0f; v.at({0, 1}) = -3.0f;
    atlas::Tensor ctx = atlas::Tensor::zeros({1, 4});
    atlas::attention(q, k, v, /*n_heads=*/2, /*n_kv_heads=*/1, /*head_dim=*/2, ctx);
    CHECK(ctx.at({0, 0}) == 5.0f);
    CHECK(ctx.at({0, 1}) == -3.0f);
    CHECK(ctx.at({0, 2}) == 5.0f);  // second query head reads the same kv head
    CHECK(ctx.at({0, 3}) == -3.0f);
}

// Causality: row 0 may only attend to key 0, so ctx row 0 == v row 0 exactly, no
// matter what q and k contain (a future-leak puts v row 1 in there).
static void test_causal_mask() {
    std::printf("test_attention: row 0 sees only key 0 (causal mask)\n");
    atlas::Tensor q = atlas::Tensor::zeros({2, 2});
    atlas::Tensor k = atlas::Tensor::zeros({2, 2});
    atlas::Tensor v = atlas::Tensor::zeros({2, 2});
    for (int64_t j = 0; j < 2; ++j) {
        q.at({0, j}) = 7.0f; q.at({1, j}) = -7.0f;
        k.at({0, j}) = 1.0f; k.at({1, j}) = 100.0f;  // key 1 would dominate if leaked
        v.at({0, j}) = static_cast<float>(j + 1);    // v0 = [1, 2]
        v.at({1, j}) = 999.0f;
    }
    atlas::Tensor ctx = atlas::Tensor::zeros({2, 2});
    atlas::attention(q, k, v, 1, 1, 2, ctx);
    CHECK(ctx.at({0, 0}) == 1.0f);
    CHECK(ctx.at({0, 1}) == 2.0f);
}

// q = 0 makes every unmasked score 0, so softmax is uniform over keys 0..i and
// ctx row i is the running mean of v rows — hand-checkable at every position.
static void test_uniform_softmax() {
    std::printf("test_attention: zero q -> ctx row i = mean(v[0..i])\n");
    atlas::Tensor q = atlas::Tensor::zeros({3, 2});
    atlas::Tensor k = atlas::Tensor::zeros({3, 2});
    atlas::Tensor v = atlas::Tensor::zeros({3, 2});
    const float vv[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 2; ++j) {
            k.at({i, j}) = static_cast<float>(i * 2 + j);  // arbitrary: q=0 ignores k
            v.at({i, j}) = vv[i][j];
        }
    atlas::Tensor ctx = atlas::Tensor::zeros({3, 2});
    atlas::attention(q, k, v, 1, 1, 2, ctx);
    CHECK(near(ctx.at({0, 0}), 1.0));  // mean of {1}
    CHECK(near(ctx.at({0, 1}), 2.0));
    CHECK(near(ctx.at({1, 0}), 2.0));  // mean of {1, 3}
    CHECK(near(ctx.at({1, 1}), 3.0));
    CHECK(near(ctx.at({2, 0}), 3.0));  // mean of {1, 3, 5}
    CHECK(near(ctx.at({2, 1}), 4.0));
}

// The GQA mapping: with 4 query heads over 2 kv heads, heads {0,1} must read kv 0 and
// heads {2,3} kv 1. head_dim = 1 and seq = 1 make the output the raw v values, so a
// wrong mapping (e.g. hq % n_kv_heads instead of hq / q_per_kv) is immediately visible.
static void test_gqa_mapping() {
    std::printf("test_attention: GQA mapping — query heads {0,1}->kv0, {2,3}->kv1\n");
    atlas::Tensor q = atlas::Tensor::zeros({1, 4});
    atlas::Tensor k = atlas::Tensor::zeros({1, 2});
    atlas::Tensor v = atlas::Tensor::zeros({1, 2});
    for (int64_t j = 0; j < 4; ++j) q.at({0, j}) = 1.0f;
    v.at({0, 0}) = 7.0f;   // kv head 0
    v.at({0, 1}) = -9.0f;  // kv head 1
    atlas::Tensor ctx = atlas::Tensor::zeros({1, 4});
    atlas::attention(q, k, v, /*n_heads=*/4, /*n_kv_heads=*/2, /*head_dim=*/1, ctx);
    CHECK(ctx.at({0, 0}) == 7.0f);
    CHECK(ctx.at({0, 1}) == 7.0f);
    CHECK(ctx.at({0, 2}) == -9.0f);
    CHECK(ctx.at({0, 3}) == -9.0f);
}

// Softmax weighting direction: a query strongly aligned with key 1 must pull ctx to
// v row 1 (and the max-subtraction path must not change that).
static void test_peaked_softmax() {
    std::printf("test_attention: peaked scores select the aligned key's v\n");
    atlas::Tensor q = atlas::Tensor::zeros({2, 1});
    atlas::Tensor k = atlas::Tensor::zeros({2, 1});
    atlas::Tensor v = atlas::Tensor::zeros({2, 1});
    q.at({1, 0}) = 10.0f;
    k.at({0, 0}) = -1.0f;  // score -10 vs +10: softmax ~ [2e-9, 1]
    k.at({1, 0}) = 1.0f;
    v.at({0, 0}) = 100.0f;
    v.at({1, 0}) = -50.0f;
    atlas::Tensor ctx = atlas::Tensor::zeros({2, 1});
    atlas::attention(q, k, v, 1, 1, 1, ctx);
    CHECK(near(ctx.at({1, 0}), -50.0, 1e-4));
}

int main() {
    test_seq1_returns_v();
    test_causal_mask();
    test_uniform_softmax();
    test_gqa_mapping();
    test_peaked_softmax();

    if (g_failures == 0) {
        std::printf("test_attention: all checks passed\n");
        return 0;
    }
    std::printf("test_attention: %d check(s) FAILED\n", g_failures);
    return 1;
}
