// Tensor foundation tests — Phase 1, Step 1.
//
// Zero-dependency harness (a tiny CHECK macro, no GoogleTest/Catch2) per the
// from-scratch ethos. main() returns non-zero if any check fails so CTest reports
// pass/fail. Values here are hand-computed; this is the oracle for the Tensor class
// before any model code is built on top of it.

#include "tensor.h"

#include <cstdio>
#include <utility>
#include <vector>

using atlas::Tensor;

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

static bool approx(float a, float b, float eps = 1e-5f) {
    float d = a - b;
    if (d < 0.0f) d = -d;
    return d <= eps;
}

static void test_shape_stride_index() {
    Tensor t = Tensor::zeros({2, 3});
    CHECK(t.numel() == 6);
    CHECK(t.shape == std::vector<int64_t>({2, 3}));
    CHECK(t.strides == std::vector<int64_t>({3, 1}));  // row-major contiguous
    CHECK(t.owns);
    CHECK(t.at({0, 0}) == 0.0f);  // zero-initialized
    t.at({1, 2}) = 7.0f;          // write via index
    CHECK(t.at({1, 2}) == 7.0f);  // read back
    CHECK(t.data[1 * 3 + 2] == 7.0f);
}

static void test_reshape() {
    Tensor t = Tensor::zeros({2, 3});
    for (int64_t i = 0; i < 6; ++i) t.data[i] = static_cast<float>(i);
    Tensor r = t.reshape({3, 2});
    CHECK(r.numel() == 6);
    CHECK(r.shape == std::vector<int64_t>({3, 2}));
    CHECK(r.strides == std::vector<int64_t>({2, 1}));
    CHECK(!r.owns);            // reshape yields a non-owning view
    CHECK(r.data == t.data);   // no copy
    CHECK(r.at({2, 1}) == 5.0f);
}

static void test_matmul() {
    // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2)
    // A*B = [[58,64],[139,154]]
    Tensor a = Tensor::zeros({2, 3});
    const float av[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; ++i) a.data[i] = av[i];
    Tensor b = Tensor::zeros({3, 2});
    const float bv[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; ++i) b.data[i] = bv[i];
    Tensor out = Tensor::zeros({2, 2});
    atlas::matmul(a, b, out);
    CHECK(approx(out.at({0, 0}), 58.0f));
    CHECK(approx(out.at({0, 1}), 64.0f));
    CHECK(approx(out.at({1, 0}), 139.0f));
    CHECK(approx(out.at({1, 1}), 154.0f));
}

static void test_add_mul() {
    Tensor a = Tensor::zeros({2, 2});
    const float av[] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) a.data[i] = av[i];
    Tensor b = Tensor::zeros({2, 2});
    const float bv[] = {10, 20, 30, 40};
    for (int i = 0; i < 4; ++i) b.data[i] = bv[i];

    Tensor s = Tensor::zeros({2, 2});
    atlas::add(a, b, s);
    CHECK(approx(s.at({0, 0}), 11.0f));
    CHECK(approx(s.at({1, 1}), 44.0f));

    Tensor p = Tensor::zeros({2, 2});
    atlas::mul(a, b, p);
    CHECK(approx(p.at({0, 1}), 40.0f));  // 2 * 20
    CHECK(approx(p.at({1, 0}), 90.0f));  // 3 * 30

    // Row broadcast: bias [n] added to each row of [m,n].
    Tensor bias = Tensor::zeros({2});
    bias.data[0] = 100.0f;
    bias.data[1] = 200.0f;
    Tensor rb = Tensor::zeros({2, 2});
    atlas::add(a, bias, rb);
    CHECK(approx(rb.at({0, 0}), 101.0f));  // 1 + 100
    CHECK(approx(rb.at({0, 1}), 202.0f));  // 2 + 200
    CHECK(approx(rb.at({1, 0}), 103.0f));  // 3 + 100
    CHECK(approx(rb.at({1, 1}), 204.0f));  // 4 + 200
}

static void test_move_semantics() {
    Tensor a = Tensor::zeros({2, 2});
    float* p = a.data;
    Tensor b = std::move(a);
    CHECK(a.data == nullptr);  // moved-from is empty (no double free)
    CHECK(!a.owns);
    CHECK(b.data == p);        // buffer was stolen, not copied
    CHECK(b.owns);
}

static void test_view_aliasing() {
    // Tensor::view is the owns=false path weight loading is built on: a window over a
    // caller-owned buffer, writes visible in both directions, nothing freed on destroy.
    float buf[6] = {0, 1, 2, 3, 4, 5};
    {
        Tensor v = Tensor::view(buf, {2, 3});
        CHECK(!v.owns);
        CHECK(v.data == buf);
        CHECK(v.at({1, 2}) == 5.0f);
        v.at({0, 1}) = 9.0f;   // write through the view...
        buf[5] = -4.0f;        // ...and through the buffer
        CHECK(buf[1] == 9.0f);
        CHECK(v.at({1, 2}) == -4.0f);
    }
    CHECK(buf[1] == 9.0f);  // view destruction must not touch the buffer
}

static void test_move_assign() {
    Tensor a = Tensor::zeros({2, 2});
    a.at({0, 0}) = 7.0f;
    float* p = a.data;
    Tensor b = Tensor::zeros({3, 3});  // existing owned buffer must be freed, not leaked
    b = std::move(a);
    CHECK(b.data == p);
    CHECK(b.owns);
    CHECK(b.shape == std::vector<int64_t>({2, 2}));
    CHECK(b.at({0, 0}) == 7.0f);
    CHECK(a.data == nullptr);
    CHECK(!a.owns);
}

static void test_3d_strides() {
    Tensor t = Tensor::zeros({2, 3, 4});
    CHECK(t.numel() == 24);
    CHECK(t.strides == std::vector<int64_t>({12, 4, 1}));  // row-major contiguous
    t.at({1, 2, 3}) = 5.0f;
    CHECK(t.data[1 * 12 + 2 * 4 + 3] == 5.0f);
    Tensor r = t.reshape({6, 4});  // rank change keeps the buffer, recomputes strides
    CHECK(r.strides == std::vector<int64_t>({4, 1}));
    CHECK(r.at({5, 3}) == 5.0f);
}

int main() {
    test_shape_stride_index();
    test_reshape();
    test_matmul();
    test_add_mul();
    test_move_semantics();
    test_view_aliasing();
    test_move_assign();
    test_3d_strides();

    if (g_failures == 0) {
        std::printf("All tensor tests passed.\n");
        return 0;
    }
    std::printf("%d check(s) failed.\n", g_failures);
    return 1;
}
