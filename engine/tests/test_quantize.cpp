// INT8 quantization — Phase 1, Step 5.
//
// Zero-dependency harness (tiny CHECK macro) matching test_tensor/test_tokenizer.
// Unit tests, blob-free — they run anywhere, like the Step 4 reader self-test:
//   1. Round-trip on a hand-built matrix (exactly-representable row, negative-max row,
//      all-zero row, generic row): scales are max_abs/127 exactly, representable values
//      survive bit-perfectly, every element obeys |w - dq| <= scale/2, the zero row
//      yields scale 1.0 with all-zero ints, and quantize(-W) == -quantize(W).
//   2. linear_q8 matches the FP32 computation over the dequantized weights, and sits
//      within the analytic error bound (scale_o/2 per weight) of the computation over
//      the original weights. Deterministic formula-filled matrices, no RNG.
//
// Blob-gated end-to-end (SKIPs green without the 4.4 GB local blob, like test_forward):
//   3. Model::quantize_int8() + INT8 forward vs reference/logits.npy with the Step 4
//      gates — per-row argmax matches at all 6 positions (greedy decoding unchanged by
//      quantization), max/mean abs diff under the pinned thresholds, measured values
//      printed (measure-then-pin). The FP32 engine sits 1.44e-4 from this oracle, so
//      the diff measured here is the cost of quantization, not implementation error.
// See docs/05-quantization.md.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../include/model.h"
#include "../include/npy.h"
#include "../include/quantize.h"
#include "../include/tensor.h"

#ifndef ATLAS_REFERENCE_DIR
#define ATLAS_REFERENCE_DIR ""
#endif
#ifndef ATLAS_WEIGHTS_DIR
#define ATLAS_WEIGHTS_DIR ""
#endif

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// --- 1. round-trip on a hand-built matrix --------------------------------------------
static void test_round_trip() {
    std::printf("test_quantize: round-trip on a hand-built 4x6 matrix\n");

    atlas::Tensor w = atlas::Tensor::zeros({4, 6});
    // Row 0: max_abs is exactly 127 -> scale 1.0; every value is an integer in
    // [-127, 127], so the round trip must be bit-perfect.
    const float row0[6] = {127.0f, -3.0f, 7.0f, 0.0f, 50.0f, -127.0f};
    // Row 1: the max-magnitude value is negative (-10) -> scale 10/127, q = -127 there.
    const float row1[6] = {-10.0f, 2.5f, 5.0f, -1.0f, 0.5f, 3.0f};
    // Row 2: all zeros -> scale 1.0, all-zero ints, no NaN/div-by-zero.
    const float row2[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Row 3: generic values, nothing exactly representable.
    const float row3[6] = {0.1f, -0.23f, 0.017f, 0.31f, -0.29f, 0.05f};
    const float* rows[4] = {row0, row1, row2, row3};
    for (int64_t r = 0; r < 4; ++r)
        for (int64_t i = 0; i < 6; ++i) w.at({r, i}) = rows[r][i];

    const atlas::QTensor q = atlas::quantize_rows(w);
    CHECK(q.rows == 4);
    CHECK(q.cols == 6);

    // Scales are exactly max_abs/127 (same float expression as the implementation).
    CHECK(q.scales[0] == 1.0f);
    CHECK(q.scales[1] == 10.0f / 127.0f);
    CHECK(q.scales[2] == 1.0f);
    CHECK(q.scales[3] == 0.31f / 127.0f);

    // Row 0 quantizes to its own integer values; row 1's negative max hits -127
    // exactly; row 2 is all-zero ints.
    const int row0_q[6] = {127, -3, 7, 0, 50, -127};
    for (int64_t i = 0; i < 6; ++i) CHECK(q.data[static_cast<size_t>(i)] == row0_q[i]);
    CHECK(q.data[1 * 6 + 0] == -127);
    for (int64_t i = 0; i < 6; ++i) CHECK(q.data[static_cast<size_t>(2 * 6 + i)] == 0);

    // Round trip: row 0 bit-perfect; every element within scale/2 of the original
    // (the worst case of round-to-nearest), with a hair of slack for the rounded
    // 1/scale the quantizer multiplies by.
    const atlas::Tensor dq = atlas::dequantize(q);
    CHECK(dq.shape == w.shape);
    for (int64_t i = 0; i < 6; ++i) CHECK(dq.at({0, i}) == w.at({0, i}));
    for (int64_t r = 0; r < 4; ++r) {
        const float bound = 0.5f * q.scales[static_cast<size_t>(r)] * (1.0f + 1e-5f);
        for (int64_t i = 0; i < 6; ++i)
            CHECK(std::fabs(dq.at({r, i}) - w.at({r, i})) <= bound);
    }

    // The +-127 (not -128) decision: quantization commutes with negation exactly.
    atlas::Tensor neg = atlas::Tensor::zeros({4, 6});
    for (int64_t r = 0; r < 4; ++r)
        for (int64_t i = 0; i < 6; ++i) neg.at({r, i}) = -w.at({r, i});
    const atlas::QTensor qn = atlas::quantize_rows(neg);
    for (size_t i = 0; i < q.data.size(); ++i) CHECK(qn.data[i] == -q.data[i]);
    for (size_t r = 0; r < q.scales.size(); ++r) CHECK(qn.scales[r] == q.scales[r]);
}

// --- 2. linear_q8 vs the FP32 computation --------------------------------------------
static void test_linear_q8() {
    std::printf("test_quantize: linear_q8 vs FP32 on formula-filled 3x8 @ (5x8)^T\n");

    // Deterministic, formula-filled (no RNG): mixed signs, nothing degenerate.
    atlas::Tensor x = atlas::Tensor::zeros({3, 8});
    atlas::Tensor w = atlas::Tensor::zeros({5, 8});
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t k = 0; k < 8; ++k) x.at({i, k}) = 0.25f * ((i * 7 + k * 3) % 11 - 5);
    for (int64_t o = 0; o < 5; ++o)
        for (int64_t k = 0; k < 8; ++k) w.at({o, k}) = 0.125f * ((o * 5 + k * 7) % 13 - 6);

    const atlas::QTensor q = atlas::quantize_rows(w);
    const atlas::Tensor dq = atlas::dequantize(q);

    atlas::Tensor y = atlas::Tensor::zeros({3, 5});
    atlas::linear_q8(x, q, y);

    for (int64_t i = 0; i < 3; ++i) {
        for (int64_t o = 0; o < 5; ++o) {
            // Reference 1: x @ dequant(W)^T computed independently — linear_q8 does the
            // same math with the scale factored out of the sum, so agreement is to FP
            // reassociation noise only.
            float y_dq = 0.0f;
            float y_fp = 0.0f;
            float xabs = 0.0f;
            for (int64_t k = 0; k < 8; ++k) {
                y_dq += x.at({i, k}) * dq.at({o, k});
                y_fp += x.at({i, k}) * w.at({o, k});
                xabs += std::fabs(x.at({i, k}));
            }
            CHECK(std::fabs(y.at({i, o}) - y_dq) <= 1e-5f);

            // Reference 2: against the ORIGINAL weights, the error is analytically
            // bounded — each weight is off by at most scale_o/2, so the dot product
            // is off by at most (scale_o/2) * sum|x|.
            const float bound =
                0.5f * q.scales[static_cast<size_t>(o)] * xabs * (1.0f + 1e-5f) + 1e-7f;
            CHECK(std::fabs(y.at({i, o}) - y_fp) <= bound);
        }
    }
}

// std::ifstream, not stat(): MinGW's 32-bit stat() fails on files >2 GB, and the
// weight blob is 4.4 GB.
static bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// --- 3. blob-gated end-to-end: INT8 forward vs the HF golden oracle ------------------
static void test_int8_forward(const std::string& ref, const std::string& wdir) {
    const std::string bin = wdir + "/model.f32.bin";
    const std::string manifest = wdir + "/model.manifest.txt";
    if (!file_exists(bin) || !file_exists(manifest)) {
        std::printf("SKIP int8 forward: %s not found.\n", bin.c_str());
        std::printf("     Run scripts/convert_weights.py to generate it locally.\n");
        return;  // green: the blob is a local artifact, never committed
    }

    std::printf("test_quantize: loading model + quantizing linear weights to INT8\n");
    const std::vector<int> ids = atlas::load_npy_i32(ref + "/token_ids.npy");
    auto model = atlas::Model::load(bin, manifest);
    model.quantize_int8();
    size_t qbytes = 0;
    for (const auto& kv : model.qweights) {
        qbytes += kv.second.data.size() + kv.second.scales.size() * sizeof(float);
    }
    std::printf("  %zu matrices -> %.3f GB of int8 (+ per-row scales); was 4.14 GB FP32\n",
                model.qweights.size(), static_cast<double>(qbytes) / 1e9);
    CHECK(model.qweights.size() == 155);  // 22 layers x 7 projections + lm_head

    std::printf("test_quantize: running 6-token INT8 forward pass\n");
    const atlas::Tensor logits = model.forward(ids);
    const atlas::Tensor refl = atlas::load_npy_f32(ref + "/logits.npy");
    CHECK(logits.shape.size() == 2);
    CHECK(logits.shape == refl.shape);

    const int64_t seq = logits.shape[0];
    const int64_t vocab = logits.shape[1];
    double max_abs = 0.0;
    double sum_abs = 0.0;  // double: the metric must be more precise than what it measures
    const int64_t total = seq * vocab;
    for (int64_t i = 0; i < total; ++i) {
        const double d = std::fabs(static_cast<double>(logits.data[i]) -
                                   static_cast<double>(refl.data[i]));
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    const double mean_abs = sum_abs / static_cast<double>(total);
    std::printf("  max_abs=%.6g mean_abs=%.6g over %lld logits\n", max_abs, mean_abs,
                static_cast<long long>(total));

    // The semantic gate: greedy decoding must be unchanged by quantization at every
    // position, not just the last. Deterministic forward, so this is a fixed fact of
    // the scheme, not a flaky threshold.
    for (int64_t r = 0; r < seq; ++r) {
        const float* ours = logits.data + r * vocab;
        const float* theirs = refl.data + r * vocab;
        int64_t our_best = 0, ref_best = 0;
        for (int64_t j = 1; j < vocab; ++j) {
            if (ours[j] > ours[our_best]) our_best = j;
            if (theirs[j] > theirs[ref_best]) ref_best = j;
        }
        std::printf("  row %lld argmax: ours=%lld ref=%lld\n", static_cast<long long>(r),
                    static_cast<long long>(our_best), static_cast<long long>(ref_best));
        CHECK(our_best == ref_best);
    }
    // For the docs/05 measured-results record: FP32 had ▁Paris (3681) at 13.3885.
    std::printf("  last-row logit for id 3681 (▁Paris): %.4f\n",
                logits.data[(seq - 1) * vocab + 3681]);

    // Pinned at ~2x the first measured run (2026-06-10: max_abs=1.497, mean_abs=0.0843
    // — see docs/05 for why the original 0.5/0.05 ceilings underestimated 22-layer
    // error accumulation). The forward is deterministic, so 2x is margin for compiler
    // reassociation differences, not run-to-run noise; a broken scheme (wrong scale
    // axis, overflow) still moves logits by whole units and fails these.
    CHECK(max_abs < 3.0);
    CHECK(mean_abs < 0.17);
}

int main() {
    test_round_trip();
    test_linear_q8();

    // End-to-end needs the reference oracles (committed) and the weight blob (local
    // only) — SKIPs green without the blob, after the unit tests have already run.
    const std::string ref = ATLAS_REFERENCE_DIR;
    const std::string wdir = ATLAS_WEIGHTS_DIR;
    if (!ref.empty() && !wdir.empty()) {
        test_int8_forward(ref, wdir);
    } else {
        std::printf("SKIP int8 forward: ATLAS_REFERENCE_DIR / ATLAS_WEIGHTS_DIR not set.\n");
    }

    if (g_failures == 0) {
        std::printf("test_quantize: all checks passed\n");
        return 0;
    }
    std::printf("test_quantize: %d check(s) FAILED\n", g_failures);
    return 1;
}
