// Forward-pass validation — Phase 1, Steps 3 + 4.
//
// Zero-dependency harness (tiny CHECK macro) matching test_tensor/test_tokenizer.
// Step 3 bar (smoke): the forward pass RUNS end-to-end —
//   - logits shape is [6, 32000] for the reference prompt ids;
//   - argmax of the last row decodes (via the Step 2 tokenizer) to "Paris", the same
//     top-1 the Step 0 HF reference reported.
// Step 4 bar (rigorous): per-logit comparison against reference/logits.npy —
//   - max |Δ| over all 6 × 32000 logits under the pinned ceiling;
//   - mean |Δ| under the pinned ceiling (catches broad systematic drift);
//   - per-row argmax matches the reference at all 6 positions.
// Plus a reader self-test: token_ids.npy must equal the hard-coded ids — this runs
// BEFORE the weight-blob SKIP, so the .npy reader is exercised even without the blob.
//
// The 4.4 GB weight blob is gitignored and regenerated locally (convert_weights.py),
// so the model checks SKIP green when it's absent — CI without weights stays green.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../include/model.h"
#include "../include/npy.h"
#include "../include/tokenizer.h"

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

// std::ifstream, not stat(): MinGW's 32-bit stat() fails on files >2 GB, and the
// weight blob is 4.4 GB.
static bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

int main() {
    const std::string ref = ATLAS_REFERENCE_DIR;
    const std::string wdir = ATLAS_WEIGHTS_DIR;
    const std::string bin = wdir + "/model.f32.bin";
    const std::string manifest = wdir + "/model.manifest.txt";

    if (ref.empty() || wdir.empty()) {
        std::printf("SKIP test_forward: ATLAS_REFERENCE_DIR / ATLAS_WEIGHTS_DIR not set.\n");
        return 0;
    }

    // The Step 0/2 reference prompt ids ([BOS=1] + "The capital of France is").
    const std::vector<int> expected_ids = {1, 450, 7483, 310, 3444, 338};

    // Reader self-test against the committed int32 oracle — proves the NPY header
    // parsing and the <i4 path on a file whose contents the test already knows.
    // Runs before the weight-blob SKIP so it executes on blob-less machines too.
    std::printf("test_forward: npy reader self-test vs token_ids.npy\n");
    const std::vector<int> npy_ids = atlas::load_npy_i32(ref + "/token_ids.npy");
    CHECK(npy_ids == expected_ids);

    if (!file_exists(bin) || !file_exists(manifest)) {
        std::printf("SKIP test_forward: %s not found.\n", bin.c_str());
        std::printf("     Run scripts/convert_weights.py to generate it locally.\n");
        return 0;  // green: the blob is a local artifact, never committed
    }

    std::printf("test_forward: loading tokenizer + model\n");
    const auto tok =
        atlas::Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");
    const auto model = atlas::Model::load(bin, manifest);

    // Encode must reproduce token_ids.npy exactly (test_tokenizer already proves
    // this — re-checked here since forward consumes it).
    const std::vector<int> ids = tok.encode("The capital of France is");
    CHECK(ids == expected_ids);

    std::printf("test_forward: running 6-token forward pass (FP32, single-threaded)\n");
    const atlas::Tensor logits = model.forward(ids);

    // Shape contract: [seq, vocab] = [6, 32000].
    CHECK(logits.shape.size() == 2);
    CHECK(logits.shape[0] == 6);
    CHECK(logits.shape[1] == 32000);

    // Greedy top-1 of the last position must be ▁Paris (decodes to "Paris"),
    // matching the HF reference's reported top-1.
    const int64_t vocab = logits.shape[1];
    const float* last = logits.data + (logits.shape[0] - 1) * vocab;
    int best = 0;
    for (int64_t j = 1; j < vocab; ++j) {
        if (last[j] > last[best]) best = static_cast<int>(j);
    }
    const std::string text = tok.decode({best});
    std::printf("  top-1: id=%d text=\"%s\" logit=%.4f\n", best, text.c_str(), last[best]);
    CHECK(text == "Paris");

    // --- Step 4: tolerance check against the HF golden oracle ---------------------
    // Both sides compute in FP32, but our triple-loop matmul accumulates in a
    // different order than PyTorch's BLAS, so agreement is to a tolerance, not
    // bitwise. First green run measured max_abs=1.44e-4, mean_abs=8.27e-6; pins are
    // ~10x those (docs/04 measure-then-pin), down from the 1e-2 / 1e-3 ceilings.
    std::printf("test_forward: comparing all logits vs logits.npy\n");
    const atlas::Tensor refl = atlas::load_npy_f32(ref + "/logits.npy");
    CHECK(refl.shape.size() == 2);
    CHECK(refl.shape == logits.shape);

    const int64_t seq = logits.shape[0];
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

    // Per-row argmax — greedy decoding must agree at every position, not just the last.
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

    CHECK(max_abs < 1.5e-3);
    CHECK(mean_abs < 1e-4);

    if (g_failures == 0) {
        std::printf("test_forward: all checks passed\n");
        return 0;
    }
    std::printf("test_forward: %d check(s) FAILED\n", g_failures);
    return 1;
}
