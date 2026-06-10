// Forward-pass smoke test — Phase 1, Step 3.
//
// Zero-dependency harness (tiny CHECK macro) matching test_tensor/test_tokenizer.
// This step's bar: the forward pass RUNS end-to-end and the top-1 prediction is sane —
//   - logits shape is [6, 32000] for the reference prompt ids;
//   - argmax of the last row decodes (via the Step 2 tokenizer) to "Paris", the same
//     top-1 the Step 0 HF reference reported.
// The rigorous tolerance check against reference/logits.npy is Step 4 (needs a C++
// .npy reader).
//
// The 4.4 GB weight blob is gitignored and regenerated locally (convert_weights.py),
// so this test SKIPs green when it's absent — CI without weights stays green.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "model.h"
#include "tokenizer.h"

#ifndef ATLAS_REFERENCE_DIR
#error "ATLAS_REFERENCE_DIR must be defined by the build (path to reference/)."
#endif
#ifndef ATLAS_WEIGHTS_DIR
#error "ATLAS_WEIGHTS_DIR must be defined by the build (path to weights/tinyllama-1.1b-chat/)."
#endif

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// NOTE: not stat() — MinGW's 32-bit stat overflows on the 4.4 GB blob and reports
// the file missing. Opening a read stream works regardless of size.
static bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

int main() {
    const std::string ref = ATLAS_REFERENCE_DIR;
    const std::string wdir = ATLAS_WEIGHTS_DIR;
    const std::string bin = wdir + "/model.f32.bin";
    const std::string manifest = wdir + "/model.manifest.txt";

    if (!file_exists(bin) || !file_exists(manifest)) {
        std::printf("SKIP test_forward: %s not found.\n", bin.c_str());
        std::printf("     Run scripts/convert_weights.py to generate it locally.\n");
        return 0;  // green: the blob is a local artifact, never committed
    }

    std::printf("test_forward: loading tokenizer + model\n");
    const auto tok =
        atlas::Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");
    const auto model = atlas::Model::load(bin, manifest);

    // The Step 0/2 reference prompt; encode must reproduce token_ids.npy exactly
    // (test_tokenizer already proves this — re-checked here since forward consumes it).
    const std::vector<int> expected_ids = {1, 450, 7483, 310, 3444, 338};
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

    if (g_failures == 0) {
        std::printf("test_forward: all checks passed\n");
        return 0;
    }
    std::printf("test_forward: %d check(s) FAILED\n", g_failures);
    return 1;
}
