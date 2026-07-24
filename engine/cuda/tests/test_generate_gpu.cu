// Greedy decode loop validation — Phase 2, Step 7 (docs/13-cuda-generate.md).
//
// Step 6 proved the one-shot GPU forward against reference/logits.npy; this proves the
// decode loop built on top of it. No new numerics are introduced, so the bar is the CPU
// engine itself: gpu.generate() must produce the EXACT same id sequence as the
// structurally-identical Model::generate() oracle. Greedy argmax makes that discrete
// equality a legitimate contract rather than a tolerance.
//
// The bar rests on both engines breaking argmax ties the same way (argmax_last_row,
// first-index-wins) AND on the top-1/top-2 logit gap comfortably exceeding the ~8e-5
// GPU-vs-CPU drift docs/12 measured — so the test prints that per-step margin. A mismatch
// would be a bug signal, not a tolerance to relax; check the margin at the diverging step.
//
// Structure + SKIP discipline copied from test_forward_gpu: npy reader self-test FIRST
// (runs on blob-less machines), missing weight blob -> SKIP green.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../../include/model.h"
#include "../../include/npy.h"
#include "../../include/tokenizer.h"
#include "../forward.h"
#include "test_harness.h"

#ifndef ATLAS_REFERENCE_DIR
#define ATLAS_REFERENCE_DIR ""
#endif
#ifndef ATLAS_WEIGHTS_DIR
#define ATLAS_WEIGHTS_DIR ""
#endif

static bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

static double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Top-1 minus top-2 logit gap in the last row of a [seq, vocab] tensor — the margin that
// must dwarf the GPU-vs-CPU drift for exact argmax equality to be a sound bar.
static float top1_minus_top2(const atlas::Tensor& logits) {
    const int64_t seq = logits.shape[0];
    const int64_t vocab = logits.shape[1];
    const float* row = logits.data + (seq - 1) * vocab;
    float top1 = row[0], top2 = -1e30f;
    for (int64_t j = 1; j < vocab; ++j) {
        if (row[j] > top1) {
            top2 = top1;
            top1 = row[j];
        } else if (row[j] > top2) {
            top2 = row[j];
        }
    }
    return top1 - top2;
}

int main() {
    const std::string ref = ATLAS_REFERENCE_DIR;
    const std::string wdir = ATLAS_WEIGHTS_DIR;
    const std::string bin = wdir + "/model.f32.bin";
    const std::string manifest = wdir + "/model.manifest.txt";

    if (ref.empty() || wdir.empty()) {
        std::printf("SKIP test_generate_gpu: ATLAS_REFERENCE_DIR / ATLAS_WEIGHTS_DIR not set.\n");
        return 0;
    }

    // [BOS=1] + "The capital of France is".
    const std::vector<int> expected_ids = {1, 450, 7483, 310, 3444, 338};

    // Reader self-test — runs before the blob SKIP so it executes on blob-less machines.
    std::printf("test_generate_gpu: npy reader self-test vs token_ids.npy\n");
    const std::vector<int> npy_ids = atlas::load_npy_i32(ref + "/token_ids.npy");
    CHECK(npy_ids == expected_ids);

    if (!file_exists(bin) || !file_exists(manifest)) {
        std::printf("SKIP test_generate_gpu: %s not found.\n", bin.c_str());
        std::printf("     Run scripts/convert_weights.py to generate it locally.\n");
        return 0;  // green: the blob is a local artifact, never committed
    }

    std::printf("test_generate_gpu: loading tokenizer + model\n");
    const auto tok =
        atlas::Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");
    const auto model = atlas::Model::load(bin, manifest);

    const std::vector<int> ids = tok.encode("The capital of France is");
    CHECK(ids == expected_ids);

    std::printf("test_generate_gpu: uploading weight blob to device\n");
    const atlas::GpuModel gpu = atlas::GpuModel::create(model);

    // --- Per-step diagnostics: manual GPU decode of 8 tokens, printing the tie-break
    //     margin and per-step wall-clock so the O(n^2) growth is visible. The manual loop
    //     mirrors generate() exactly, so it doubles as a reference for the API below.
    std::printf("test_generate_gpu: manual GPU decode (8 tokens) — margins + per-step time\n");
    std::vector<int> manual_ids = ids;
    std::vector<int> gpu_manual;
    auto t_gpu = std::chrono::steady_clock::now();
    for (int step = 0; step < 8; ++step) {
        auto ts = std::chrono::steady_clock::now();
        const atlas::Tensor logits = gpu.forward(manual_ids);
        const int next = atlas::argmax_last_row(logits);
        const float margin = top1_minus_top2(logits);
        std::printf("  step %d: id=%d margin(top1-top2)=%.4g  %.4f s\n", step, next, margin,
                    seconds_since(ts));
        if (next == atlas::Tokenizer::kEosId) break;
        gpu_manual.push_back(next);
        manual_ids.push_back(next);
    }
    const double gpu_secs = seconds_since(t_gpu);

    // --- API path: gpu.generate() must match the manual reference token-for-token.
    const std::vector<int> gpu_ids = gpu.generate(ids, 8);
    CHECK(gpu_ids == gpu_manual);

    // (2) Anchor — the first generated id is 3681 ("Paris"), the value docs/12 recorded.
    // Fails loudly if the forward pass regresses, independent of CPU/GPU agreeing.
    CHECK(!gpu_ids.empty());
    CHECK(gpu_ids[0] == 3681);

    // (3) Decoded text — round-trips through decode(); asserted only non-empty.
    const std::string text = tok.decode(gpu_ids);
    std::printf("test_generate_gpu: completion = \"%s%s\"\n",
                "The capital of France is", text.c_str());
    CHECK(!text.empty());

    // (1) Oracle equality — the CPU twin must produce the EXACT same ids. This is the
    // expensive step (~7-15 s per CPU token), so it stays at 8 tokens.
    std::printf("test_generate_gpu: running CPU oracle Model::generate(8) — slow\n");
    auto t_cpu = std::chrono::steady_clock::now();
    const std::vector<int> cpu_ids = model.generate(ids, 8);
    const double cpu_secs = seconds_since(t_cpu);
    CHECK(gpu_ids == cpu_ids);

    // (4) Determinism — two consecutive generate() calls return identical ids.
    const std::vector<int> a = gpu.generate(ids, 4);
    const std::vector<int> b = gpu.generate(ids, 4);
    CHECK(a == b);

    // (5) Stopping conditions.
    //  - budget honoured, no off-by-one:
    CHECK(gpu.generate(ids, 4).size() == 4);
    //  - on_token returning false on its 2nd call stops after exactly 2 ids:
    int calls = 0;
    const std::vector<int> stopped =
        gpu.generate(ids, 8, [&calls](int) { return ++calls < 2; });
    CHECK(stopped.size() == 2);
    //  - zero budget returns empty and runs no forward (loop body never executes):
    CHECK(gpu.generate(ids, 0).empty());

    // EOS is asserted structurally: 8 greedy tokens from this prompt do not hit EOS, so we
    // check kEosId never appears in any returned sequence (the break stays covered by
    // inspection — promote to a real assertion if a natural EOS turns up).
    for (int id : gpu_ids) CHECK(id != atlas::Tokenizer::kEosId);
    for (int id : cpu_ids) CHECK(id != atlas::Tokenizer::kEosId);

    // (7) Wall-clock, informational — shared box, noisy, never asserted.
    std::printf("  wall-clock: GPU %.4f s vs CPU %.4f s for 8 tokens "
                "(informational, %.3f vs %.3f s/token)\n",
                gpu_secs, cpu_secs, gpu_secs / 8.0, cpu_secs / 8.0);

    if (g_failures == 0) {
        std::printf("test_generate_gpu: all checks passed\n");
        return 0;
    }
    std::printf("test_generate_gpu: %d check(s) FAILED\n", g_failures);
    return 1;
}
