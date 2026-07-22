// Full GPU forward-pass validation — Phase 2, Step 6 (docs/12-cuda-forward.md).
//
// Steps 2-5 proved each kernel against its CPU oracle in isolation; this proves the
// COMPOSITION against the project's end-to-end oracle — the same reference/logits.npy (HF
// golden logits for "The capital of France is") that gates the CPU engine in test_forward,
// with the same three-part bar:
//   - per-logit max-abs / mean-abs under pinned ceilings;
//   - per-row argmax agreement at all 6 positions (binary, non-negotiable);
//   - argmax of the last row decodes to "Paris" via the Step 2 tokenizer.
// Plus an attribution diagnostic: GPU-vs-CPU max-abs/mean-abs, printed but NOT pinned. If
// the HF check ever reddens, that number says whether the drift is kernel-side (GPU != CPU)
// or shared engine-side (both != HF).
//
// Mirrors test_forward.cpp's structure and SKIP discipline exactly: the npy reader
// self-test runs FIRST, so it is exercised even on machines without the 4.4 GB weight
// blob; a missing blob then SKIPs green (the blob is a local artifact, never committed).

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

// std::ifstream, not stat() — same reason as test_forward: the blob is 4.4 GB.
static bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// Index of the largest element of a vocab-wide logit row.
static int64_t argmax(const float* row, int64_t n) {
    int64_t best = 0;
    for (int64_t j = 1; j < n; ++j) {
        if (row[j] > row[best]) best = j;
    }
    return best;
}

static double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    const std::string ref = ATLAS_REFERENCE_DIR;
    const std::string wdir = ATLAS_WEIGHTS_DIR;
    const std::string bin = wdir + "/model.f32.bin";
    const std::string manifest = wdir + "/model.manifest.txt";

    if (ref.empty() || wdir.empty()) {
        std::printf("SKIP test_forward_gpu: ATLAS_REFERENCE_DIR / ATLAS_WEIGHTS_DIR not set.\n");
        return 0;
    }

    // The Step 0/2 reference prompt ids ([BOS=1] + "The capital of France is").
    const std::vector<int> expected_ids = {1, 450, 7483, 310, 3444, 338};

    // Reader self-test — runs before the blob SKIP so it executes on blob-less machines.
    std::printf("test_forward_gpu: npy reader self-test vs token_ids.npy\n");
    const std::vector<int> npy_ids = atlas::load_npy_i32(ref + "/token_ids.npy");
    CHECK(npy_ids == expected_ids);

    if (!file_exists(bin) || !file_exists(manifest)) {
        std::printf("SKIP test_forward_gpu: %s not found.\n", bin.c_str());
        std::printf("     Run scripts/convert_weights.py to generate it locally.\n");
        return 0;  // green: the blob is a local artifact, never committed
    }

    std::printf("test_forward_gpu: loading tokenizer + model\n");
    const auto tok =
        atlas::Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");
    const auto model = atlas::Model::load(bin, manifest);

    const std::vector<int> ids = tok.encode("The capital of France is");
    CHECK(ids == expected_ids);

    // One cudaMalloc + one H2D of the whole 4.4 GB blob. Timed because it is the cost the
    // upload-once design exists to amortize.
    std::printf("test_forward_gpu: uploading weight blob to device\n");
    auto t0 = std::chrono::steady_clock::now();
    const atlas::GpuModel gpu = atlas::GpuModel::create(model);
    std::printf("  GpuModel::create: %.3f s\n", seconds_since(t0));

    std::printf("test_forward_gpu: running 6-token GPU forward pass (FP32)\n");
    t0 = std::chrono::steady_clock::now();
    const atlas::Tensor logits = gpu.forward(ids);
    const double gpu_secs = seconds_since(t0);

    // Shape contract: [seq, vocab] = [6, 32000].
    CHECK(logits.shape.size() == 2);
    CHECK(logits.shape[0] == 6);
    CHECK(logits.shape[1] == 32000);

    const int64_t seq = logits.shape[0];
    const int64_t vocab = logits.shape[1];

    // Smoke: greedy top-1 of the last position must decode to "Paris".
    const float* last = logits.data + (seq - 1) * vocab;
    const int best = static_cast<int>(argmax(last, vocab));
    const std::string text = tok.decode({best});
    std::printf("  top-1: id=%d text=\"%s\" logit=%.4f\n", best, text.c_str(), last[best]);
    CHECK(text == "Paris");

    // --- Rigorous: the HF golden oracle ------------------------------------------------
    // The gate is agreement with the real model, same bar as test_forward. Every kernel is
    // within ~1e-6 of its CPU oracle in isolation, but 22 layers compound and the dominant
    // diff source is the tiled matmul's accumulation reorder across 155 projections.
    std::printf("test_forward_gpu: comparing all logits vs logits.npy\n");
    const atlas::Tensor refl = atlas::load_npy_f32(ref + "/logits.npy");
    CHECK(refl.shape.size() == 2);
    CHECK(refl.shape == logits.shape);

    const Diff hf = compare(logits, refl);
    std::printf("  GPU vs HF: max_abs=%.6g mean_abs=%.6g over %lld logits\n", hf.max_abs,
                hf.mean_abs, static_cast<long long>(seq * vocab));

    // Per-row argmax — greedy decoding must agree at every position, not just the last.
    for (int64_t r = 0; r < seq; ++r) {
        const int64_t our_best = argmax(logits.data + r * vocab, vocab);
        const int64_t ref_best = argmax(refl.data + r * vocab, vocab);
        std::printf("  row %lld argmax: ours=%lld ref=%lld\n", static_cast<long long>(r),
                    static_cast<long long>(our_best), static_cast<long long>(ref_best));
        CHECK(our_best == ref_best);
    }

    // Measured on the first green run (Suramar, A6000 sm_86, 2026-07-22):
    // max_abs=2.04e-4, mean_abs=8.10e-6 — the same order of magnitude as the CPU engine's
    // 1.44e-4 / 8.27e-6, as predicted: both differ from PyTorch BLAS only by summation
    // order, and the GPU adds the tiled matmul's accumulation reorder on top. Pinned at
    // ~10x measured, per the standing measure-then-pin rule.
    CHECK(hf.max_abs < 2e-3);
    CHECK(hf.mean_abs < 8e-5);

    // --- Attribution diagnostic (printed, never pinned) --------------------------------
    // Splits any future HF failure into kernel-side vs shared engine-side drift.
    std::printf("test_forward_gpu: running the CPU forward pass for attribution\n");
    t0 = std::chrono::steady_clock::now();
    const atlas::Tensor cpu_logits = model.forward(ids);
    const double cpu_secs = seconds_since(t0);

    const Diff gc = compare(logits, cpu_logits);
    std::printf("  GPU vs CPU: max_abs=%.6g mean_abs=%.6g  (diagnostic only)\n", gc.max_abs,
                gc.mean_abs);

    // Informational wall-clock only — shared box, noisy, every launcher still syncing.
    // The real benchmark pass comes after the perf follow-ups.
    std::printf("  wall-clock: GPU %.4f s vs CPU %.4f s (informational, not asserted)\n",
                gpu_secs, cpu_secs);

    if (g_failures == 0) {
        std::printf("test_forward_gpu: all checks passed\n");
        return 0;
    }
    std::printf("test_forward_gpu: %d check(s) FAILED\n", g_failures);
    return 1;
}
