// Atlas CLI — Phase 1, Steps 3 + 5. Load the tokenizer + model, encode a prompt, run
// the full-sequence forward pass, greedy-decode the next token, print everything.
//
// Prefill only: no KV cache, no generation loop yet (those arrive after Phase 1 wraps).
// Paths to the committed tokenizer fixtures and the locally-generated weight blob are
// injected by CMake as compile definitions, same pattern as the tests. A prompt can be
// passed as an argument; defaults to the Step 0 reference prompt. `--int8` quantizes
// the linear weights to per-row symmetric INT8 before the forward (Step 5, W8A32 —
// docs/05-quantization.md).

#include <cstdio>
#include <cstdlib>
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

int main(int argc, char** argv) {
    const std::string ref = ATLAS_REFERENCE_DIR;
    const std::string wdir = ATLAS_WEIGHTS_DIR;
    bool int8 = false;
    int max_new_tokens = 1;  // default 1: preserves the single-token prediction verbatim
    std::string prompt = "The capital of France is";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--int8") {
            int8 = true;
        } else if (arg == "-n" && i + 1 < argc) {
            max_new_tokens = std::atoi(argv[++i]);
        } else {
            prompt = arg;
        }
    }

    std::printf("[1/4] loading tokenizer\n");
    const auto tok =
        atlas::Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");

    std::printf("[2/4] loading model (mmap %s/model.f32.bin)\n", wdir.c_str());
    auto model = atlas::Model::load(wdir + "/model.f32.bin", wdir + "/model.manifest.txt");
    if (int8) {
        std::printf("      quantizing linear weights to INT8 (per-row symmetric)\n");
        model.quantize_int8();
        size_t bytes = 0;
        for (const auto& kv : model.qweights) {
            bytes += kv.second.data.size() + kv.second.scales.size() * sizeof(float);
        }
        std::printf("      %zu matrices -> %.2f GB of int8 (+ per-row scales)\n",
                    model.qweights.size(), static_cast<double>(bytes) / 1e9);
    }

    std::printf("[3/4] encoding: \"%s\"\n", prompt.c_str());
    const std::vector<int> ids = tok.encode(prompt);
    std::printf("      ids = [");
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
    std::printf("]\n");

    std::printf("[4/4] greedy decode (%d new token(s), %zu-token prompt, 22 layers, %s,"
                " single-threaded)\n",
                max_new_tokens, ids.size(),
                int8 ? "INT8 weights / FP32 activations" : "FP32");

    // Greedy decode loop (Model::generate): each step re-runs the full forward over the
    // grown sequence and argmaxes the last row. No KV cache — CPU-only and slow at ~7 s
    // per token, but it makes a real completion demonstrable (docs/13-cuda-generate.md).
    const std::vector<int> gen = model.generate(ids, max_new_tokens);

    std::printf("\ngenerated ids = [");
    for (size_t i = 0; i < gen.size(); ++i) std::printf("%s%d", i ? ", " : "", gen[i]);
    std::printf("]\n");
    // decode() strips the word-initial "▁"-space; prepend one so the join reads naturally.
    std::printf("completion: \"%s %s\"\n", prompt.c_str(), tok.decode(gen).c_str());
    return 0;
}
