// Atlas CLI — Phase 1, Step 3. Load the tokenizer + model, encode a prompt, run the
// full-sequence forward pass, greedy-decode the next token, print everything.
//
// Prefill only: no KV cache, no generation loop yet (those arrive after Step 4 proves
// the math against reference/logits.npy). Paths to the committed tokenizer fixtures and
// the locally-generated weight blob are injected by CMake as compile definitions, same
// pattern as the tests. A prompt can be passed as argv[1]; defaults to the Step 0
// reference prompt.

#include <cstdio>
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
    const std::string prompt = argc > 1 ? argv[1] : "The capital of France is";

    std::printf("[1/4] loading tokenizer\n");
    const auto tok =
        atlas::Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");

    std::printf("[2/4] loading model (mmap %s/model.f32.bin)\n", wdir.c_str());
    const auto model =
        atlas::Model::load(wdir + "/model.f32.bin", wdir + "/model.manifest.txt");

    std::printf("[3/4] encoding: \"%s\"\n", prompt.c_str());
    const std::vector<int> ids = tok.encode(prompt);
    std::printf("      ids = [");
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
    std::printf("]\n");

    std::printf("[4/4] forward pass (%zu tokens, 22 layers, FP32, single-threaded)\n",
                ids.size());
    const atlas::Tensor logits = model.forward(ids);

    // Greedy decode: argmax of the last row is the model's next-token prediction.
    const int64_t vocab = logits.shape[1];
    const float* last = logits.data + (logits.shape[0] - 1) * vocab;
    int best = 0;
    for (int64_t j = 1; j < vocab; ++j) {
        if (last[j] > last[best]) best = static_cast<int>(j);
    }

    std::printf("\nnext token: id=%d  text=\"%s\"  logit=%.4f\n", best,
                tok.decode({best}).c_str(), last[best]);
    // decode() strips the word-initial "▁"-space, so re-insert one for display.
    std::printf("completion: \"%s %s\"\n", prompt.c_str(), tok.decode({best}).c_str());
    return 0;
}
