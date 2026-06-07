// BPE tokenizer tests — Phase 1, Step 2.
//
// Zero-dependency harness (a tiny CHECK macro, no GoogleTest/Catch2) matching
// test_tensor.cpp. The primary oracle is the literal id list from docs/02-tokenizer.md:
// encode("The capital of France is") == [1, 450, 7483, 310, 3444, 338] (BOS prepended).
// The vocab/merges fixtures live in reference/tokenizer/; ATLAS_REFERENCE_DIR is injected
// by CMake so the test resolves them regardless of the CTest working directory.

#include "tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

using atlas::Tokenizer;

#ifndef ATLAS_REFERENCE_DIR
#error "ATLAS_REFERENCE_DIR must be defined by the build (path to reference/)."
#endif

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

static void print_ids(const char* label, const std::vector<int>& ids) {
    std::printf("  %s = [", label);
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
    std::printf("]\n");
}

static Tokenizer load_tokenizer() {
    const std::string ref = ATLAS_REFERENCE_DIR;
    return Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");
}

// The contract: byte-for-byte parity with HuggingFace on the reference prompt.
static void test_encode_oracle(const Tokenizer& tok) {
    const std::vector<int> expected = {1, 450, 7483, 310, 3444, 338};
    std::vector<int> ids = tok.encode("The capital of France is");
    if (ids != expected) print_ids("got", ids);
    CHECK(ids == expected);
    CHECK(ids.front() == Tokenizer::kBosId);  // BOS prepended by default
}

// add_bos=false drops the leading BOS but is otherwise identical.
static void test_no_bos(const Tokenizer& tok) {
    std::vector<int> with = tok.encode("The capital of France is", true);
    std::vector<int> without = tok.encode("The capital of France is", false);
    CHECK(without.size() + 1 == with.size());
    CHECK(with.front() == Tokenizer::kBosId);
    CHECK(without.front() != Tokenizer::kBosId);
    CHECK(std::vector<int>(with.begin() + 1, with.end()) == without);
}

static void test_roundtrip(const Tokenizer& tok) {
    // ASCII, including multiple consecutive spaces (each becomes a meta-space).
    CHECK(tok.decode(tok.encode("The capital of France is")) == "The capital of France is");
    CHECK(tok.decode(tok.encode("a  b")) == "a  b");
    // UTF-8 that exercises byte-fallback reassembly on decode.
    CHECK(tok.decode(tok.encode("caf\xC3\xA9")) == "caf\xC3\xA9");  // "café"
    // Empty string: normalization still prepends one meta-space, which decode strips.
    CHECK(tok.decode(tok.encode("")) == "");
}

// A literal newline is out-of-vocab, so byte-fallback emits <0x0A> = id 13, and decode
// reassembles it back to '\n'.
static void test_byte_fallback(const Tokenizer& tok) {
    std::vector<int> ids = tok.encode("\n", /*add_bos=*/false);
    CHECK(!ids.empty() && ids.back() == 13);  // <0x0A>
    CHECK(tok.decode(tok.encode("\n")) == "\n");
}

int main() {
    Tokenizer tok = load_tokenizer();
    CHECK(tok.vocab_size() == 32000);

    test_encode_oracle(tok);
    test_no_bos(tok);
    test_roundtrip(tok);
    test_byte_fallback(tok);

    if (g_failures == 0) {
        std::printf("test_tokenizer: all checks passed\n");
        return 0;
    }
    std::printf("test_tokenizer: %d check(s) FAILED\n", g_failures);
    return 1;
}
