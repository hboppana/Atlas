// Tokenizer edge cases — Phase 1 test hardening.
//
// test_tokenizer pins exact ids against the committed fixtures; this target owns the
// properties the fixtures can't reach: the BOS contract, decode∘encode == identity on
// awkward inputs (multiple spaces, leading space, unicode through the byte-fallback
// path, newlines/tabs, empty string), and an idempotence sweep over the full 32000-entry
// vocab. Property checks compare *text*, not ids — encode is free to pick a different
// segmentation as long as decode inverts it. Blob-free (fixtures are committed).

#include <cstdio>
#include <string>
#include <vector>

#include "../include/tokenizer.h"

#ifndef ATLAS_REFERENCE_DIR
#define ATLAS_REFERENCE_DIR ""
#endif

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

static void check_round_trip(const atlas::Tokenizer& tok, const std::string& s) {
    const std::string back = tok.decode(tok.encode(s));
    if (back != s) {
        std::printf("  FAIL round trip: \"%s\" -> \"%s\"\n", s.c_str(), back.c_str());
        ++g_failures;
    }
}

int main() {
    const std::string ref = ATLAS_REFERENCE_DIR;
    const auto tok =
        atlas::Tokenizer::load(ref + "/tokenizer/vocab.txt", ref + "/tokenizer/merges.txt");

    std::printf("test_tokenizer_edges: vocab + BOS contract\n");
    CHECK(tok.vocab_size() == 32000);

    // BOS (id 1) is prepended by default and only by default — the forward pass
    // depends on encode matching token_ids.npy's leading 1.
    const std::vector<int> with_bos = tok.encode("hello");
    const std::vector<int> without = tok.encode("hello", /*add_bos=*/false);
    CHECK(!with_bos.empty() && with_bos[0] == atlas::Tokenizer::kBosId);
    CHECK(without.empty() || without[0] != atlas::Tokenizer::kBosId);
    CHECK(with_bos.size() == without.size() + 1);

    std::printf("test_tokenizer_edges: decode(encode(s)) == s on awkward inputs\n");
    check_round_trip(tok, "");
    check_round_trip(tok, "Hello, world!");
    check_round_trip(tok, "The capital of France is");
    check_round_trip(tok, "a  b");        // run of spaces — each maps to its own ▁
    check_round_trip(tok, " leading");    // explicit leading space survives the strip
    check_round_trip(tok, "trailing ");
    check_round_trip(tok, "h\xC3\xA9llo w\xC3\xB6rld");      // héllo wörld (2-byte UTF-8)
    check_round_trip(tok, "\xE4\xBD\xA0\xE5\xA5\xBD");       // 你好 (3-byte UTF-8)
    check_round_trip(tok, "\xF0\x9F\x9A\x80");               // 🚀 (4-byte, byte fallback)
    check_round_trip(tok, "line\nbreak");
    check_round_trip(tok, "tab\tchar");
    check_round_trip(tok, "x86_64-w64-mingw32");             // digits + punctuation runs

    // Idempotence sweep over the whole vocab: for every token's surface text t,
    // decode(encode(t)) must reproduce t. Skipped: ids whose decode is a single
    // non-printable/non-ASCII byte (specials decode to ""; lone <0xNN> byte tokens
    // are raw bytes that aren't valid UTF-8 on their own — multi-byte text covers
    // the byte-fallback path above).
    std::printf("test_tokenizer_edges: idempotence sweep over 32000 vocab entries\n");
    int swept = 0, skipped = 0, bad = 0;
    for (int id = 0; id < tok.vocab_size(); ++id) {
        const std::string t = tok.decode({id});
        if (t.empty() ||
            (t.size() == 1 && (static_cast<unsigned char>(t[0]) >= 0x80 ||
                               static_cast<unsigned char>(t[0]) < 0x20))) {
            ++skipped;
            continue;
        }
        if (tok.decode(tok.encode(t)) != t) {
            if (++bad <= 5)  // report the first few, not 32000 lines
                std::printf("  FAIL idempotence: id=%d text=\"%s\"\n", id, t.c_str());
        }
        ++swept;
    }
    std::printf("  swept=%d skipped=%d failed=%d\n", swept, skipped, bad);
    CHECK(bad == 0);

    if (g_failures == 0) {
        std::printf("test_tokenizer_edges: all checks passed\n");
        return 0;
    }
    std::printf("test_tokenizer_edges: %d check(s) FAILED\n", g_failures);
    return 1;
}
