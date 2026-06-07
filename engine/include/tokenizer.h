#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace atlas {

// The TinyLlama (Llama) SentencePiece-BPE tokenizer, reproduced in C++ to match
// HuggingFace byte-for-byte. It turns text into the token-id sequence the forward pass
// consumes, and turns generated ids back into text. CPU-only, no model math.
//
// Dependency-free by design: the engine never parses tokenizer.json or SentencePiece
// protobuf. scripts/export_tokenizer.py does that once and writes two flat, line-oriented
// UTF-8 files this class loads directly:
//   vocab.txt  — 32000 lines, one token per line, line index == token id.
//   merges.txt — 61249 lines, one "a b" pair per line, line order == merge rank
//                (lower rank = higher priority).
// Both are C-style escaped (\\ \r \n) by the exporter so embedded backslash/CR/LF can't
// collide with the line format; load() reverses that escaping. See docs/02-tokenizer.md.
//
// Pipeline reproduced (read off weights/.../tokenizer.json):
//   encode: normalize (prepend "▁", spaces -> "▁") -> split to codepoints, byte-fallback
//           out-of-vocab chars to <0xNN> tokens -> greedily merge the lowest-rank adjacent
//           pair until none remain -> map to ids -> prepend BOS (id 1).
//   decode: ids -> tokens (skip specials) -> concatenate -> reassemble <0xNN> runs into
//           raw bytes -> "▁" -> space -> strip the single leading space.
struct Tokenizer {
    // Special token ids, fixed by this vocab: <unk>=0, <s>=BOS=1, </s>=EOS=2.
    static constexpr int kUnkId = 0;
    static constexpr int kBosId = 1;
    static constexpr int kEosId = 2;

    // Load the vocab + merges exported by scripts/export_tokenizer.py. Reverses the
    // exporter's C-style escaping and builds the lookup tables. Exits on a malformed or
    // unreadable file (the fixtures are a committed contract, not user input).
    static Tokenizer load(const std::string& vocab_path, const std::string& merges_path);

    // Text -> ids. Prepends BOS (id 1) when add_bos (default), matching token_ids.npy.
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // ids -> text: skip specials, "▁" -> space, <0xNN> byte tokens -> raw bytes, strip the
    // single leading space introduced by normalization.
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

private:
    std::vector<std::string> id_to_token_;              // index = id
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<uint64_t, int> merge_rank_;      // packed (lhs_id<<32 | rhs_id) -> rank
};

}  // namespace atlas
