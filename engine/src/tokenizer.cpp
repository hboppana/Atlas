#include "../include/tokenizer.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>

namespace atlas {

namespace {

// SentencePiece meta-space U+2581 ("▁"), UTF-8. Normalization turns every space into
// this; decode turns it back. Spelled as raw bytes so the source file needs no UTF-8.
const std::string kMetaSpace = "\xE2\x96\x81";

[[noreturn]] void fail(const std::string& msg) {
    std::fprintf(stderr, "Tokenizer: %s\n", msg.c_str());
    std::exit(1);
}

// Reverse scripts/export_tokenizer.py's C-style escaping: \\ -> \, \r -> CR, \n -> LF.
// An unknown escape is kept verbatim (shouldn't occur; the exporter only emits the three).
std::string unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char n = s[++i];
            if (n == '\\') out.push_back('\\');
            else if (n == 'r') out.push_back('\r');
            else if (n == 'n') out.push_back('\n');
            else { out.push_back('\\'); out.push_back(n); }
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

// Byte count of a UTF-8 codepoint from its lead byte; 0 if this is a continuation/invalid
// byte (caller falls back to consuming a single byte).
size_t utf8_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;   // 110xxxxx
    if ((lead >> 4) == 0xE) return 3;   // 1110xxxx
    if ((lead >> 3) == 0x1E) return 4;  // 11110xxx
    return 0;
}

// Pack an ordered token-id pair into one 64-bit key for the merge-rank table.
uint64_t pack(int lhs, int rhs) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(lhs)) << 32) |
           static_cast<uint32_t>(rhs);
}

// Vocab id of the byte-fallback token for raw byte b, e.g. 0x0A -> "<0x0A>" (uppercase hex,
// matching the Llama vocab). Every one of the 256 byte tokens is present.
int byte_token_id(const std::unordered_map<std::string, int>& vocab, unsigned char b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    auto it = vocab.find(buf);
    if (it == vocab.end()) fail(std::string("missing byte-fallback token ") + buf);
    return it->second;
}

// If `t` is a byte-fallback token "<0xNN>", return its byte value 0..255; else -1.
int byte_token_value(const std::string& t) {
    if (t.size() != 6 || t[0] != '<' || t[1] != '0' || t[2] != 'x' || t[5] != '>') return -1;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    int hi = hex(t[3]), lo = hex(t[4]);
    if (hi < 0 || lo < 0) return -1;
    return hi * 16 + lo;
}

// Append `tok` to `out`, replacing each meta-space run with a regular space.
void append_unmetaspaced(std::string& out, const std::string& tok) {
    size_t i = 0;
    while (i < tok.size()) {
        if (i + kMetaSpace.size() <= tok.size() &&
            tok.compare(i, kMetaSpace.size(), kMetaSpace) == 0) {
            out.push_back(' ');
            i += kMetaSpace.size();
        } else {
            out.push_back(tok[i]);
            ++i;
        }
    }
}

// Read one LF-terminated line's content. Files are eol=lf (see .gitattributes), but strip a
// stray trailing CR defensively in case of a CRLF checkout — a *legitimate* trailing CR is
// escaped as "\r" by the exporter, so a literal CR here is only a line-ending artifact.
bool read_line(std::ifstream& f, std::string& line) {
    if (!std::getline(f, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

}  // namespace

Tokenizer Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    Tokenizer tok;

    // vocab.txt: one token per line, line index == id. Never skip a line (even an empty
    // one) or the line-index == id invariant breaks.
    std::ifstream vf(vocab_path, std::ios::binary);
    if (!vf) fail("cannot open vocab file: " + vocab_path);
    std::string line;
    while (read_line(vf, line)) {
        std::string token = unescape(line);
        int id = static_cast<int>(tok.id_to_token_.size());
        tok.id_to_token_.push_back(token);
        tok.token_to_id_.emplace(std::move(token), id);
    }
    if (tok.id_to_token_.empty()) fail("vocab file is empty: " + vocab_path);

    // merges.txt: "lhs rhs" per line, line index == rank. The single ASCII space is the
    // delimiter; escaped halves never contain a literal space, so split before unescaping.
    std::ifstream mf(merges_path, std::ios::binary);
    if (!mf) fail("cannot open merges file: " + merges_path);
    int rank = 0;
    while (read_line(mf, line)) {
        size_t sp = line.find(' ');
        if (sp == std::string::npos) fail("merge line missing space delimiter");
        std::string lhs = unescape(line.substr(0, sp));
        std::string rhs = unescape(line.substr(sp + 1));
        auto li = tok.token_to_id_.find(lhs);
        auto ri = tok.token_to_id_.find(rhs);
        if (li == tok.token_to_id_.end() || ri == tok.token_to_id_.end()) {
            fail("merge references a token not in vocab");
        }
        tok.merge_rank_.emplace(pack(li->second, ri->second), rank);
        ++rank;
    }

    return tok;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    // 1. Normalize: prepend a meta-space, then replace every space with a meta-space.
    std::string norm = kMetaSpace;
    for (char c : text) {
        if (c == ' ') norm += kMetaSpace;
        else norm.push_back(c);
    }

    // 2. Initial symbols: one vocab id per codepoint, byte-falling-back OOV codepoints to
    //    one <0xNN> token per UTF-8 byte. Symbols are carried as ids so merges are id-keyed.
    std::vector<int> symbols;
    size_t i = 0;
    while (i < norm.size()) {
        size_t len = utf8_len(static_cast<unsigned char>(norm[i]));
        if (len == 0 || i + len > norm.size()) len = 1;  // invalid/truncated -> one byte
        std::string cp = norm.substr(i, len);
        auto it = token_to_id_.find(cp);
        if (it != token_to_id_.end()) {
            symbols.push_back(it->second);
        } else {
            for (size_t b = 0; b < len; ++b) {
                symbols.push_back(byte_token_id(token_to_id_, static_cast<unsigned char>(norm[i + b])));
            }
        }
        i += len;
    }

    // 3. BPE: repeatedly merge the adjacent pair with the lowest rank (ties -> leftmost),
    //    until no adjacent pair is mergeable. O(n^2) over a short prompt — correctness first.
    while (symbols.size() >= 2) {
        int best_rank = INT_MAX;
        int best_pos = -1;
        for (size_t p = 0; p + 1 < symbols.size(); ++p) {
            auto it = merge_rank_.find(pack(symbols[p], symbols[p + 1]));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = static_cast<int>(p);
            }
        }
        if (best_pos < 0) break;
        // The merge of two vocab tokens is itself a vocab token (string concatenation).
        std::string merged = id_to_token_[symbols[best_pos]] + id_to_token_[symbols[best_pos + 1]];
        auto mit = token_to_id_.find(merged);
        if (mit == token_to_id_.end()) fail("merged token not in vocab: " + merged);
        symbols[best_pos] = mit->second;
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    // 4. Prepend BOS (id 1) to match the post-processor / token_ids.npy.
    std::vector<int> ids;
    ids.reserve(symbols.size() + (add_bos ? 1 : 0));
    if (add_bos) ids.push_back(kBosId);
    ids.insert(ids.end(), symbols.begin(), symbols.end());
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id == kUnkId || id == kBosId || id == kEosId) continue;  // skip specials
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        const std::string& tok = id_to_token_[id];
        int bv = byte_token_value(tok);
        if (bv >= 0) {
            out.push_back(static_cast<char>(bv));  // reassemble raw byte
        } else {
            append_unmetaspaced(out, tok);         // meta-space -> space
        }
    }
    // Strip the single leading space introduced by normalization's prepended meta-space.
    if (!out.empty() && out[0] == ' ') out.erase(out.begin());
    return out;
}

}  // namespace atlas
