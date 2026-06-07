# Phase 1 · Step 2 — BPE Tokenizer

> Status: **done** — `test_tokenizer` green (exact-match oracle, round-trips, byte fallback).
> Predecessor: Step 1 — tensor foundation + C++ build — **done** ([01-tensor-foundation.md](01-tensor-foundation.md))
> Successors: model + weight loading → forward-pass validation → quantize

## Goal

Implement `encode` / `decode` for the TinyLlama (Llama) SentencePiece-**BPE** tokenizer in
C++, reproducing HuggingFace **exactly**, so that:

```
encode("The capital of France is")  ==  reference/token_ids.npy  ==  [1, 450, 7483, 310, 3444, 338]
```

This is the second from-scratch primitive (after `Tensor`). It turns text into the token-id
sequence the forward pass will consume, and turns generated ids back into text. CPU-only,
no model math yet.

## Definition of done

- `cmake --build` still green (MinGW-w64 GCC, C++17). `tokenizer.cpp` joins `atlas_engine`.
- `test_tokenizer` passes via `ctest`:
  - `encode("The capital of France is")` equals `reference/token_ids.npy` **exactly**,
    including the prepended **BOS = 1**.
  - `decode(encode(s)) == s` round-trips for ASCII and a UTF-8 (byte-fallback) string.
  - byte-fallback and a couple of edge cases (empty string, multiple spaces) behave as HF.

## What the tokenizer must reproduce

The HF pipeline for this model (read off `weights/tinyllama-1.1b-chat/tokenizer.json`):

1. **Normalize** (`Sequence`): `Prepend "▁"` then `Replace " " -> "▁"`. So spaces become the
   SentencePiece meta-space `▁` (U+2581) and one `▁` is prepended to the whole string.
   `"The capital of France is"` → `"▁The▁capital▁of▁France▁is"`. (`legacy=false`,
   `pre_tokenizer=null` — the whole normalized string is one piece fed to BPE.)
2. **BPE encode** the normalized string:
   - Split into Unicode characters. For each char: if its string is in the vocab, it is a
     symbol; otherwise **byte_fallback** emits one `<0xNN>` token per UTF-8 byte (e.g. an
     out-of-vocab char, or a literal newline → `<0x0A>` = id 13).
   - Greedily apply merges: repeatedly find the adjacent symbol pair with the **lowest merge
     rank** (rank = line index in the merges list; lower = higher priority), merge it, repeat
     until no adjacent pair is in the merge table. Ties → leftmost pair.
   - `fuse_unk=true` (mostly moot under byte_fallback, but documented).
3. **Map** the resulting tokens to ids via the vocab (token → id).
4. **Post-process** (`TemplateProcessing`, single = `[<s>, A]`): **prepend BOS id 1**. No EOS
   for a single sequence.

**Decode** (`decoder` = `Sequence`): ids → tokens → concatenate → `Replace "▁" -> " "` →
reassemble `<0xNN>` byte tokens into raw bytes (`ByteFallback`/`Fuse`) → strip the single
leading space. Special tokens are skipped.

## Where the vocab comes from

C++ stays dependency-free — **no JSON or SentencePiece-protobuf parsing in the engine**. A
small Python exporter (we already use Python for ground truth) reads `tokenizer.json` once and
writes a trivial, line-oriented UTF-8 format the C++ reads directly:

- `reference/tokenizer/vocab.txt` — 32000 lines, **one token per line, line index = id**
  (Llama ids are contiguous 0…31999).
- `reference/tokenizer/merges.txt` — 61249 lines, one `"a b"` pair per line, **line order =
  rank**. The single space is the delimiter; token halves never contain a literal space
  (spaces are `▁`).

**Escaping (discovered during implementation).** The original assumption that tokens are
"single-line safe" is **false**: 24 vocab tokens contain a literal carriage return (e.g.
`;\r`, `});\r` — learned from CRLF source), and many tokens (and merge halves) contain a
backslash. A literal-token line format would let the embedded `\r` collide with the line
separator, and `core.autocrlf=true` would rewrite separators to CRLF on checkout and
corrupt the C++ read. The exporter therefore writes both files with **C-style escaping**
(`\` → `\\`, CR → `\r`, LF → `\n`); ~167 of 32000 vocab lines and ~289 merge lines carry an
escape, every other line is byte-identical to the raw token (still easy to diff). A
committed `.gitattributes` pins `reference/tokenizer/*.txt` to `eol=lf` (safe now that no
literal CR remains) and marks `*.npy binary`. The C++ loader reverses the escaping with a
trivial unescape (`\\`→`\`, `\r`→CR, `\n`→LF).

**Decision (confirmed): commit these to `reference/`** alongside the other oracles, so
`test_tokenizer` is self-contained — it runs on a fresh clone and in CI **without** the 2 GB
`weights/` download. Combined size ≈ 1 MB, accepted as in the same spirit as the already-
committed `logits.npy`. (The alternative — generating them locally and gitignoring — was
rejected because it couples every test run to the full model download.)

> **Deviation from `atlas-repo-structure.jsx`:** the structure note says tokenizer loads
> "vocab from binary file." We use a **plain-text** export instead — easier to diff/debug on
> Windows and zero-dependency to parse. Recorded here as a deliberate choice; the
> `atlas-cpp-engine` skill gets updated when this lands.

## Files created in this step

| File | Contents |
|------|----------|
| `scripts/export_tokenizer.py` | Read `weights/.../tokenizer.json`, write `reference/tokenizer/vocab.txt` + `merges.txt` (C-style escaped). Asserts vocab=32000, contiguous ids, BOS/EOS/UNK ids = 1/2/0. |
| `.gitattributes` (new) | Pin `reference/tokenizer/*.txt` to `eol=lf`; mark `*.npy binary`. Protects oracle fixtures from `core.autocrlf` rewriting. |
| `engine/include/tokenizer.h` | `Tokenizer` class — `load`, `encode`, `decode`, special-token ids |
| `engine/src/tokenizer.cpp` | vocab/merges loading, normalization, BPE merge loop, byte fallback, BOS prepend, decode |
| `engine/tests/test_tokenizer.cpp` | encode vs `reference/token_ids.npy`, decode round-trip, edge cases (zero-dep `CHECK` harness, as in Step 1) |
| `engine/CMakeLists.txt` (edit) | add `src/tokenizer.cpp` to `atlas_engine`; add `test_tokenizer` target + `add_test` |

`tensor.*` is untouched. No `.npy` reader is needed yet — the test compares against the
literal id list `[1,450,7483,310,3444,338]` (re-deriving the `.npy` is a forward-validation
concern, Step "forward").

## Design decisions

- **Exact-match, correctness-first.** The bar is byte-for-byte parity with HF on the oracle,
  not "close." No heuristics; implement the real merge-rank algorithm.
- **Dependency-free C++.** Python does the one-time JSON parse; the engine reads flat text.
  No ICU — UTF-8 is handled manually (we only need codepoint splitting + byte fallback).
- **BPE via merge ranks**, not a re-derivation from frequencies. Initial symbols = vocab
  characters or `<0xNN>` byte tokens; merge loop picks the globally lowest rank each pass.
  A straightforward O(n²) loop over the (short) prompt is fine; correctness over speed.
- **`encode` prepends BOS by default** (`add_bos=true`) to match the oracle and the
  post-processor; expose the flag so later steps can opt out. `decode` skips specials and
  strips the one leading space introduced by normalization.
- **Raw string only** — no chat template (`<|user|>` …) this step, matching Step 0's prompt.

## Tokenizer API sketch (indicative, not final)

```cpp
// engine/include/tokenizer.h
struct Tokenizer {
    static constexpr int kUnkId = 0;
    static constexpr int kBosId = 1;
    static constexpr int kEosId = 2;

    // Load the vocab + merges exported by scripts/export_tokenizer.py.
    static Tokenizer load(const std::string& vocab_path, const std::string& merges_path);

    // Text -> ids. Prepends BOS (id 1) when add_bos (default), matching token_ids.npy.
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // ids -> text: ▁->space, byte tokens -> bytes, strip leading space, skip specials.
    std::string decode(const std::vector<int>& ids) const;

private:
    std::vector<std::string> id_to_token_;                    // index = id
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<uint64_t, int> merge_rank_;            // packed (lhs_id,rhs_id) -> rank
};
```

## Explicitly deferred (not this step)

- Chat-template / special-token-aware encoding (`<|user|>`, `<|assistant|>`) — after the base
  engine is proven correct.
- Reading `tokenizer.model` (SentencePiece protobuf) or `tokenizer.json` directly in C++ — the
  Python-exported flat files replace this.
- A C++ `.npy` reader → arrives with **forward-pass validation**.
- Any model math, quantization, or CUDA.

## Reference oracle

- `reference/prompt.txt` = `"The capital of France is"`.
- `reference/token_ids.npy` = `[1, 450, 7483, 310, 3444, 338]` (int32; BOS 1 prepended) —
  the primary `test_tokenizer` oracle.
- `reference/tokenizer/{vocab.txt,merges.txt}` — produced by `scripts/export_tokenizer.py`,
  the vocab/merge source the C++ tokenizer loads (token ids: `<unk>`=0, `<s>`=1, `</s>`=2;
  `▁The`=450, `▁capital`=7483, `▁of`=310, `▁France`=3444, `▁is`=338).
