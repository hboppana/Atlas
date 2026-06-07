"""Export the TinyLlama tokenizer to flat text the C++ engine reads directly.

The engine stays dependency-free: no JSON or SentencePiece-protobuf parsing in C++.
This one-time exporter reads `weights/.../tokenizer.json` and writes two line-oriented
UTF-8 files into `reference/tokenizer/`:

  vocab.txt   32000 lines, one token per line, line index == token id. Llama ids are
              contiguous 0..31999, so the line number *is* the id.
  merges.txt  61249 lines, one "a b" pair per line. Line order == merge rank (lower =
              higher priority), which the BPE loop relies on. The single space is the
              delimiter; token halves never contain a literal space (spaces are U+2581).

Escaping: a handful of vocab tokens contain a literal carriage return (24 tokens, e.g.
`;\r`, learned from CRLF source) and many contain a backslash. A naive line format would
let these collide with the line separator or be mangled by git's autocrlf on checkout, so
both files are written with C-style escaping -- backslash -> `\\`, CR -> `\r`, LF -> `\n`
-- applied to each token. ~167 of 32000 vocab lines and ~289 merge lines carry an escape;
every other line is byte-identical to the raw token (so the files stay easy to diff). The
C++ loader reverses this with a trivial unescape. See docs/02-tokenizer.md.

Both are committed to `reference/` so `test_tokenizer` is self-contained -- it runs on a
fresh clone and in CI without the 2 GB `weights/` download (see docs/02-tokenizer.md).

Run from the repo root:  python scripts/export_tokenizer.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = REPO_ROOT / "weights" / "tinyllama-1.1b-chat" / "tokenizer.json"
OUT_DIR = REPO_ROOT / "reference" / "tokenizer"

VOCAB_SIZE = 32000
NUM_MERGES = 61249
# Identity checks so a wrong/updated tokenizer.json fails loudly instead of silently
# producing a mismatched export.
EXPECTED_SPECIALS = {"<unk>": 0, "<s>": 1, "</s>": 2}


def escape(token: str) -> str:
    r"""C-style escape so a token is safe on a single line: \ -> \\, CR -> \r, LF -> \n.

    Backslash must be escaped first so we don't double-escape the \r / \n we introduce.
    The C++ loader's unescape is the exact inverse.
    """
    return token.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def load_model(tokenizer_json: Path) -> dict:
    if not tokenizer_json.is_file():
        sys.exit(
            f"tokenizer.json not found at {tokenizer_json}\n"
            "Download the weights first (scripts/download_weights.py) or pass the path "
            "as the first argument."
        )
    with tokenizer_json.open(encoding="utf-8") as f:
        return json.load(f)["model"]


def build_vocab_lines(vocab: dict) -> list:
    """Invert {token: id} into a list indexed by id, asserting contiguity 0..N-1."""
    if len(vocab) != VOCAB_SIZE:
        sys.exit(f"expected {VOCAB_SIZE} vocab entries, got {len(vocab)}")

    id_to_token = [None] * VOCAB_SIZE
    for token, tid in vocab.items():
        if not 0 <= tid < VOCAB_SIZE:
            sys.exit(f"token {token!r} has out-of-range id {tid}")
        if id_to_token[tid] is not None:
            sys.exit(f"duplicate id {tid}: {id_to_token[tid]!r} and {token!r}")
        id_to_token[tid] = token

    missing = [i for i, t in enumerate(id_to_token) if t is None]
    if missing:
        sys.exit(f"vocab ids are not contiguous; missing ids: {missing[:10]}...")

    for token, expected_id in EXPECTED_SPECIALS.items():
        if vocab.get(token) != expected_id:
            sys.exit(f"special {token!r} expected id {expected_id}, got {vocab.get(token)}")

    # Control chars other than CR/LF would also be invisible/fragile in the line format;
    # this tokenizer has none, so fail loudly if that ever changes.
    for tid, token in enumerate(id_to_token):
        if any(ord(c) < 0x20 and c not in "\r\n" for c in token):
            sys.exit(f"token id {tid} {token!r} has an unexpected control char")

    return [escape(t) for t in id_to_token]


def build_merge_lines(merges: list) -> list:
    """Merges are "a b" strings; line order is rank. Validate the single-space delimiter
    on the raw string, then escape each half (halves never contain a literal space)."""
    if len(merges) != NUM_MERGES:
        sys.exit(f"expected {NUM_MERGES} merges, got {len(merges)}")
    lines = []
    for i, merge in enumerate(merges):
        if not isinstance(merge, str):
            sys.exit(f"merge {i} is {type(merge).__name__}, expected a 'a b' string")
        if merge.count(" ") != 1:
            sys.exit(f"merge {i} {merge!r} is not a single-space-separated pair")
        lhs, rhs = merge.split(" ")
        lines.append(f"{escape(lhs)} {escape(rhs)}")
    return lines


def write_lines(path: Path, lines: list) -> None:
    # Explicit UTF-8 + LF (newline="\n") so the engine reads identical bytes on every
    # platform; no BOM. C++ reads these in binary mode.
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def main() -> None:
    tokenizer_json = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOKENIZER
    model = load_model(tokenizer_json)

    if model.get("type") != "BPE":
        sys.exit(f"expected a BPE model, got {model.get('type')!r}")

    vocab_lines = build_vocab_lines(model["vocab"])
    merge_lines = build_merge_lines(model["merges"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_lines(OUT_DIR / "vocab.txt", vocab_lines)
    write_lines(OUT_DIR / "merges.txt", merge_lines)

    print(f"wrote {len(vocab_lines)} tokens  -> {OUT_DIR / 'vocab.txt'}")
    print(f"wrote {len(merge_lines)} merges -> {OUT_DIR / 'merges.txt'}")


if __name__ == "__main__":
    main()
