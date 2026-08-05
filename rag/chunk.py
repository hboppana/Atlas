"""Section-aware chunking — docs/16-rag-chunking.md.

    data/extracted/<paper_id>.json  ->  data/chunks/<paper_id>.json

Pure local text processing over JSON: no PDF is opened here, no vectors are produced, no
ML dependency is imported. Token budgets are honoured through an injectable
`token_counter`, defaulting to a word heuristic, so this module and its tests run on a
CPU-only checkout; Step 3 injects the real MiniLM tokenizer and the same code path becomes
exact.

The pipeline per paper:

    normalize   de-hyphenate, fold ligatures/dashes, collapse whitespace  (per page, keeps lines)
    strip junk  running heads/feet, page numbers, arXiv stamps
    detect      numbered headers gated on a monotonic run starting at 1
    truncate    everything from References/Bibliography onward is dropped
    split       paragraphs -> sentences -> ~200-token chunks with ~40 tokens of overlap

Page provenance is carried per character from the source blocks all the way to the emitted
chunk, so `page_span` is measured rather than guessed.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Callable, Sequence

from .ingest import (
    CHUNKED,
    FAILED,
    Log,
    Manifest,
    PaperError,
    Paths,
    _utc_now,
    sha256_file,
    write_json,
)

# Bumping this re-chunks the corpus on the next `--chunk` — the intended iteration loop.
CHUNKER_VERSION = 1

# all-MiniLM-L6-v2 truncates at 256 wordpieces; 200 leaves headroom for the estimate
# undercounting, and the overlap protects claims that straddle a boundary.
TARGET_TOKENS = 200
OVERLAP_TOKENS = 40
MIN_CHUNK_CHARS = 200

MAX_HEADER_WORDS = 12
MIN_SECTIONS = 3

# A head whose number is up to this far ahead of the expected one still continues the run:
# one head per paper is routinely mangled beyond recognition by pypdf, and stalling the run
# there costs the whole paper its structure. Larger jumps are table cells, not sections.
MAX_HEADER_GAP = 2

# A line repeated on more than this fraction of pages is a running head/foot, not content.
REPEATED_LINE_PAGE_FRACTION = 0.6
REPEATED_LINE_MAX_CHARS = 80
REPEATED_LINE_MIN_PAGES = 4

# A `References` line before this point in the document is a table-of-contents entry.
REFERENCES_MIN_POSITION = 0.3

TokenCounter = Callable[[str], int]


# --------------------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------------------

LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
# Unicode dashes/minus and the curly quotes pypdf passes straight through.
PUNCTUATION = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    " ": " ",
    " ": "\n",
    " ": "\n",
}

_HYPHEN_BREAK = re.compile(r"(\w)-\n[ \t]*(\w[^\s]*)")
_WHITESPACE = re.compile(r"[ \t\f\v]+")


def normalize(text: str) -> str:
    """Fold the artefacts of PDF text extraction, **preserving line structure**.

    Line structure is the only signal a header has, so rejoining soft-wrapped lines happens
    later, on section bodies (see `_paragraphs`). Doing it here would destroy detection.
    """
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    for source, replacement in {**LIGATURES, **PUNCTUATION}.items():
        text = text.replace(source, replacement)

    # De-hyphenate across a line break. `objec-\ntive` -> `objective`, but a compound such
    # as `state-\nof-the-art` keeps its hyphen: the giveaway is that the continuation is
    # itself hyphenated. Either way the two lines join — a wrapped word is never a header.
    def _join(match: re.Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        keep_hyphen = "-" in tail or not tail[:1].islower()
        return f"{head}-{tail}" if keep_hyphen else f"{head}{tail}"

    while True:
        joined = _HYPHEN_BREAK.sub(_join, text)
        if joined == text:
            break
        text = joined

    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------------------
# page-tracked text
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    text: str
    page: int


@dataclass(frozen=True)
class Passage:
    """Text plus, per character, the source page it came from."""

    text: str
    pages: tuple[int, ...]

    def __post_init__(self) -> None:
        assert len(self.text) == len(self.pages), "page map must be per character"

    @property
    def page_span(self) -> list[int]:
        return [min(self.pages), max(self.pages)] if self.pages else [0, 0]

    def slice(self, start: int, stop: int) -> "Passage":
        return Passage(self.text[start:stop], self.pages[start:stop])

    def strip(self) -> "Passage":
        start = len(self.text) - len(self.text.lstrip())
        stop = len(self.text.rstrip())
        return self.slice(start, max(start, stop))


def _passage_from_lines(lines: Sequence[Line], separator: str = " ") -> Passage:
    text: list[str] = []
    pages: list[int] = []
    for index, line in enumerate(lines):
        if index:
            text.append(separator)
            pages.extend([lines[index - 1].page] * len(separator))
        text.append(line.text)
        pages.extend([line.page] * len(line.text))
    return Passage("".join(text), tuple(pages))


def _join(passages: Sequence[tuple[Passage, str]]) -> Passage:
    """Join `(passage, separator-before)` pairs; the separator inherits the previous page."""
    text: list[str] = []
    pages: list[int] = []
    for index, (passage, separator) in enumerate(passages):
        if index and separator:
            text.append(separator)
            pages.extend([pages[-1] if pages else passage.pages[0]] * len(separator))
        text.append(passage.text)
        pages.extend(passage.pages)
    return Passage("".join(text), tuple(pages))


# --------------------------------------------------------------------------------------
# junk removal
# --------------------------------------------------------------------------------------

_PAGE_NUMBER = re.compile(r"^[\[\(]?\d{1,3}[\]\)]?$")
_ARXIV_STAMP = re.compile(r"^arXiv:\s?\d{4}\.\d{4,5}", re.IGNORECASE)
_PREPRINT_STAMP = re.compile(
    r"^(preprint|under review|published as a conference paper|accepted (at|to)|"
    r"to appear (in|at)|workshop paper)\b",
    re.IGNORECASE,
)


def _document_lines(document: dict) -> list[Line]:
    """Normalized lines across every page block, in reading order."""
    lines: list[Line] = []
    for block in document.get("blocks") or []:
        page = int(block.get("page") or 0)
        for line in normalize(block.get("text") or "").split("\n"):
            lines.append(Line(line, page))
    return lines


def _strip_junk(lines: Sequence[Line], page_count: int) -> tuple[list[Line], int]:
    """Drop running heads/feet, page numbers and preprint stamps. Returns dropped chars."""
    pages_by_line: dict[str, set[int]] = {}
    for line in lines:
        key = line.text.strip()
        if key and len(key) <= REPEATED_LINE_MAX_CHARS:
            pages_by_line.setdefault(key, set()).add(line.page)

    repeated = set()
    if page_count >= REPEATED_LINE_MIN_PAGES:
        threshold = REPEATED_LINE_PAGE_FRACTION * page_count
        repeated = {key for key, pages in pages_by_line.items() if len(pages) > threshold}

    kept: list[Line] = []
    dropped = 0
    for line in lines:
        text = line.text.strip()
        junk = (
            text in repeated
            or _PAGE_NUMBER.match(text)
            or _ARXIV_STAMP.match(text)
            or _PREPRINT_STAMP.match(text)
        )
        if junk:
            dropped += len(line.text)
            continue
        kept.append(line)
    return kept, dropped


# --------------------------------------------------------------------------------------
# section detection
# --------------------------------------------------------------------------------------

# `1 Introduction`, `2. Related Work`, `3.1 Encoder` — a title-cased head. The length and
# word caps are deliberately loose (measured: real heads run to
# `3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning`); the monotonic-run
# gate, not the regex, is what rejects the false positives.
_NUMBERED_HEADER = re.compile(r"^\s{0,3}(\d{1,2})((?:\.\d+)*)\.?\s+([A-Z].{2,100})$")
# pypdf renders small-caps heads as `1 I NTRODUCTION` — a split capital, not two words.
_SMALL_CAPS = re.compile(r"\b([A-Z]) ([A-Z]{2,})")
_SPACED_PUNCTUATION = re.compile(r"\s+([,.:;?!])")
_ABSTRACT = re.compile(r"^\s*(?:\d{1,2}\.?\s+)?abstract\s*[:.]?\s*$", re.IGNORECASE)
_REFERENCES = re.compile(
    r"^\s*(?:\d{1,2}\.?\s+)?(references|bibliography|works cited)\s*[:.]?\s*$", re.IGNORECASE
)
_TAIL_HEADS = re.compile(
    r"^\s*(?:\d{1,2}\.?\s+)?(acknowledgements|acknowledgments|acknowledgement|acknowledgment|"
    r"appendix(?:\s+[A-Z0-9][^\s]*)?(?:\s*[:.]\s*.{0,60})?)\s*[:.]?\s*$",
    re.IGNORECASE,
)

FRONT_MATTER = "Front Matter"


@dataclass
class Section:
    title: str
    index: int
    lines: list[Line]


def clean_title(title: str) -> str:
    """Undo the small-caps and spaced-punctuation artefacts of PDF heading extraction, so a
    citation reads `Self-Rag: Learning To Retrieve` rather than `S ELF -R AG: L EARNING ...`."""
    while True:
        merged = _SMALL_CAPS.sub(r"\1\2", title)
        if merged == title:
            break
        title = merged
    title = _SPACED_PUNCTUATION.sub(r"\1", title)
    title = " ".join(title.split()).strip(" .:")
    letters = [character for character in title if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        title = " ".join(word[:1].upper() + word[1:].lower() for word in title.split())
    return title


def _is_numbered_header(text: str) -> tuple[int, bool, str] | None:
    """-> (top-level number, is_sub_numbered, title) for a header-shaped line."""
    match = _NUMBERED_HEADER.match(text)
    if not match:
        return None
    number, subs, title = match.groups()
    title = clean_title(title)
    if len(title.split()) > MAX_HEADER_WORDS or len(title) < 3:
        return None
    if title.endswith((",", ";")):  # a wrapped sentence, not a head
        return None
    return int(number), bool(subs), title


def _cut_references(lines: list[Line]) -> tuple[list[Line], int]:
    """Drop everything from the References line onward — bibliography text matches every
    query about any cited work and answers none of them. An Appendix *after* it stays cut."""
    floor = int(len(lines) * REFERENCES_MIN_POSITION)
    for index in range(floor, len(lines)):
        if _REFERENCES.match(lines[index].text):
            dropped = sum(len(line.text) for line in lines[index:])
            return lines[:index], dropped
    return lines, 0


def detect_sections(lines: Sequence[Line]) -> tuple[list[Section], int]:
    """-> (sections, accepted top-level numbered headers).

    The monotonic-run gate does the real work: a candidate is accepted only when its
    top-level number continues an ascending run starting at 1. Table cells (`4 Model`,
    `12 OOM`) and figure captions are header-shaped in isolation but are never part of
    such a run, and no amount of per-line regex refinement separates them as cleanly.
    """
    heads: list[tuple[int, str]] = []  # (line index, title)
    expected = 1
    accepted = 0
    for index, line in enumerate(lines):
        text = line.text
        if _ABSTRACT.match(text):
            heads.append((index, "Abstract"))
            continue
        if _TAIL_HEADS.match(text):
            heads.append((index, text.strip().rstrip(":.")))
            continue
        parsed = _is_numbered_header(text)
        if parsed is None:
            continue
        number, is_sub, title = parsed
        if is_sub:
            continue  # 3.1 attaches to its parent; the retrievable unit is a top section
        gap = MAX_HEADER_GAP if accepted else 0  # the run must *start* at 1
        if not expected <= number <= expected + gap:
            continue
        heads.append((index, title))
        accepted += 1
        expected = number + 1

    if accepted < MIN_SECTIONS:
        return [], accepted

    sections: list[Section] = []
    front = [line for line in lines[: heads[0][0]] if line.text.strip()]
    if front:
        # Title/authors/affiliations — and, on the papers where `Abstract` is not its own
        # line, the abstract itself. Keeping it costs one chunk and never silently loses it.
        sections.append(Section(FRONT_MATTER, 0, front))

    for order, (start, title) in enumerate(heads):
        stop = heads[order + 1][0] if order + 1 < len(heads) else len(lines)
        body = [line for line in lines[start + 1 : stop] if line.text.strip()]
        if body:
            sections.append(Section(title, len(sections), body))
    for index, section in enumerate(sections):
        section.index = index
    return sections, accepted


# --------------------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Dependency-free stand-in for the MiniLM tokenizer, calibrated on a corpus sample."""
    return math.ceil(len(text.split()) * 1.35)


# Two fixed-width lookbehinds rather than one variable one: the closing quote/paren of a
# quoted sentence must stay with the sentence, not be eaten as separator.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"(\[]?[A-Z0-9])")
_ABBREVIATIONS = {
    "e.g.", "i.e.", "et", "al.", "fig.", "figs.", "eq.", "eqs.", "sec.", "secs.", "ref.",
    "refs.", "cf.", "vs.", "etc.", "no.", "vol.", "approx.", "resp.", "tab.", "ch.",
}


def _split_sentences(passage: Passage) -> list[Passage]:
    """Sentence boundaries, with the abbreviations that litter academic prose guarded."""
    out: list[Passage] = []
    start = 0
    for match in _SENTENCE_END.finditer(passage.text):
        head = passage.text[start : match.start()]
        last = head.split()[-1].lower() if head.split() else ""
        if last in _ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", last or "", re.IGNORECASE):
            continue
        out.append(passage.slice(start, match.start()).strip())
        start = match.end()
    tail = passage.slice(start, len(passage.text)).strip()
    if tail.text:
        out.append(tail)
    return [piece for piece in out if piece.text]


def _split_long(passage: Passage, token_counter: TokenCounter) -> list[Passage]:
    """A single sentence over budget (an equation-dense line): split at word boundaries."""
    words = list(re.finditer(r"\S+", passage.text))
    if len(words) <= 1:
        return [passage]
    pieces: list[Passage] = []
    start = words[0].start()
    count = 0
    for index, word in enumerate(words):
        count += 1
        over_budget = token_counter(passage.text[start : word.end()]) > TARGET_TOKENS
        if over_budget and count > 1:
            pieces.append(passage.slice(start, words[index - 1].end()).strip())
            start = word.start()
            count = 1
    tail = passage.slice(start, len(passage.text)).strip()
    if tail.text:
        pieces.append(tail)
    return pieces


def _paragraphs(lines: Sequence[Line]) -> list[Passage]:
    """Rejoin soft-wrapped lines into paragraphs.

    pypdf gives hard-wrapped near-full-width lines and no blank line between paragraphs, so
    the boundary signal is a *short* line that ends a sentence — the last line of a
    paragraph rarely fills the column.
    """
    widths = [len(line.text) for line in lines if line.text.strip()]
    median = statistics.median(widths) if widths else 1.0
    out: list[Passage] = []
    current: list[Line] = []

    def flush() -> None:
        if current:
            passage = _passage_from_lines(current).strip()
            if passage.text:
                out.append(passage)
            current.clear()

    for line in lines:
        if not line.text.strip():
            flush()
            continue
        current.append(line)
        ends_sentence = line.text.rstrip().endswith((".", "?", "!", ":"))
        if ends_sentence and len(line.text) < 0.75 * median:
            flush()
    flush()
    return out


@dataclass
class _Atom:
    """The smallest unit the packer moves: a paragraph, or a piece of an oversized one."""

    passage: Passage
    tokens: int
    opens_paragraph: bool


def _atoms(paragraphs: Sequence[Passage], token_counter: TokenCounter) -> list[_Atom]:
    atoms: list[_Atom] = []
    for paragraph in paragraphs:
        if token_counter(paragraph.text) <= TARGET_TOKENS:
            atoms.append(_Atom(paragraph, token_counter(paragraph.text), True))
            continue
        first = True
        for sentence in _split_sentences(paragraph):
            pieces = (
                [sentence]
                if token_counter(sentence.text) <= TARGET_TOKENS
                else _split_long(sentence, token_counter)
            )
            for piece in pieces:
                atoms.append(_Atom(piece, token_counter(piece.text), first))
                first = False
    return atoms


def _render(atoms: Sequence[_Atom]) -> Passage:
    parts = [
        (atom.passage, "" if index == 0 else ("\n\n" if atom.opens_paragraph else " "))
        for index, atom in enumerate(atoms)
    ]
    return _join(parts)


def _overlap(atoms: Sequence[_Atom]) -> list[_Atom]:
    """Carry the tail of the emitted chunk into the next one, at sentence granularity, so a
    claim spanning the boundary survives whole somewhere. Never carries the whole chunk."""
    carried: list[_Atom] = []
    total = 0
    for atom in reversed(atoms[1:]):
        if total + atom.tokens > OVERLAP_TOKENS:
            break  # a tail that does not fit the overlap budget is simply not carried
        carried.insert(0, atom)
        total += atom.tokens
    return carried


def _pack(atoms: Sequence[_Atom], token_counter: TokenCounter) -> list[Passage]:
    """Greedy packing to TARGET_TOKENS with overlap; paragraph boundaries are preferred
    because atoms are whole paragraphs until one does not fit on its own."""
    if not atoms:
        return []

    packed: list[tuple[list[_Atom], int]] = []  # (atoms, count carried in from the previous)
    current: list[_Atom] = []
    carried = 0
    total = 0
    index = 0
    while index < len(atoms):
        atom = atoms[index]
        if current and total + atom.tokens > TARGET_TOKENS:
            if len(current) > carried:
                packed.append((current, carried))
                current = _overlap(current)
            else:
                # Nothing new has been added yet: the carried tail alone leaves no room for
                # the next atom, so drop the overlap rather than blow the token budget.
                current = []
            carried = len(current)
            total = sum(item.tokens for item in current)
            continue
        current.append(atom)
        total += atom.tokens
        index += 1
    if len(current) > carried:
        packed.append((current, carried))

    # A short trailing remainder is appended to the previous chunk (its new atoms only —
    # the carried ones are already there) rather than emitted as a stub.
    if len(packed) > 1:
        last, last_carried = packed[-1]
        previous, previous_carried = packed[-2]
        new_atoms = last[last_carried:]
        merged_tokens = sum(atom.tokens for atom in previous) + sum(a.tokens for a in new_atoms)
        if len(_render(last).text) < MIN_CHUNK_CHARS and merged_tokens <= TARGET_TOKENS:
            packed[-2] = (previous + new_atoms, previous_carried)
            packed.pop()

    return [_render(group) for group, _ in packed]


# --------------------------------------------------------------------------------------
# document -> chunk file
# --------------------------------------------------------------------------------------


def chunk_document(
    document: dict,
    *,
    source_sha256: str = "",
    token_counter: TokenCounter = estimate_tokens,
    chunker_version: int = CHUNKER_VERSION,
) -> dict:
    """Extracted document -> the chunk-file payload. Pure, deterministic, no I/O."""
    paper_id = document.get("paper_id")
    if not paper_id:
        raise PaperError("extracted document has no paper_id")

    page_count = int(document.get("page_count") or 0)
    lines, dropped = _strip_junk(_document_lines(document), page_count)
    lines, dropped_refs = _cut_references(lines)
    dropped += dropped_refs

    sections, accepted = detect_sections(lines)
    strategy = "sections" if sections else "fixed"
    if not sections:
        body = [line for line in lines if line.text.strip()]
        sections = [Section("", 0, body)] if body else []

    chunks: list[dict] = []
    for section in sections:
        passages = _pack(_atoms(_paragraphs(section.lines), token_counter), token_counter)
        for passage in passages:
            if not passage.text.strip():
                continue
            chunks.append(
                {
                    "chunk_id": f"{paper_id}:{len(chunks)}",
                    "chunk_index": len(chunks),
                    "text": passage.text,
                    "section": section.title if strategy == "sections" else "",
                    "section_index": section.index,
                    "page_span": passage.page_span,
                    "n_chars": len(passage.text),
                    "n_tokens_est": token_counter(passage.text),
                }
            )

    if not chunks:
        raise PaperError("no chunks produced (empty or unusable text)")

    return {
        "paper_id": paper_id,
        "chunker_version": chunker_version,
        "source_sha256": source_sha256,
        "strategy": strategy,
        "sections_detected": len(sections) if strategy == "sections" else 0,
        "headers_accepted": accepted,
        "dropped_chars": dropped,
        "chunks": chunks,
    }


def chunk_papers(
    paths: Paths,
    *,
    force: bool = False,
    token_counter: TokenCounter = estimate_tokens,
    chunker_version: int = CHUNKER_VERSION,
    log: Log = print,
) -> Manifest:
    """Chunk every extracted document that is new, changed, or built by an older chunker."""
    paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(paths.manifest_path)
    documents = sorted(paths.extracted_dir.glob("*.json"))
    if not documents:
        log(f"chunk: no extracted documents under {paths.extracted_dir}")
        return manifest

    log(f"chunk: {len(documents)} extracted document(s) on disk")
    for document_path in documents:
        paper_id = document_path.stem
        digest = sha256_file(document_path)
        record = manifest.get(paper_id) or {}
        chunk_path = paths.chunk_path(paper_id)

        current = (
            record.get("status") == CHUNKED
            and record.get("extracted_sha256") == digest
            and record.get("chunker_version") == chunker_version
            and chunk_path.exists()
        )
        if not force and current:
            log(f"  skip     {paper_id}  (unchanged)")
            continue

        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
            payload = chunk_document(
                document,
                source_sha256=digest,
                token_counter=token_counter,
                chunker_version=chunker_version,
            )
        except (PaperError, json.JSONDecodeError) as exc:
            chunk_path.unlink(missing_ok=True)  # never leave a stale chunk file behind
            manifest.record(
                paper_id,
                status=FAILED,
                extracted_sha256=digest,
                error=f"chunk: {exc}",
            )
            log(f"  FAILED   {paper_id}  {exc}")
            continue

        write_json(chunk_path, payload)
        manifest.record(
            paper_id,
            status=CHUNKED,
            extracted_sha256=digest,
            chunked_at=_utc_now(),
            chunker_version=chunker_version,
            chunk_count=len(payload["chunks"]),
            chunk_strategy=payload["strategy"],
            error=None,
        )
        log(
            f"  chunk    {paper_id}  {len(payload['chunks'])} chunk(s), "
            f"{payload['strategy']} ({payload['sections_detected']} section(s))"
        )

    return manifest


def load_chunks(paths: Paths, paper_id: str) -> dict:
    """Read one chunk file — the entry point Step 3's embedder consumes."""
    path = paths.chunk_path(paper_id)
    if not path.exists():
        raise PaperError(f"no chunk file for {paper_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CHUNKED",
    "CHUNKER_VERSION",
    "MIN_CHUNK_CHARS",
    "OVERLAP_TOKENS",
    "TARGET_TOKENS",
    "Line",
    "Passage",
    "Section",
    "chunk_document",
    "chunk_papers",
    "detect_sections",
    "estimate_tokens",
    "load_chunks",
    "normalize",
]
