"""Corpus acquisition + PDF -> text extraction — docs/15-rag-ingest-extraction.md.

Two phases that share one manifest and are driven by scripts/ingest_papers.py:

    fetch_papers()    network: arXiv metadata + PDF bytes  -> data/papers/*.pdf
    extract_papers()  local:   pypdf page text             -> data/extracted/*.json

Both are idempotent and resumable. `data/manifest.json` is written after every paper, so
a crash loses at most one; a paper that fails is recorded with `status: "failed"` and a
stated reason rather than vanishing. There is no OCR path — a PDF with no text layer is a
recorded failure, because ingesting it empty produces a paper that exists in the store and
returns nothing from retrieval forever.

Nothing here keys off subject matter, and nothing assigns a topic: extraction records only
what the source asserts (`arxiv_categories`). Topics are derived by clustering in Step 3.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:  # pypdf is a Phase 3 dependency; import failure is reported at use, not at import.
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised only on a checkout without pypdf
    PdfReader = None

# rag/ingest.py -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

EXTRACTOR = "pypdf"

# Manifest statuses. `fetched` means bytes are on disk; `extracted` means data/extracted/
# holds a document for the *current* PDF hash; `chunked` means data/chunks/ does too, and
# `embedded` means data/embeddings/ does as well — the statuses are a pipeline position,
# not a set of flags, so `embedded` implies `chunked` implies `extracted`.
FETCHED = "fetched"
EXTRACTED = "extracted"
CHUNKED = "chunked"
EMBEDDED = "embedded"
FAILED = "failed"

# Extraction is done for all of these; only the downstream stage differs.
EXTRACTED_STATUSES = (EXTRACTED, CHUNKED, EMBEDDED)
# ...and chunking is done for these, which is why `--embed` does not undo `--chunk`'s skip.
CHUNKED_STATUSES = (CHUNKED, EMBEDDED)

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{paper_id}"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

# arXiv asks for >=3 s between programmatic requests; being a good citizen is cheaper than
# being rate-limited halfway through a 40-paper run.
DEFAULT_DELAY = 3.0
HTTP_TIMEOUT = 60.0
USER_AGENT = "atlas-ingest/0.1 (+https://github.com/hboppana/Atlas)"
METADATA_BATCH = 50

# A scanned paper is not always *exactly* zero characters — a title stamp or a page number
# can survive. Below this many characters over the whole PDF there is no usable text layer.
MIN_TEXT_CHARS = 32

# Only used to flag suspicious papers in --report: a 12-page paper chunked into 3 passages
# (detection collapsed) or 900 (detection shattered) is a bug, not a short paper.
CHUNK_COUNT_FLOOR = 8
CHUNK_COUNT_CEIL = 400

Log = Callable[[str], None]


class IngestError(RuntimeError):
    """Aborts the run (bad input, missing dependency) — as opposed to a per-paper failure."""


class PaperError(RuntimeError):
    """One paper failed. Recorded in the manifest; the run continues."""


@dataclass(frozen=True)
class Paths:
    """Everything under data/. Tests point this at a tmp_path; the CLI uses default()."""

    data_dir: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(data_dir=REPO_ROOT / "data")

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def topics_path(self) -> Path:
        return self.data_dir / "topics.json"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    def ensure(self) -> "Paths":
        for directory in (self.papers_dir, self.extracted_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def chunk_path(self, paper_id: str) -> Path:
        return self.chunks_dir / f"{paper_id}.json"

    def embedding_path(self, paper_id: str) -> Path:
        return self.embeddings_dir / f"{paper_id}.npz"

    def pdf_path(self, paper_id: str) -> Path:
        return self.papers_dir / f"{paper_id}.pdf"

    def document_path(self, paper_id: str) -> Path:
        return self.extracted_dir / f"{paper_id}.json"


# --------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------


class Manifest:
    """paper_id -> record. Persisted on every mutation, so a crash costs one paper."""

    def __init__(self, path: Path, records: dict[str, dict] | None = None) -> None:
        self.path = path
        self.records: dict[str, dict] = records or {}

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if not path.exists():
            return cls(path)
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(records, dict):
            raise IngestError(f"{path} must hold a JSON object of paper_id -> record")
        return cls(path, records)

    def get(self, paper_id: str) -> dict | None:
        return self.records.get(paper_id)

    def save(self) -> None:
        write_json(self.path, dict(sorted(self.records.items())))

    def record(self, paper_id: str, **fields) -> dict:
        """Merge `fields` into the record and flush. Keys with value None are kept (the
        schema is fixed) — callers pass error=None explicitly to clear a previous failure."""
        record = dict(self.records.get(paper_id) or {})
        record.update(fields)
        self.records[paper_id] = record
        self.save()
        return record

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records.values():
            status = record.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def failures(self) -> list[tuple[str, str]]:
        return [
            (paper_id, record.get("error") or "unknown error")
            for paper_id, record in sorted(self.records.items())
            if record.get("status") == FAILED
        ]


def write_json(path: Path, payload) -> None:
    """Atomic, deterministic JSON write — same bytes for the same object, every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def paper_id_for(pdf_path: Path, pdf_sha256: str) -> str:
    """arXiv ID when the filename is one (the fetch path names files that way), otherwise a
    content hash — a local PDF dropped into data/papers/ still gets a stable join key."""
    stem = pdf_path.stem
    return stem if ARXIV_ID_RE.match(stem) else pdf_sha256[:16]


# --------------------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------------------


def read_id_list(path: Path) -> list[str]:
    """One arXiv ID per line; `#` comments and blanks ignored, order and dedupe preserved."""
    if not path.exists():
        raise IngestError(f"ID list not found: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not ARXIV_ID_RE.match(line):
            raise IngestError(f"{path}:{lineno}: {line!r} is not an arXiv ID (e.g. 1706.03762)")
        if line not in seen:
            seen.add(line)
            ids.append(line)
    return ids


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise PaperError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PaperError(f"network error for {url}: {exc}") from exc


def parse_arxiv_atom(payload: bytes) -> dict[str, dict]:
    """arXiv Atom feed -> {bare_id: metadata}. Versions are stripped from the key so a
    v2 answer still matches the v-less ID we asked for."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise PaperError(f"arXiv returned unparseable XML: {exc}") from exc

    out: dict[str, dict] = {}
    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = (entry.findtext(f"{ATOM_NS}id") or "").rsplit("/", 1)[-1]
        if not raw_id:
            continue
        bare_id = raw_id.split("v")[0]
        published = entry.findtext(f"{ATOM_NS}published") or ""
        title = " ".join((entry.findtext(f"{ATOM_NS}title") or "").split())
        out[bare_id] = {
            "title": title,
            "authors": [
                " ".join((author.findtext(f"{ATOM_NS}name") or "").split())
                for author in entry.findall(f"{ATOM_NS}author")
            ],
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "arxiv_id": raw_id,
            "arxiv_categories": [
                term
                for category in entry.findall(f"{ATOM_NS}category")
                if (term := category.get("term"))
            ],
        }
    return out


def fetch_metadata(paper_ids: Sequence[str], *, delay: float = DEFAULT_DELAY) -> dict[str, dict]:
    """Batched arXiv API lookup. A batch that fails leaves those papers metadata-less; the
    PDF fetch still runs and the document simply carries nulls, which --report makes visible."""
    metadata: dict[str, dict] = {}
    for start in range(0, len(paper_ids), METADATA_BATCH):
        batch = paper_ids[start : start + METADATA_BATCH]
        url = f"{ARXIV_API}?id_list={','.join(batch)}&max_results={len(batch)}"
        metadata.update(parse_arxiv_atom(_http_get(url)))
        if start + METADATA_BATCH < len(paper_ids):
            time.sleep(delay)
    return metadata


def fetch_papers(
    paper_ids: Iterable[str],
    paths: Paths,
    *,
    force: bool = False,
    delay: float = DEFAULT_DELAY,
    log: Log = print,
) -> Manifest:
    """Download PDFs + arXiv metadata. Skips anything already on disk with a matching hash."""
    paths.ensure()
    manifest = Manifest.load(paths.manifest_path)

    todo: list[str] = []
    for paper_id in paper_ids:
        record = manifest.get(paper_id)
        pdf_path = paths.pdf_path(paper_id)
        if not force and record and pdf_path.exists() and record.get("pdf_sha256") == sha256_file(pdf_path):
            log(f"  skip     {paper_id}  (already fetched)")
            continue
        todo.append(paper_id)

    if not todo:
        log("fetch: nothing to do")
        return manifest

    log(f"fetch: {len(todo)} paper(s)")
    try:
        metadata = fetch_metadata(todo, delay=delay)
    except PaperError as exc:  # one bad batch must not sink the whole run
        log(f"  warning  arXiv metadata lookup failed: {exc}")
        metadata = {}

    for index, paper_id in enumerate(todo):
        if index:
            time.sleep(delay)
        pdf_path = paths.pdf_path(paper_id)
        try:
            payload = _http_get(ARXIV_PDF.format(paper_id=paper_id))
            if not payload.startswith(b"%PDF"):
                raise PaperError("response body is not a PDF (arXiv may be rate-limiting)")
            pdf_path.write_bytes(payload)
        except PaperError as exc:
            manifest.record(paper_id, status=FAILED, error=f"fetch: {exc}")
            log(f"  FAILED   {paper_id}  {exc}")
            continue

        paper_metadata = metadata.get(paper_id)
        manifest.record(
            paper_id,
            status=FETCHED,
            pdf_sha256=sha256_file(pdf_path),
            source_path=str(pdf_path.relative_to(REPO_ROOT)) if _under(pdf_path, REPO_ROOT) else str(pdf_path),
            metadata=paper_metadata,
            error=None,
        )
        title = (paper_metadata or {}).get("title") or "(no metadata)"
        log(f"  fetched  {paper_id}  {title[:70]}")

    return manifest


def _under(path: Path, root: Path) -> bool:
    return root == path or root in path.parents


# --------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def read_pdf_pages(pdf_path: Path) -> list[str]:
    """Per-page text, in page order. Empty pages are kept so block index == page order."""
    if PdfReader is None:
        raise IngestError("pypdf is not installed — `pip install -r requirements.txt`")
    try:
        reader = PdfReader(str(pdf_path))
        return [_normalize(page.extract_text() or "") for page in reader.pages]
    except IngestError:
        raise
    except Exception as exc:  # pypdf raises a wide, version-dependent set of exceptions
        raise PaperError(f"{type(exc).__name__}: {exc}") from exc


def extract_document(pdf_path: Path, paper_id: str, metadata: dict | None) -> dict:
    """The Step 2 contract: paper-level metadata + ordered page blocks. Raises PaperError
    for a PDF with no text layer — see the no-OCR decision in docs/15."""
    pages = read_pdf_pages(pdf_path)
    if sum(len(page) for page in pages) < MIN_TEXT_CHARS:
        raise PaperError(f"no text layer ({len(pages)} page(s), scanned?); OCR is not supported")

    metadata = metadata or {}
    return {
        "paper_id": paper_id,
        "title": metadata.get("title"),
        "authors": metadata.get("authors") or [],
        "year": metadata.get("year"),
        "arxiv_id": metadata.get("arxiv_id"),
        "arxiv_categories": metadata.get("arxiv_categories") or [],
        "source_path": str(pdf_path.relative_to(REPO_ROOT)) if _under(pdf_path, REPO_ROOT) else str(pdf_path),
        "extractor": EXTRACTOR,
        "page_count": len(pages),
        "blocks": [{"page": number, "text": text} for number, text in enumerate(pages, start=1)],
    }


def extract_papers(paths: Paths, *, force: bool = False, log: Log = print) -> Manifest:
    """Extract every PDF in data/papers/ that is new, changed, or not yet extracted."""
    paths.ensure()
    manifest = Manifest.load(paths.manifest_path)
    pdfs = sorted(paths.papers_dir.glob("*.pdf"))
    if not pdfs:
        log(f"extract: no PDFs under {paths.papers_dir}")
        return manifest

    log(f"extract: {len(pdfs)} PDF(s) on disk")
    for pdf_path in pdfs:
        digest = sha256_file(pdf_path)
        paper_id = paper_id_for(pdf_path, digest)
        record = manifest.get(paper_id) or {}
        document_path = paths.document_path(paper_id)

        unchanged = record.get("pdf_sha256") == digest
        if (
            not force
            and unchanged
            and record.get("status") in EXTRACTED_STATUSES
            and document_path.exists()
        ):
            log(f"  skip     {paper_id}  (unchanged)")
            continue

        try:
            document = extract_document(pdf_path, paper_id, record.get("metadata"))
        except PaperError as exc:
            document_path.unlink(missing_ok=True)  # never leave a stale document behind
            manifest.record(
                paper_id,
                status=FAILED,
                pdf_sha256=digest,
                extractor=EXTRACTOR,
                error=f"extract: {exc}",
            )
            log(f"  FAILED   {paper_id}  {exc}")
            continue

        write_json(document_path, document)
        manifest.record(
            paper_id,
            status=EXTRACTED,
            pdf_sha256=digest,
            extracted_at=_utc_now(),
            extractor=EXTRACTOR,
            error=None,
        )
        log(f"  extract  {paper_id}  {document['page_count']} page(s)")

    return manifest


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def report(paths: Paths) -> str:
    """Counts by status plus every failure with its reason. The whole ingestion UI."""
    manifest = Manifest.load(paths.manifest_path)
    if not manifest.records:
        return f"manifest {paths.manifest_path} is empty — nothing ingested yet"

    counts = manifest.counts()
    lines = [f"{paths.manifest_path}: {len(manifest.records)} paper(s)"]
    lines += [f"  {status:<10} {count}" for status, count in sorted(counts.items())]

    chunked = [
        record for record in manifest.records.values() if record.get("status") in CHUNKED_STATUSES
    ]
    if chunked:
        chunk_counts = sorted(record.get("chunk_count") or 0 for record in chunked)
        strategies: dict[str, int] = {}
        counters: dict[str, int] = {}
        for record in chunked:
            strategy = record.get("chunk_strategy") or "unknown"
            strategies[strategy] = strategies.get(strategy, 0) + 1
            counter = record.get("token_counter") or "unknown"
            counters[counter] = counters.get(counter, 0) + 1
        lines.append("")
        lines.append(
            f"chunks: {sum(chunk_counts)} across {len(chunked)} paper(s) "
            f"(min {chunk_counts[0]}, median {chunk_counts[len(chunk_counts) // 2]}, "
            f"max {chunk_counts[-1]})"
        )
        lines += [f"  {strategy:<10} {count}" for strategy, count in sorted(strategies.items())]
        # Which token counter measured the budget — the difference between "200 estimated
        # tokens" and "200 wordpieces the encoder will actually see".
        lines += [f"  {counter:<10} {count}" for counter, count in sorted(counters.items())]
        # A paper at either extreme is a detection bug wearing a plausible number.
        outliers = sorted(
            (
                (paper_id, record.get("chunk_count") or 0, record.get("chunk_strategy") or "?")
                for paper_id, record in manifest.records.items()
                if record.get("status") in CHUNKED_STATUSES
                and not (CHUNK_COUNT_FLOOR <= (record.get("chunk_count") or 0) <= CHUNK_COUNT_CEIL)
            ),
            key=lambda row: row[1],
        )
        if outliers:
            lines.append("  outliers (chunk count outside "
                         f"{CHUNK_COUNT_FLOOR}..{CHUNK_COUNT_CEIL}):")
            lines += [f"    {paper_id:<14} {count:>4}  {strategy}" for paper_id, count, strategy in outliers]

    # Embeddings, topics and the collection size — imported here rather than at module
    # scope so `--report` still works on a checkout with neither numpy nor chromadb.
    try:
        from .embed import report_lines
    except ImportError as exc:
        lines.append("")
        lines.append(f"embeddings: unavailable ({exc})")
    else:
        lines += report_lines(paths)

    failures = manifest.failures()
    if failures:
        lines.append("")
        lines.append("failures:")
        lines += [f"  {paper_id:<14} {error}" for paper_id, error in failures]
    return "\n".join(lines)
