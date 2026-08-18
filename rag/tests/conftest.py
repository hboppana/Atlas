"""Shared fixtures for rag/tests — docs/15-rag-ingest-extraction.md.

Two disciplines carried over from Phase 1/2: a missing optional dependency is a green SKIP
rather than a failure, and no test may touch the network or the gitignored corpus. Every
test runs against `rag/tests/fixtures/*.pdf` copied into a tmp_path corpus.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from rag.chunk import CHUNKER_VERSION, HEURISTIC_COUNTER
from rag.ingest import CHUNKED, Manifest, Paths, sha256_file, write_json

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TWO_PAGE = FIXTURES / "two_page.pdf"
NO_TEXT_LAYER = FIXTURES / "no_text_layer.pdf"


@pytest.fixture(scope="session", autouse=True)
def require_pypdf() -> None:
    pytest.importorskip("pypdf", reason="Phase 3 dependency; pip install -r requirements.txt")


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(data_dir=tmp_path / "data").ensure()


@pytest.fixture
def add_pdf(paths: Paths):
    """Drop a fixture PDF into the corpus under a chosen paper_id, returning its path."""

    def _add(source: Path, paper_id: str) -> Path:
        destination = paths.pdf_path(paper_id)
        shutil.copyfile(source, destination)
        return destination

    return _add


# --------------------------------------------------------------------------------------
# Step 3 fixtures — a synthetic corpus already at `chunked`, and a stub encoder
# --------------------------------------------------------------------------------------


def write_paper(
    paths: Paths,
    manifest: Manifest,
    paper_id: str,
    chunks: list[tuple[str, str]],
    *,
    title: str = "A Paper",
    year: int | None = 2024,
    categories: tuple[str, ...] = ("cs.CL", "cs.LG"),
    chunker_version: int = CHUNKER_VERSION,
) -> None:
    """Write the extracted document + chunk file a Step 2 run would have left behind, and
    record the paper as `chunked`. `chunks` is a list of (section, text)."""
    write_json(
        paths.document_path(paper_id),
        {
            "paper_id": paper_id,
            "title": title,
            "authors": ["A. Author"],
            "year": year,
            "arxiv_id": paper_id,
            "arxiv_categories": list(categories),
            "extractor": "pypdf",
            "page_count": max(1, len(chunks)),
            "blocks": [],
        },
    )
    write_json(
        paths.chunk_path(paper_id),
        {
            "paper_id": paper_id,
            "chunker_version": chunker_version,
            "token_counter": HEURISTIC_COUNTER,
            "source_sha256": "",
            "strategy": "sections",
            "sections_detected": len({section for section, _ in chunks}),
            "headers_accepted": 3,
            "dropped_chars": 0,
            "chunks": [
                {
                    "chunk_id": f"{paper_id}:{index}",
                    "chunk_index": index,
                    "text": text,
                    "section": section,
                    "section_index": index,
                    "page_span": [index + 1, index + 1],
                    "n_chars": len(text),
                    "n_tokens_est": len(text.split()),
                }
                for index, (section, text) in enumerate(chunks)
            ],
        },
    )
    manifest.record(
        paper_id,
        status=CHUNKED,
        extracted_sha256=sha256_file(paths.document_path(paper_id)),
        chunker_version=chunker_version,
        token_counter=HEURISTIC_COUNTER,
        chunk_count=len(chunks),
        chunk_strategy="sections",
        error=None,
    )


@pytest.fixture
def make_corpus(paths: Paths):
    """-> a callable that populates the tmp corpus with chunked papers."""

    def _make(papers: dict[str, list[tuple[str, str]]], **kwargs) -> Manifest:
        manifest = Manifest.load(paths.manifest_path)
        for paper_id, chunks in papers.items():
            write_paper(paths, manifest, paper_id, chunks, **kwargs)
        return manifest

    return _make


class StubEmbedder:
    """A deterministic encoder: SHA-256 of the text seeds a unit vector.

    Identical text embeds identically (so the query smoke test has an exact answer) and
    different text embeds to an essentially orthogonal direction — enough to exercise the
    cache format, the skip rules, the metadata contract and the whole Chroma write path
    without torch on the machine. `calls` counts encode batches, which is how the skip
    tests assert "no re-embedding happened".
    """

    dim = 384

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls = 0
        self.encoded: list[str] = []

    def vector(self, text: str):
        import numpy as np

        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        raw = np.random.default_rng(seed).standard_normal(self.dim).astype("float32")
        return raw / float(np.linalg.norm(raw))

    def encode(self, texts):
        import numpy as np

        self.calls += 1
        self.encoded.extend(texts)
        return np.stack([self.vector(text) for text in texts])

    def count_tokens(self, texts):
        return [len(text.split()) for text in texts]


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    pytest.importorskip("numpy", reason="Step 3 dependency; pip install -r requirements.txt")
    return StubEmbedder()
