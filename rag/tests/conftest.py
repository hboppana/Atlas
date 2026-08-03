"""Shared fixtures for rag/tests — docs/15-rag-ingest-extraction.md.

Two disciplines carried over from Phase 1/2: a missing optional dependency is a green SKIP
rather than a failure, and no test may touch the network or the gitignored corpus. Every
test runs against `rag/tests/fixtures/*.pdf` copied into a tmp_path corpus.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag.ingest import Paths

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
