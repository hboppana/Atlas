"""Step 1 validation — docs/15-rag-ingest-extraction.md § Validation.

Covers the five properties the design names: schema round-trip, idempotency, resumability,
failure recording, and hash invalidation. Plus the pure helpers (ID list parsing, Atom
parsing) that the network path is built out of, so `fetch` is covered without a socket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.ingest import (
    EXTRACTED,
    EXTRACTOR,
    FAILED,
    IngestError,
    Manifest,
    Paths,
    extract_papers,
    paper_id_for,
    parse_arxiv_atom,
    read_id_list,
    report,
    sha256_file,
)

from .conftest import NO_TEXT_LAYER, TWO_PAGE

PAPER_A = "2103.00020"
PAPER_B = "1706.03762"


def _silent(_: str) -> None:
    pass


def _manifest(paths: Paths) -> dict:
    return json.loads(paths.manifest_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# schema round-trip
# --------------------------------------------------------------------------------------


def test_extract_produces_the_documented_shape(paths: Paths, add_pdf) -> None:
    add_pdf(TWO_PAGE, PAPER_A)
    extract_papers(paths, log=_silent)

    document = json.loads(paths.document_path(PAPER_A).read_text(encoding="utf-8"))
    assert document["paper_id"] == PAPER_A
    assert document["extractor"] == EXTRACTOR
    assert document["page_count"] == 2
    assert document["source_path"].endswith(f"{PAPER_A}.pdf")
    assert set(document) == {
        "paper_id", "title", "authors", "year", "arxiv_id", "arxiv_categories",
        "source_path", "extractor", "page_count", "blocks",
    }
    # No `topic` field: topics are derived by clustering in Step 3, never declared here.
    assert "topic" not in document

    assert [block["page"] for block in document["blocks"]] == [1, 2]
    assert "page one" in document["blocks"][0]["text"]
    assert "page two" in document["blocks"][1]["text"]

    record = _manifest(paths)[PAPER_A]
    assert record["status"] == EXTRACTED
    assert record["pdf_sha256"] == sha256_file(paths.pdf_path(PAPER_A))
    assert record["extracted_at"].endswith("Z")
    assert record["error"] is None


def test_paper_metadata_flows_from_the_manifest_into_the_document(paths: Paths, add_pdf) -> None:
    """Fetch stores what arXiv asserted; extract copies it into the Step 2 contract."""
    add_pdf(TWO_PAGE, PAPER_A)
    Manifest.load(paths.manifest_path).record(
        PAPER_A,
        status="fetched",
        metadata={
            "title": "Learning Transferable Visual Models",
            "authors": ["Alec Radford", "Jong Wook Kim"],
            "year": 2021,
            "arxiv_id": f"{PAPER_A}v1",
            "arxiv_categories": ["cs.CV", "cs.LG"],
        },
    )
    extract_papers(paths, log=_silent)

    document = json.loads(paths.document_path(PAPER_A).read_text(encoding="utf-8"))
    assert document["title"] == "Learning Transferable Visual Models"
    assert document["authors"] == ["Alec Radford", "Jong Wook Kim"]
    assert document["year"] == 2021
    assert document["arxiv_categories"] == ["cs.CV", "cs.LG"]


def test_local_pdf_gets_a_content_hash_id(paths: Paths, add_pdf) -> None:
    pdf = add_pdf(TWO_PAGE, "some-local-paper")
    digest = sha256_file(pdf)
    assert paper_id_for(pdf, digest) == digest[:16]

    extract_papers(paths, log=_silent)
    assert paths.document_path(digest[:16]).exists()


# --------------------------------------------------------------------------------------
# idempotency, resumability, hash invalidation
# --------------------------------------------------------------------------------------


def test_second_run_is_a_no_op(paths: Paths, add_pdf) -> None:
    add_pdf(TWO_PAGE, PAPER_A)
    extract_papers(paths, log=_silent)
    first_document = paths.document_path(PAPER_A).read_bytes()
    first_extracted_at = _manifest(paths)[PAPER_A]["extracted_at"]

    logged: list[str] = []
    extract_papers(paths, log=logged.append)

    assert paths.document_path(PAPER_A).read_bytes() == first_document
    assert _manifest(paths)[PAPER_A]["extracted_at"] == first_extracted_at
    assert any("skip" in line for line in logged)


def test_force_re_extracts_to_identical_bytes(paths: Paths, add_pdf) -> None:
    """--force redoes the work; the document is byte-identical because nothing in it is
    time- or run-dependent. Only the manifest's extracted_at moves."""
    add_pdf(TWO_PAGE, PAPER_A)
    extract_papers(paths, log=_silent)
    first = paths.document_path(PAPER_A).read_bytes()

    extract_papers(paths, force=True, log=_silent)
    assert paths.document_path(PAPER_A).read_bytes() == first


def test_run_touches_only_the_new_paper(paths: Paths, add_pdf) -> None:
    add_pdf(TWO_PAGE, PAPER_A)
    extract_papers(paths, log=_silent)
    untouched_mtime = paths.document_path(PAPER_A).stat().st_mtime_ns

    add_pdf(TWO_PAGE, PAPER_B)
    logged: list[str] = []
    extract_papers(paths, log=logged.append)

    assert paths.document_path(PAPER_B).exists()
    assert paths.document_path(PAPER_A).stat().st_mtime_ns == untouched_mtime
    assert [line for line in logged if PAPER_B in line and "extract" in line]


def test_mutated_pdf_is_re_extracted(paths: Paths, add_pdf) -> None:
    add_pdf(TWO_PAGE, PAPER_A)
    extract_papers(paths, log=_silent)
    old_hash = _manifest(paths)[PAPER_A]["pdf_sha256"]

    add_pdf(NO_TEXT_LAYER, PAPER_A)  # same paper_id, different bytes
    extract_papers(paths, log=_silent)

    record = _manifest(paths)[PAPER_A]
    assert record["pdf_sha256"] != old_hash
    assert record["status"] == FAILED  # the new bytes have no text layer


# --------------------------------------------------------------------------------------
# failure recording
# --------------------------------------------------------------------------------------


def test_no_text_layer_is_a_recorded_failure(paths: Paths, add_pdf) -> None:
    add_pdf(NO_TEXT_LAYER, PAPER_A)
    extract_papers(paths, log=_silent)

    record = _manifest(paths)[PAPER_A]
    assert record["status"] == FAILED
    assert record["error"] and "no text layer" in record["error"]
    # The cardinal sin would be an empty document in the store; it must not exist.
    assert not paths.document_path(PAPER_A).exists()


def test_failure_removes_a_stale_document(paths: Paths, add_pdf) -> None:
    add_pdf(TWO_PAGE, PAPER_A)
    extract_papers(paths, log=_silent)
    assert paths.document_path(PAPER_A).exists()

    add_pdf(NO_TEXT_LAYER, PAPER_A)
    extract_papers(paths, log=_silent)
    assert not paths.document_path(PAPER_A).exists()


def test_a_broken_pdf_does_not_abort_the_run(paths: Paths, add_pdf) -> None:
    paths.pdf_path("broken").write_bytes(b"not a pdf at all")
    add_pdf(TWO_PAGE, PAPER_A)

    extract_papers(paths, log=_silent)

    manifest = _manifest(paths)
    assert manifest[PAPER_A]["status"] == EXTRACTED
    broken = [record for pid, record in manifest.items() if pid != PAPER_A]
    assert len(broken) == 1 and broken[0]["status"] == FAILED


def test_report_lists_counts_and_failures(paths: Paths, add_pdf) -> None:
    add_pdf(TWO_PAGE, PAPER_A)
    add_pdf(NO_TEXT_LAYER, PAPER_B)
    extract_papers(paths, log=_silent)

    text = report(paths)
    assert f"{EXTRACTED:<10} 1" in text
    assert f"{FAILED:<10} 1" in text
    assert PAPER_B in text and "no text layer" in text


def test_report_on_an_empty_corpus(paths: Paths) -> None:
    assert "nothing ingested yet" in report(paths)


# --------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------


def test_manifest_is_written_after_each_paper(paths: Paths, add_pdf) -> None:
    """The crash-safety property: the file on disk is current mid-run, not written at exit."""
    add_pdf(TWO_PAGE, PAPER_A)
    add_pdf(NO_TEXT_LAYER, PAPER_B)

    seen: list[int] = []

    def snapshot(_: str) -> None:
        if paths.manifest_path.exists():
            seen.append(len(_manifest(paths)))

    extract_papers(paths, log=snapshot)

    # One record on disk after the first paper's log line, both after the second.
    assert seen == [1, 2]


def test_corrupt_manifest_is_an_error_not_a_silent_reset(paths: Paths) -> None:
    paths.manifest_path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(IngestError, match="not valid JSON"):
        extract_papers(paths, log=_silent)


# --------------------------------------------------------------------------------------
# acquisition helpers (no network)
# --------------------------------------------------------------------------------------


def test_read_id_list_strips_comments_and_dedupes(tmp_path: Path) -> None:
    listing = tmp_path / "ids.txt"
    listing.write_text(
        "# header\n\n1706.03762   # Attention Is All You Need\n2005.11401\n1706.03762\n",
        encoding="utf-8",
    )
    assert read_id_list(listing) == ["1706.03762", "2005.11401"]


def test_read_id_list_rejects_a_non_id(tmp_path: Path) -> None:
    listing = tmp_path / "ids.txt"
    listing.write_text("attention-is-all-you-need\n", encoding="utf-8")
    with pytest.raises(IngestError, match="not an arXiv ID"):
        read_id_list(listing)


def test_the_checked_in_seed_list_parses() -> None:
    seed = Path(__file__).resolve().parents[2] / "scripts" / "arxiv_ids.txt"
    ids = read_id_list(seed)
    assert len(ids) >= 40


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


def test_parse_arxiv_atom(_atom: str = ATOM) -> None:
    metadata = parse_arxiv_atom(_atom.encode("utf-8"))
    entry = metadata["1706.03762"]  # keyed by the bare ID we asked for, not the versioned one
    assert entry["title"] == "Attention Is All You Need"  # whitespace normalized
    assert entry["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert entry["year"] == 2017
    assert entry["arxiv_id"] == "1706.03762v7"
    assert entry["arxiv_categories"] == ["cs.CL", "cs.LG"]
