"""Step 2 validation — docs/16-rag-chunking.md § Validation.

Chunking's input is JSON, so its fixtures are JSON: synthetic extracted-documents built
inline by `make_document`. No PDFs, no network, no corpus, and no ML dependency — the token
budget is exercised through the injectable counter, exactly as Step 3 will inject the real
MiniLM tokenizer.
"""

from __future__ import annotations

import json

import pytest

from rag.chunk import (
    CHUNKED,
    CHUNKER_VERSION,
    MIN_CHUNK_CHARS,
    TARGET_TOKENS,
    Line,
    chunk_document,
    chunk_papers,
    detect_sections,
    estimate_tokens,
    load_chunks,
    normalize,
)
from rag.ingest import FAILED, Manifest, Paths, write_json

# A paragraph of ~35 tokens under the default counter, so a handful of them cross the budget.
PARAGRAPH = (
    "The system processes each input sequence independently and reports the aggregate "
    "score. We evaluate on three benchmarks and observe consistent gains across every "
    "configuration we tried. Ablations isolate the contribution of each component."
)


def make_document(pages: list[str], paper_id: str = "1234.56789") -> dict:
    """The Step 1 output shape, with one block per page."""
    return {
        "paper_id": paper_id,
        "title": "A Synthetic Paper",
        "authors": ["A. Author"],
        "year": 2026,
        "arxiv_id": paper_id,
        "arxiv_categories": ["cs.CL"],
        "source_path": f"data/papers/{paper_id}.pdf",
        "extractor": "pypdf",
        "page_count": len(pages),
        "blocks": [{"page": number, "text": text} for number, text in enumerate(pages, start=1)],
    }


def sectioned_pages(body: str = PARAGRAPH) -> list[str]:
    """Five numbered sections spread over two pages — the shape 43/43 real papers have."""
    return [
        f"Abstract\n{body}\n1 Introduction\n{body}\n2 Related Work\n{body}",
        f"3 Method\n{body}\n4 Results\n{body}\n5 Conclusion\n{body}",
    ]


# --------------------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------------------


def test_dehyphenation_joins_a_wrapped_word():
    assert normalize("the MLM ob-\njective enables") == "the MLM objective enables"


def test_a_hyphenated_compound_keeps_its_hyphen():
    assert normalize("the state-\nof-the-art model") == "the state-of-the-art model"
    assert normalize("we use GPT-\nNeoX here") == "we use GPT-NeoX here"


def test_line_structure_survives_normalization():
    # Rejoining wrapped lines here would destroy the only signal a header has.
    assert normalize("1 Introduction\nRecurrent networks") == "1 Introduction\nRecurrent networks"


def test_ligatures_dashes_and_quotes_fold():
    assert normalize("the ﬁrst ﬂow") == "the first flow"
    assert normalize("pages 3–7 and “quoted” text") == 'pages 3-7 and "quoted" text'


def test_whitespace_collapses_without_losing_the_line_break():
    assert normalize("a   b \t c\n\n  d  ") == "a b c\n\nd"


# --------------------------------------------------------------------------------------
# section detection
# --------------------------------------------------------------------------------------


def _lines(*texts: str) -> list[Line]:
    return [Line(text, 1) for text in texts]


def test_numbered_headers_are_detected_in_order():
    payload = chunk_document(make_document(sectioned_pages()))
    assert payload["strategy"] == "sections"
    titles = [chunk["section"] for chunk in payload["chunks"]]
    assert titles[0] == "Abstract"
    # Abstract + 5 numbered sections, each contributing at least one chunk, in reading order.
    seen = list(dict.fromkeys(titles))
    assert seen == ["Abstract", "Introduction", "Related Work", "Method", "Results", "Conclusion"]


def test_out_of_sequence_candidates_are_rejected():
    # Table cells are header-shaped in isolation; only the monotonic run separates them.
    sections, accepted = detect_sections(
        _lines("4 Model", "12 OOM", "7 Baseline", *[PARAGRAPH] * 3)
    )
    assert accepted == 0
    assert sections == []


def test_a_missing_head_does_not_stall_the_run():
    # pypdf mangles roughly one head per paper; the run tolerates a small gap (measured:
    # 5/43 real papers lose exactly one head).
    _, accepted = detect_sections(_lines("1 Introduction", "2 Background", "4 Results", "5 End"))
    assert accepted == 4


def test_subnumbered_heads_attach_to_their_parent():
    payload = chunk_document(
        make_document(
            [
                f"1 Introduction\n{PARAGRAPH}\n2 Method\n{PARAGRAPH}\n"
                f"2.1 Encoder\n{PARAGRAPH}\n3 Results\n{PARAGRAPH}"
            ]
        )
    )
    assert "Encoder" not in {chunk["section"] for chunk in payload["chunks"]}
    assert "Encoder" in " ".join(chunk["text"] for chunk in payload["chunks"])


def test_small_caps_heads_are_cleaned():
    payload = chunk_document(
        make_document(
            [
                f"1 I NTRODUCTION\n{PARAGRAPH}\n2 R ELATED WORK\n{PARAGRAPH}\n"
                f"3 M ETHOD\n{PARAGRAPH}"
            ]
        )
    )
    assert payload["chunks"][-1]["section"] == "Method"


def test_header_free_document_falls_back_to_fixed_windows():
    payload = chunk_document(make_document([PARAGRAPH * 6, PARAGRAPH * 6]))
    assert payload["strategy"] == "fixed"
    assert payload["sections_detected"] == 0
    assert len(payload["chunks"]) > 1
    assert all(chunk["section"] == "" for chunk in payload["chunks"])


# --------------------------------------------------------------------------------------
# junk removal
# --------------------------------------------------------------------------------------


def test_references_and_everything_after_are_dropped():
    pages = sectioned_pages()
    pages.append(
        "References\n[1] A. Author. A cited paper nobody asked about. 2019.\n"
        "Appendix A\nresurrected appendix text"
    )
    payload = chunk_document(make_document(pages))
    body = " ".join(chunk["text"] for chunk in payload["chunks"])
    assert "cited paper nobody asked about" not in body
    assert "resurrected appendix text" not in body
    assert payload["dropped_chars"] > 0


def test_running_heads_page_numbers_and_stamps_are_dropped():
    head = "Preprint. Under review."
    pages = [f"{head}\narXiv:2501.01234v1 [cs.CL] 1 Jan 2025\n{PARAGRAPH}\n{number}" for number in range(1, 6)]
    payload = chunk_document(make_document(pages))
    body = " ".join(chunk["text"] for chunk in payload["chunks"])
    assert "arXiv:" not in body
    assert head not in body


# --------------------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------------------


def test_no_chunk_exceeds_the_token_budget():
    payload = chunk_document(make_document(sectioned_pages(PARAGRAPH * 8)))
    assert len(payload["chunks"]) > 3
    for chunk in payload["chunks"]:
        assert chunk["n_tokens_est"] <= TARGET_TOKENS
        assert estimate_tokens(chunk["text"]) == chunk["n_tokens_est"]


def test_an_injected_counter_is_what_the_budget_is_measured_in():
    # Step 3 passes the real MiniLM tokenizer through this seam; a harsher counter must
    # produce strictly more chunks over the same text.
    document = make_document(sectioned_pages(PARAGRAPH * 8))
    default = chunk_document(document)
    harsh = chunk_document(document, token_counter=lambda text: 4 * len(text.split()))
    assert len(harsh["chunks"]) > len(default["chunks"])
    assert all(chunk["n_tokens_est"] <= TARGET_TOKENS for chunk in harsh["chunks"])


def test_consecutive_chunks_in_a_section_overlap():
    payload = chunk_document(
        make_document(
            [f"1 Introduction\n{PARAGRAPH * 8}\n2 Method\n{PARAGRAPH}\n3 Results\n{PARAGRAPH}"]
        )
    )
    same_section = [chunk for chunk in payload["chunks"] if chunk["section"] == "Introduction"]
    assert len(same_section) > 1
    for first, second in zip(same_section, same_section[1:]):
        tail = first["text"].split()[-8:]
        assert " ".join(tail) in second["text"]


def test_only_a_final_chunk_may_fall_under_the_minimum():
    payload = chunk_document(make_document(sectioned_pages(PARAGRAPH * 8)))
    finals = {
        max(
            chunk["chunk_index"]
            for chunk in payload["chunks"]
            if chunk["section_index"] == section_index
        )
        for section_index in {chunk["section_index"] for chunk in payload["chunks"]}
    }
    for chunk in payload["chunks"]:
        if chunk["chunk_index"] not in finals:
            assert chunk["n_chars"] >= MIN_CHUNK_CHARS


# --------------------------------------------------------------------------------------
# provenance and schema
# --------------------------------------------------------------------------------------


def test_page_spans_are_measured_not_guessed():
    # Section 3 straddles the page boundary; sections 1-2 sit wholly on page 1.
    pages = [
        f"1 Introduction\n{PARAGRAPH}\n2 Background\n{PARAGRAPH}\n3 Method\n{PARAGRAPH}",
        f"{PARAGRAPH}\n4 Results\n{PARAGRAPH}",
    ]
    payload = chunk_document(make_document(pages, paper_id="2101.00001"))
    spans = {chunk["section"]: chunk["page_span"] for chunk in payload["chunks"]}
    assert spans["Introduction"] == [1, 1]
    assert spans["Method"] == [1, 2]
    assert spans["Results"] == [2, 2]


def test_chunk_ids_and_indices_are_contiguous():
    payload = chunk_document(make_document(sectioned_pages(), paper_id="2101.00002"))
    for index, chunk in enumerate(payload["chunks"]):
        assert chunk["chunk_index"] == index
        assert chunk["chunk_id"] == f"2101.00002:{index}"


def test_chunk_metadata_is_chroma_insertable():
    # Chroma rejects lists in a metadata dict; page_span is flattened at insert time in
    # Step 3, and everything else must already be scalar.
    chunk = chunk_document(make_document(sectioned_pages()))["chunks"][0]
    for key, value in chunk.items():
        if key == "page_span":
            assert [type(part) for part in value] == [int, int]
        else:
            assert isinstance(value, (str, int, float, bool)), key


def test_chunking_is_deterministic():
    document = make_document(sectioned_pages(PARAGRAPH * 4))
    assert json.dumps(chunk_document(document)) == json.dumps(chunk_document(document))


def test_an_empty_document_is_a_paper_failure_not_a_crash(paths: Paths):
    write_json(paths.document_path("2101.00003"), make_document(["", ""], "2101.00003"))
    manifest = chunk_papers(paths, log=lambda _: None)
    record = manifest.get("2101.00003")
    assert record["status"] == FAILED
    assert "chunk:" in record["error"]
    assert not paths.chunk_path("2101.00003").exists()


# --------------------------------------------------------------------------------------
# the corpus loop: manifest, skipping, invalidation
# --------------------------------------------------------------------------------------


@pytest.fixture
def corpus(paths: Paths) -> Paths:
    for paper_id in ("2101.00010", "2101.00011"):
        write_json(paths.document_path(paper_id), make_document(sectioned_pages(), paper_id))
    return paths


def test_chunk_papers_writes_files_and_manifest_records(corpus: Paths):
    manifest = chunk_papers(corpus, log=lambda _: None)
    record = manifest.get("2101.00010")
    assert record["status"] == CHUNKED
    assert record["chunk_strategy"] == "sections"
    assert record["chunker_version"] == CHUNKER_VERSION
    assert record["chunk_count"] == len(load_chunks(corpus, "2101.00010")["chunks"])
    assert record["chunked_at"].endswith("Z")


def test_a_second_run_is_a_no_op(corpus: Paths):
    chunk_papers(corpus, log=lambda _: None)
    before = corpus.chunk_path("2101.00010").read_bytes()
    stamp = Manifest.load(corpus.manifest_path).get("2101.00010")["chunked_at"]

    logged: list[str] = []
    chunk_papers(corpus, log=logged.append)
    assert sum("skip" in line for line in logged) == 2
    assert corpus.chunk_path("2101.00010").read_bytes() == before
    assert Manifest.load(corpus.manifest_path).get("2101.00010")["chunked_at"] == stamp


def test_a_bumped_chunker_version_rechunks(corpus: Paths):
    chunk_papers(corpus, log=lambda _: None)
    logged: list[str] = []
    manifest = chunk_papers(corpus, chunker_version=CHUNKER_VERSION + 1, log=logged.append)
    assert not any("skip" in line for line in logged)
    assert manifest.get("2101.00010")["chunker_version"] == CHUNKER_VERSION + 1
    assert load_chunks(corpus, "2101.00010")["chunker_version"] == CHUNKER_VERSION + 1


def test_the_default_counter_is_the_dependency_free_heuristic(corpus: Paths):
    """docs/17: the chunker's output must never depend on whether ML deps are installed."""
    from rag.chunk import HEURISTIC_COUNTER

    manifest = chunk_papers(corpus, log=lambda _: None)
    assert manifest.get("2101.00010")["token_counter"] == HEURISTIC_COUNTER
    assert load_chunks(corpus, "2101.00010")["token_counter"] == HEURISTIC_COUNTER


def test_swapping_the_token_counter_rechunks(corpus: Paths):
    """The seam that repairs a failed truncation check: same chunker version, different
    counter, so a heuristic-chunked file is never mistaken for an exactly-chunked one."""
    from rag.chunk import EXACT_COUNTER

    chunk_papers(corpus, log=lambda _: None)
    logged: list[str] = []
    manifest = chunk_papers(
        corpus,
        token_counter=lambda text: len(text.split()) * 3,  # stands in for the real tokenizer
        token_counter_name=EXACT_COUNTER,
        log=logged.append,
    )
    assert not any("skip" in line for line in logged)
    assert manifest.get("2101.00010")["token_counter"] == EXACT_COUNTER
    assert load_chunks(corpus, "2101.00010")["token_counter"] == EXACT_COUNTER

    # ...and running it again with the same counter is a no-op, not a permanent rewrite.
    logged.clear()
    chunk_papers(
        corpus,
        token_counter=lambda text: len(text.split()) * 3,
        token_counter_name=EXACT_COUNTER,
        log=logged.append,
    )
    assert sum("skip" in line for line in logged) == 2


def test_an_expensive_counter_bounds_chunks_by_its_own_measure(corpus: Paths):
    """A counter that reports 3x the heuristic must produce correspondingly smaller chunks —
    this is what makes injecting the real tokenizer actually fix over-budget chunks."""
    from rag.chunk import EXACT_COUNTER

    triple = lambda text: len(text.split()) * 3  # noqa: E731
    chunk_papers(corpus, token_counter=triple, token_counter_name=EXACT_COUNTER, log=lambda _: None)
    for chunk in load_chunks(corpus, "2101.00010")["chunks"]:
        assert triple(chunk["text"]) <= TARGET_TOKENS or len(chunk["text"].split()) == 1


def test_a_changed_extraction_rechunks(corpus: Paths):
    chunk_papers(corpus, log=lambda _: None)
    write_json(
        corpus.document_path("2101.00010"),
        make_document(sectioned_pages(PARAGRAPH * 3), "2101.00010"),
    )
    logged: list[str] = []
    chunk_papers(corpus, log=logged.append)
    assert any("chunk    2101.00010" in line for line in logged)
    assert any("skip     2101.00011" in line for line in logged)


def test_force_rechunks_everything(corpus: Paths):
    chunk_papers(corpus, log=lambda _: None)
    logged: list[str] = []
    chunk_papers(corpus, force=True, log=logged.append)
    assert not any("skip" in line for line in logged)


def test_the_chunk_file_records_the_source_it_was_built_from(corpus: Paths):
    from rag.ingest import sha256_file

    chunk_papers(corpus, log=lambda _: None)
    payload = load_chunks(corpus, "2101.00010")
    assert payload["source_sha256"] == sha256_file(corpus.document_path("2101.00010"))
