"""Step 3 store validation — docs/17-rag-embeddings-store.md § Validation.

Chroma is exercised for real (a `PersistentClient` under tmp_path) but the *encoder* is
always the stub, so these tests need chromadb and numpy and nothing heavier. The rules under
test are the two that docs/17 calls load-bearing: writes are delete-then-add, and no query
in Atlas ever lets Chroma embed anything itself.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy", reason="Step 3 dependency; pip install -r requirements.txt")
pytest.importorskip("chromadb", reason="Step 3 dependency; pip install -r requirements.txt")

from rag.chunk import CHUNKER_VERSION  # noqa: E402
from rag.embed import chunk_metadata, embed_papers, query_store  # noqa: E402
from rag.ingest import IngestError, Manifest, Paths  # noqa: E402
from rag.store import ChunkStore  # noqa: E402

CORPUS = {
    "2401.00001": [
        ("Abstract", "Sparse attention kernels reduce memory traffic on accelerators."),
        ("Introduction", "We study tiling strategies for attention on GPU hardware."),
        ("Method", "The kernel fuses softmax into the matmul and never materializes scores."),
        ("Results", "Throughput improves by a factor of two on long sequences."),
    ],
    "2401.00002": [
        ("Abstract", "Dense retrieval with maximal marginal relevance improves coverage."),
        ("Introduction", "Retrieval augmented generation grounds a model in a corpus."),
    ],
}


def embed(paths: Paths, embedder, **kwargs) -> Manifest:
    return embed_papers(paths, embedder=embedder, log=lambda _: None, **kwargs)


@pytest.fixture
def populated(paths: Paths, make_corpus, stub_embedder):
    make_corpus(CORPUS)
    embed(paths, stub_embedder)
    return ChunkStore(paths.chroma_dir)


# --------------------------------------------------------------------------------------
# the write path
# --------------------------------------------------------------------------------------


def test_collection_holds_every_chunk_keyed_by_chunk_id(populated):
    assert populated.count() == 6
    ids, _ = populated.paper_rows("2401.00001")
    assert sorted(ids) == [f"2401.00001:{index}" for index in range(4)]


def test_documents_are_the_chunk_text(populated):
    rows = populated.collection.get(ids=["2401.00001:0"], include=["documents"])
    assert rows["documents"][0] == CORPUS["2401.00001"][0][1]


def test_metadata_contract_on_stored_rows(populated):
    _, metadatas = populated.paper_rows("2401.00001")
    for metadata in metadatas:
        assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
        assert metadata["paper_id"] == "2401.00001"
        assert metadata["arxiv_categories"] == "cs.CL,cs.LG"
        assert metadata["title"] == "A Paper" and metadata["year"] == 2024
        assert "page_start" in metadata and "page_end" in metadata
        assert "page_span" not in metadata
        assert metadata["chunker_version"] == CHUNKER_VERSION
        assert metadata["embedder_version"] == 1
        assert "topic_id" in metadata and "topic_label" in metadata


def test_re_chunking_to_fewer_chunks_leaves_no_orphans(paths, make_corpus, stub_embedder):
    """The reason writes are delete-then-add: an orphan retrieves, and cites a page span
    that no longer exists."""
    make_corpus(CORPUS)
    embed(paths, stub_embedder)
    assert ChunkStore(paths.chroma_dir).count() == 6

    payload = json.loads(paths.chunk_path("2401.00001").read_text(encoding="utf-8"))
    payload["chunks"] = payload["chunks"][:2]
    paths.chunk_path("2401.00001").write_text(json.dumps(payload), encoding="utf-8")
    embed(paths, stub_embedder)

    store = ChunkStore(paths.chroma_dir)
    assert store.count() == 4
    ids, _ = store.paper_rows("2401.00001")
    assert sorted(ids) == ["2401.00001:0", "2401.00001:1"]


def test_a_failed_paper_is_removed_from_the_collection(paths, make_corpus, stub_embedder):
    from rag.ingest import PaperError

    make_corpus(CORPUS)
    embed(paths, stub_embedder)
    paths.chunk_path("2401.00002").write_text("{not json", encoding="utf-8")

    with pytest.raises(PaperError):
        embed(paths, stub_embedder)

    store = ChunkStore(paths.chroma_dir)
    assert store.count() == 4
    assert "2401.00002" not in store.paper_summary()


def test_replace_paper_rejects_a_non_scalar_metadata_value(paths, stub_embedder):
    store = ChunkStore(paths.chroma_dir)
    with pytest.raises(IngestError, match="metadata"):
        store.replace_paper(
            "p",
            ids=["p:0"],
            documents=["text"],
            vectors=stub_embedder.encode(["text"]),
            metadatas=[{"paper_id": "p", "page_span": [1, 2]}],
        )


def test_replace_paper_rejects_misaligned_inputs(paths, stub_embedder):
    store = ChunkStore(paths.chroma_dir)
    with pytest.raises(IngestError, match="disagree"):
        store.replace_paper(
            "p",
            ids=["p:0", "p:1"],
            documents=["text"],
            vectors=stub_embedder.encode(["text"]),
            metadatas=[{"paper_id": "p"}],
        )


# --------------------------------------------------------------------------------------
# topics: recomputed corpus-wide, rewritten in place without re-embedding
# --------------------------------------------------------------------------------------


def test_topic_metadata_is_rewritten_without_re_embedding(paths, make_corpus, stub_embedder):
    make_corpus(CORPUS)
    embed(paths, stub_embedder)

    # Force a topic move on an untouched paper: pass 2 must fix its rows with a metadata
    # update, not by running the encoder again.
    store = ChunkStore(paths.chroma_dir)
    store.set_topic("2401.00001", topic_id=99, topic_label="stale")
    assert store.paper_summary()["2401.00001"] == (99, "stale")

    stub_embedder.calls = 0
    embed(paths, stub_embedder)
    assert stub_embedder.calls == 0

    store = ChunkStore(paths.chroma_dir)
    topic_id, topic_label = store.paper_summary()["2401.00001"]
    assert topic_id == 0 and topic_label != "stale"
    _, metadatas = store.paper_rows("2401.00001")
    # The merge must keep the rest of the row, not replace it with the two topic fields.
    assert all(metadata["title"] == "A Paper" for metadata in metadatas)


def test_set_topic_on_an_absent_paper_is_a_no_op(populated):
    assert populated.set_topic("nope", topic_id=1, topic_label="x") == 0


def test_store_is_rebuilt_when_the_collection_is_dropped(paths, make_corpus, stub_embedder):
    make_corpus(CORPUS)
    embed(paths, stub_embedder)
    ChunkStore(paths.chroma_dir).reset()
    assert ChunkStore(paths.chroma_dir).count() == 0

    stub_embedder.calls = 0
    embed(paths, stub_embedder)  # vectors are cached; only the collection is rewritten
    assert stub_embedder.calls == 0
    assert ChunkStore(paths.chroma_dir).count() == 6


# --------------------------------------------------------------------------------------
# the query path
# --------------------------------------------------------------------------------------


def test_querying_a_chunk_verbatim_returns_that_chunk_first(paths, populated, stub_embedder):
    for index, (_, text) in enumerate(CORPUS["2401.00001"]):
        results = query_store(paths, text, embedder=stub_embedder, k=3)
        assert results[0]["chunk_id"] == f"2401.00001:{index}"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
        assert results[0]["text"] == text


def test_a_paper_id_filter_returns_only_that_paper(paths, populated, stub_embedder):
    results = query_store(
        paths,
        CORPUS["2401.00001"][0][1],
        embedder=stub_embedder,
        k=6,
        where={"paper_id": "2401.00002"},
    )
    assert results and all(hit["metadata"]["paper_id"] == "2401.00002" for hit in results)


def test_query_carries_citation_metadata(paths, populated, stub_embedder):
    hit = query_store(paths, CORPUS["2401.00002"][0][1], embedder=stub_embedder, k=1)[0]
    metadata = hit["metadata"]
    assert metadata["paper_id"] == "2401.00002"
    assert metadata["section"] == "Abstract"
    assert metadata["page_start"] == 1 and metadata["page_end"] == 1
    assert metadata["title"] == "A Paper"


def test_chroma_never_embeds_anything_itself(populated):
    """The regression this guards is measured, not theoretical: with `embedding_function=
    None` chromadb 1.5 downloads its own ONNX MiniLM and answers `query_texts` with it —
    a different encoder on the query side than the corpus side, and no error anywhere."""
    with pytest.raises(IngestError, match="explicit vectors"):
        populated.collection.query(query_texts=["attention kernels"], n_results=1)
    with pytest.raises(IngestError, match="explicit vectors"):
        populated.collection.add(ids=["x:0"], documents=["text"])
    assert populated.count() == 6  # and nothing leaked into the collection


def test_query_on_an_empty_store_returns_nothing(paths, stub_embedder):
    ChunkStore(paths.chroma_dir)
    assert query_store(paths, "anything", embedder=stub_embedder, k=5) == []


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def test_report_shows_embed_topic_and_store_sections(paths, make_corpus, stub_embedder):
    from rag.ingest import report

    make_corpus(CORPUS)
    embed(paths, stub_embedder)
    text = report(paths)

    assert "embedded   2" in text
    assert "embeddings: 6 vector(s) across 2 paper(s)" in text
    assert "tokens: max" in text
    assert "topics: k=1" in text
    assert "store: 6 chunk(s) in collection atlas_chunks" in text
    assert "chunks: 6 across 2 paper(s)" in text  # chunk stats survive the status change


def test_metadata_builder_and_stored_rows_agree(populated, paths):
    payload = json.loads(paths.chunk_path("2401.00002").read_text(encoding="utf-8"))
    document = json.loads(paths.document_path("2401.00002").read_text(encoding="utf-8"))
    expected = chunk_metadata(
        payload["chunks"][0], document, chunker_version=CHUNKER_VERSION, embedder_version=1, topic_id=0,
        topic_label=populated.paper_summary()["2401.00002"][1],
    )
    stored = populated.collection.get(ids=["2401.00002:0"], include=["metadatas"])["metadatas"][0]
    assert stored == expected
