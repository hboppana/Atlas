"""Step 3 validation — docs/17-rag-embeddings-store.md § Validation.

Everything that is not literally "call the model" runs against the injected `StubEmbedder`
from conftest, so the cache format, the skip rules, k-means, silhouette, the labelling and
the manifest transitions are all covered on a checkout with no sentence-transformers
installed. The two tests that need the real encoder are importorskip-guarded at the bottom.
No network, no corpus.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

np = pytest.importorskip("numpy", reason="Step 3 dependency; pip install -r requirements.txt")

from rag.embed import (  # noqa: E402
    EMBED_DIM,
    EMBEDDER_VERSION,
    MAX_SEQ_TOKENS,
    chunk_metadata,
    choose_k,
    derive_topics,
    embed_papers,
    embed_texts,
    kmeans,
    label_clusters,
    load_cache,
    paper_label_text,
    silhouette,
    truncation_stats,
)
from rag.chunk import CHUNKER_VERSION  # noqa: E402
from rag.ingest import EMBEDDED, FAILED, Manifest, PaperError, Paths  # noqa: E402

TWO_PAPERS = {
    "2401.00001": [
        ("Abstract", "Sparse attention kernels reduce memory traffic on modern accelerators."),
        ("Introduction", "We study tiling strategies for attention on GPU hardware."),
        ("Method", "The kernel fuses softmax into the matmul and never materializes scores."),
    ],
    "2401.00002": [
        ("Abstract", "Dense retrieval with maximal marginal relevance improves answer coverage."),
        ("Introduction", "Retrieval augmented generation grounds a model in a document corpus."),
    ],
}


def embed(paths: Paths, embedder, **kwargs) -> Manifest:
    return embed_papers(paths, embedder=embedder, log=lambda _: None, **kwargs)


# --------------------------------------------------------------------------------------
# cache round-trip
# --------------------------------------------------------------------------------------


def test_cache_round_trip(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)

    cached = load_cache(paths.embedding_path("2401.00001"))
    assert cached.ids == ["2401.00001:0", "2401.00001:1", "2401.00001:2"]
    assert cached.vectors.shape == (3, EMBED_DIM)
    assert cached.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(cached.vectors, axis=1), 1.0, atol=1e-6)
    assert cached.meta["embedder_version"] == EMBEDDER_VERSION
    assert cached.meta["chunker_version"] == CHUNKER_VERSION
    assert cached.meta["dim"] == EMBED_DIM
    assert cached.meta["source_sha256"]
    assert cached.meta["model_name"] and cached.meta["model_revision"]


def test_rows_are_in_chunk_index_order_despite_length_sorted_batching(stub_embedder):
    # Batching sorts by length to cut padding; the written rows must not inherit that order.
    texts = ["z" * 90, "a", "m" * 40, "q" * 5]
    vectors, tokens = embed_texts(texts, stub_embedder)
    for index, text in enumerate(texts):
        assert np.allclose(vectors[index], stub_embedder.vector(text), atol=1e-6)
    assert tokens.tolist() == [1, 1, 1, 1]


def test_manifest_transition(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    manifest = embed(paths, stub_embedder)

    record = manifest.get("2401.00001")
    assert record["status"] == EMBEDDED
    assert record["embedding_count"] == 3
    assert record["embedder_version"] == EMBEDDER_VERSION
    assert record["embedded_at"].endswith("Z")
    assert record["topic_id"] == 0  # one topic: 2 papers is below the k-selection floor
    assert record["error"] is None


# --------------------------------------------------------------------------------------
# skip / invalidation
# --------------------------------------------------------------------------------------


def test_second_run_is_a_no_op(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)
    stub_embedder.calls = 0

    embed(paths, stub_embedder)
    assert stub_embedder.calls == 0


def test_touching_one_chunk_file_re_embeds_only_that_paper(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)
    stub_embedder.calls = 0
    stub_embedder.encoded.clear()

    payload = json.loads(paths.chunk_path("2401.00002").read_text(encoding="utf-8"))
    payload["chunks"][0]["text"] = "Rewritten abstract about reranking and recall."
    paths.chunk_path("2401.00002").write_text(json.dumps(payload), encoding="utf-8")

    embed(paths, stub_embedder)
    assert stub_embedder.calls == 1
    assert all(text in [c[1] for c in TWO_PAPERS["2401.00002"]] or "Rewritten" in text
               for text in stub_embedder.encoded)
    assert len(stub_embedder.encoded) == 2  # only the two chunks of the touched paper


def test_embedder_version_bump_re_embeds_everything(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)
    stub_embedder.encoded.clear()

    embed(paths, stub_embedder, embedder_version=EMBEDDER_VERSION + 1)
    assert len(stub_embedder.encoded) == 5
    assert load_cache(paths.embedding_path("2401.00001")).meta["embedder_version"] == (
        EMBEDDER_VERSION + 1
    )


def test_force_re_embeds_everything(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)
    stub_embedder.encoded.clear()

    embed(paths, stub_embedder, force=True)
    assert len(stub_embedder.encoded) == 5


def test_chunking_still_skips_an_embedded_paper(paths: Paths, make_corpus, stub_embedder):
    """`embedded` is downstream of `chunked`; --chunk must not undo it."""
    from rag.chunk import chunk_papers

    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)

    manifest = chunk_papers(paths, log=lambda _: None)
    assert manifest.get("2401.00001")["status"] == EMBEDDED


# --------------------------------------------------------------------------------------
# failure isolation
# --------------------------------------------------------------------------------------


def test_one_corrupt_chunk_file_fails_only_that_paper(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    paths.chunk_path("2401.00002").write_text("{not json", encoding="utf-8")

    with pytest.raises(PaperError):  # non-zero exit for the CLI
        embed(paths, stub_embedder)

    manifest = Manifest.load(paths.manifest_path)
    assert manifest.get("2401.00002")["status"] == FAILED
    assert "embed:" in manifest.get("2401.00002")["error"]
    assert not paths.embedding_path("2401.00002").exists()
    assert manifest.get("2401.00001")["status"] == EMBEDDED
    assert paths.embedding_path("2401.00001").exists()


def test_empty_chunk_list_is_a_recorded_failure(paths: Paths, make_corpus, stub_embedder):
    make_corpus({"2401.00003": []})
    with pytest.raises(PaperError):
        embed(paths, stub_embedder)
    assert Manifest.load(paths.manifest_path).get("2401.00003")["status"] == FAILED


# --------------------------------------------------------------------------------------
# the truncation check Step 3 owes Step 2
# --------------------------------------------------------------------------------------


def test_truncation_stats():
    stats = truncation_stats([10, 20, 231, 100])
    assert stats == {"n": 4, "max": 231, "p99": 227, "over": 0, "limit": MAX_SEQ_TOKENS}
    assert truncation_stats([10, 300, 257])["over"] == 2
    assert truncation_stats([])["n"] == 0


def test_truncation_stats_are_cached_with_the_vectors(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)
    counts = load_cache(paths.embedding_path("2401.00001")).token_counts
    assert counts.tolist() == [
        len(text.split()) for _, text in TWO_PAPERS["2401.00001"]
    ]


# --------------------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------------------


def _three_groups(per_group: int = 6, dim: int = 8, spread: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(7)
    centres = np.eye(3, dim)
    rows = [
        centres[group] + spread * rng.standard_normal(dim)
        for group in range(3)
        for _ in range(per_group)
    ]
    matrix = np.stack(rows)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_kmeans_recovers_well_separated_groups():
    vectors = _three_groups()
    labels, centroids = kmeans(vectors, 3, seed=0)
    assert centroids.shape == (3, vectors.shape[1])
    for group in range(3):
        block = labels[group * 6 : (group + 1) * 6]
        assert len(set(block.tolist())) == 1  # every member of a group agrees
    assert len(set(labels.tolist())) == 3  # ...and the three groups disagree


def test_kmeans_is_deterministic_across_runs():
    vectors = _three_groups()
    first, _ = kmeans(vectors, 3, seed=0)
    second, _ = kmeans(vectors, 3, seed=0)
    assert first.tolist() == second.tolist()


def test_silhouette_prefers_the_true_k():
    vectors = _three_groups()
    scores = {k: silhouette(vectors, kmeans(vectors, k, seed=0)[0]) for k in (2, 3, 4)}
    assert scores[3] == max(scores.values())
    assert silhouette(vectors, np.zeros(len(vectors), dtype=int)) == 0.0  # one cluster


def test_choose_k_finds_three_and_survives_a_tiny_corpus():
    k, labels, score = choose_k(_three_groups(), seed=0)
    assert k == 3 and score > 0.5 and len(set(labels.tolist())) == 3

    # 3 papers is below n // MIN_PAPERS_PER_TOPIC, so k-selection has no valid k to try.
    k, labels, score = choose_k(_three_groups(per_group=1)[:3], seed=0)
    assert k == 1 and labels.tolist() == [0, 0, 0] and score == 0.0


def test_kmeans_never_returns_an_empty_cluster():
    # Duplicated points make it easy for a centroid to lose every member mid-iteration.
    vectors = np.repeat(_three_groups(per_group=2), 3, axis=0)
    labels, _ = kmeans(vectors, 5, seed=0)
    assert len(set(labels.tolist())) == 5


# --------------------------------------------------------------------------------------
# labelling
# --------------------------------------------------------------------------------------


def test_label_uses_distinguishing_terms_and_drops_corpus_wide_ones():
    from rag.embed import _terms  # the real term extractor, so stopwords apply

    shared = "neural network neural network"  # every paper says this
    terms = [
        Counter(_terms("attention kernel tiling " + shared)),
        Counter(_terms("attention kernel tiling " + shared)),
        Counter(_terms("retrieval reranking recall " + shared)),
        Counter(_terms("retrieval reranking recall " + shared)),
    ]
    clusters = label_clusters(terms, [0, 0, 1, 1])

    assert "attention" in clusters[0]["label"]
    assert "retrieval" in clusters[1]["label"]
    for cluster in clusters:
        assert "neural" not in cluster["label"]  # shared by the whole corpus -> divided out
        assert len(cluster["terms"]) <= 3


def test_label_ignores_a_term_only_one_member_paper_uses():
    terms = [
        Counter(["quantization"] * 20 + ["kernel"] * 3),
        Counter(["kernel"] * 3),
        Counter(["retrieval"] * 5),
        Counter(["retrieval"] * 5),
    ]
    clusters = label_clusters(terms, [0, 0, 1, 1])
    assert "quantization" not in clusters[0]["label"]
    assert "kernel" in clusters[0]["label"]


def test_paper_label_text_falls_back_when_no_abstract_is_detected():
    chunks = [
        {"section": "Introduction", "text": "opening passage"},
        {"section": "Method", "text": "method passage"},
        {"section": "Results", "text": "results passage"},
    ]
    text = paper_label_text({"title": "A Title"}, chunks)
    assert "A Title" in text and "opening passage" in text and "results passage" not in text

    text = paper_label_text({"title": "A Title"}, [{"section": "Abstract", "text": "the abstract"}])
    assert "the abstract" in text


def test_derive_topics_shape():
    vectors = _three_groups()
    paper_ids = [f"p{index}" for index in range(len(vectors))]
    terms = [Counter(["alpha"] if index < 6 else ["beta"] if index < 12 else ["gamma"])
             for index in range(len(vectors))]
    topics = derive_topics(paper_ids, vectors, terms, seed=0)

    assert topics["k"] == 3
    assert set(topics["assignments"]) == set(paper_ids)
    assert sum(len(topic["paper_ids"]) for topic in topics["topics"]) == len(paper_ids)
    assert topics["derived_at"].endswith("Z")
    labels = {topic["label"] for topic in topics["topics"]}
    assert labels == {"alpha", "beta", "gamma"}


def test_topics_file_is_written(paths: Paths, make_corpus, stub_embedder):
    make_corpus(TWO_PAPERS)
    embed(paths, stub_embedder)
    topics = json.loads(paths.topics_path.read_text(encoding="utf-8"))
    assert topics["n_papers"] == 2 and topics["k"] == 1
    assert set(topics["assignments"]) == set(TWO_PAPERS)


# --------------------------------------------------------------------------------------
# metadata contract
# --------------------------------------------------------------------------------------


def test_chunk_metadata_is_scalars_only():
    chunk = {
        "chunk_id": "2401.00001:7",
        "chunk_index": 7,
        "text": "body",
        "section": "Method",
        "section_index": 3,
        "page_span": [4, 5],
        "n_tokens_est": 188,
    }
    document = {"title": "Attention Kernels", "year": 2024, "arxiv_categories": ["cs.CL", "cs.LG"]}
    metadata = chunk_metadata(
        chunk, document, chunker_version=1, embedder_version=1, topic_id=2, topic_label="kernels"
    )

    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
    assert metadata["paper_id"] == "2401.00001"
    assert metadata["page_start"] == 4 and metadata["page_end"] == 5
    assert "page_span" not in metadata  # Chroma rejects lists
    assert metadata["arxiv_categories"] == "cs.CL,cs.LG"
    assert metadata["title"] == "Attention Kernels" and metadata["year"] == 2024
    assert metadata["topic_id"] == 2 and metadata["topic_label"] == "kernels"


def test_chunk_metadata_omits_unknown_year_rather_than_writing_none():
    metadata = chunk_metadata(
        {"chunk_id": "abc:0", "page_span": [1, 1]},
        {"title": None, "year": None, "arxiv_categories": []},
        chunker_version=1,
        embedder_version=1,
    )
    assert "year" not in metadata
    assert metadata["title"] == "" and metadata["arxiv_categories"] == ""
    assert all(value is not None for value in metadata.values())


# --------------------------------------------------------------------------------------
# real-model tests — skipped without the heavy deps
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minilm():
    pytest.importorskip("sentence_transformers", reason="heavy Step 3 dependency")
    pytest.importorskip("torch", reason="heavy Step 3 dependency")
    from rag.embed import MiniLMEmbedder

    try:
        return MiniLMEmbedder(device="cpu")
    except Exception as exc:  # no cached weights and no network -> skip, never fail
        pytest.skip(f"all-MiniLM-L6-v2 unavailable: {exc}")


def test_real_model_output_is_384d_and_unit_norm(minilm):
    vectors = minilm.encode(["the cat sat on the mat", "gradient checkpointing"])
    assert vectors.shape == (2, EMBED_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_real_model_orders_paraphrase_above_unrelated(minilm):
    query, near, far = (
        "the cat sat on the mat",
        "a cat is on a rug",
        "gradient checkpointing reduces activation memory",
    )
    vectors = minilm.encode([query, near, far])
    # Not a bit-exactness assertion: GPU and CPU reductions differ at ~1e-6.
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])
    assert float(minilm.encode([query])[0] @ vectors[0]) > 0.9999


def test_real_tokenizer_counts_wordpieces(minilm):
    counts = minilm.count_tokens(["hello world", "hello world " * 200])
    assert counts[0] < counts[1]
    assert counts[1] > MAX_SEQ_TOKENS  # the check that would catch a Step 2 budget miss
