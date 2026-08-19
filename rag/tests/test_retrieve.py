"""Step 4 — MMR retrieval and the eval harness (docs/18-rag-retrieval.md § Validation).

Same quarantine as Step 3: everything except the two store-backed tests runs with no
chromadb, no torch and no corpus. MMR, the per-paper cap, the filter translation and the
metrics are pure functions, so they are tested as pure functions; the store is replaced by a
double that records the arguments it was called with, which is how "filters are applied in
the fetch, not after selection" becomes an assertion rather than a comment.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag.eval import EvalQuery, evaluate, format_report, load_queries
from rag.ingest import IngestError, Paths
from rag.retrieve import (
    DEFAULT_K,
    Result,
    build_where,
    mmr,
    resolve_topic_label,
    retrieve,
    topic_labels,
)


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


# --------------------------------------------------------------------------------------
# MMR
# --------------------------------------------------------------------------------------


def test_lambda_one_is_plain_top_k():
    """The assertion that proves the MMR path does not quietly reorder anything when
    diversity is switched off."""
    vectors = np.stack([unit(1, 0), unit(0.99, 0.14), unit(0.98, 0.2), unit(0, 1)])
    relevance = [0.9, 0.8, 0.7, 0.1]
    assert mmr(relevance, vectors, k=3, lambda_mult=1.0) == [0, 1, 2]


def test_lambda_zero_prefers_the_outlier():
    """Three near-duplicates and one orthogonal chunk: pure diversity takes the outlier
    second, even though it is the least relevant candidate."""
    vectors = np.stack([unit(1, 0), unit(0.999, 0.045), unit(0.997, 0.077), unit(0, 1)])
    relevance = [0.9, 0.89, 0.88, 0.10]
    assert mmr(relevance, vectors, k=2, lambda_mult=0.0) == [0, 3]


def test_selection_is_deterministic():
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((20, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    relevance = list(rng.random(20))
    first = mmr(relevance, vectors, k=5, lambda_mult=0.6)
    assert all(mmr(relevance, vectors, k=5, lambda_mult=0.6) == first for _ in range(3))


def test_ties_break_by_candidate_rank():
    """Identical vectors and identical relevance: order must come from the store's ranking,
    not from float noise inside an argmax."""
    vectors = np.stack([unit(1, 0)] * 4)
    assert mmr([0.5] * 4, vectors, k=3, lambda_mult=0.6) == [0, 1, 2]


def test_k_larger_than_candidates_returns_everything():
    vectors = np.stack([unit(1, 0), unit(0, 1)])
    assert mmr([0.9, 0.8], vectors, k=10) == [0, 1]
    assert mmr([], np.zeros((0, 4), dtype=np.float32), k=5) == []


def test_lambda_out_of_range_raises():
    with pytest.raises(IngestError, match="lambda"):
        mmr([0.5], np.stack([unit(1, 0)]), k=1, lambda_mult=1.5)


def test_per_paper_cap_still_fills_k():
    """MMR diversifies in vector space, which is not diversity across papers. Ten distinct
    chunks from one paper must not take every slot."""
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((12, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    relevance = list(np.linspace(0.9, 0.5, 12))
    groups = ["A"] * 10 + ["B", "B"]

    order = mmr(relevance, vectors, k=4, lambda_mult=0.6, groups=groups, max_per_group=2)
    picked = [groups[index] for index in order]
    assert len(order) == 4
    assert picked.count("A") == 2 and picked.count("B") == 2


def test_cap_relaxes_rather_than_under_filling_k():
    """When every candidate is from one paper the cap cannot be honoured. Returning 1 of 3
    would look like a retrieval failure, so the cap is dropped for the remaining slots."""
    vectors = np.stack([unit(1, 0)] * 3)
    order = mmr([0.9, 0.8, 0.7], vectors, k=3, groups=["A"] * 3, max_per_group=1)
    assert order == [0, 1, 2]


# --------------------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------------------


def test_where_clauses():
    assert build_where() is None
    assert build_where(paper_id="2205.14135") == {"paper_id": {"$in": ["2205.14135"]}}
    assert build_where(paper_id=["a", "b"]) == {"paper_id": {"$in": ["a", "b"]}}
    assert build_where(year_min=2023) == {"year": {"$gte": 2023}}
    assert build_where(section="Method") == {"section": "Method"}


def test_multiple_filters_nest_under_and():
    where = build_where(topic_label="retrieval", year_min=2022, year_max=2024)
    assert where == {
        "$and": [
            {"topic_label": "retrieval"},
            {"year": {"$gte": 2022}},
            {"year": {"$lte": 2024}},
        ]
    }


def write_topics(paths: Paths, labels: list[str]) -> None:
    paths.topics_path.write_text(
        json.dumps({"topics": [{"topic_id": i, "label": l} for i, l in enumerate(labels)]}),
        encoding="utf-8",
    )


def test_topic_label_resolution(paths: Paths):
    write_topics(paths, ["retrieval, passages, question", "quantization, gpu, bit"])
    assert topic_labels(paths) == ["retrieval, passages, question", "quantization, gpu, bit"]
    assert resolve_topic_label(paths, "QUANTIZATION, GPU, BIT") == "quantization, gpu, bit"
    assert resolve_topic_label(paths, "retrieval") == "retrieval, passages, question"


def test_unknown_topic_names_the_known_ones(paths: Paths):
    """An unknown label must not degrade to an empty result set — that looks exactly like a
    retrieval quality problem and is not one."""
    write_topics(paths, ["retrieval, passages, question"])
    with pytest.raises(IngestError) as exc:
        resolve_topic_label(paths, "cooking")
    assert "retrieval, passages, question" in str(exc.value)


# --------------------------------------------------------------------------------------
# retrieve() against a store double
# --------------------------------------------------------------------------------------


class FakeStore:
    """Records what it was asked for, and answers from a fixed candidate list."""

    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def query(self, vector, *, k=5, where=None, include_vectors=False):
        self.calls.append({"k": k, "where": where, "include_vectors": include_vectors})
        return [dict(hit) for hit in self.hits[:k]]


def make_hit(index: int, paper_id: str, score: float, vector, **metadata) -> dict:
    row = {
        "paper_id": paper_id,
        "chunk_index": index,
        "section": "Method",
        "page_start": index + 1,
        "page_end": index + 2,
        "title": f"Paper {paper_id}",
        "year": 2023,
        "topic_label": "retrieval, passages, question",
    }
    row.update(metadata)
    return {
        "chunk_id": f"{paper_id}:{index}",
        "text": f"chunk {index} of {paper_id}",
        "score": score,
        "metadata": row,
        "vector": np.asarray(vector, dtype=np.float32),
    }


class OneVector:
    """An embedder that returns a fixed query vector; retrieve() must not care."""

    def encode(self, texts):
        return np.stack([unit(1, 0)] * len(texts))


@pytest.fixture
def hits() -> list[dict]:
    return [
        make_hit(0, "A", 0.90, unit(1, 0)),
        make_hit(1, "A", 0.88, unit(0.999, 0.045)),
        make_hit(2, "A", 0.86, unit(0.997, 0.077)),
        make_hit(0, "B", 0.60, unit(0.3, 0.95)),
        make_hit(0, "C", 0.40, unit(0, 1), year=None, title="", section=""),
    ]


def test_retrieve_shape_and_ranks(paths: Paths, hits):
    store = FakeStore(hits)
    results = retrieve(paths, "a question", k=3, embedder=OneVector(), store=store)

    assert [r.rank for r in results] == [0, 1, 2]
    assert all(isinstance(r, Result) for r in results)
    first = results[0]
    assert first.chunk_id == "A:0" and first.paper_id == "A"
    assert first.score == pytest.approx(0.90)  # the store's cosine, not the MMR objective
    assert (first.page_start, first.page_end) == (1, 2) and isinstance(first.page_start, int)


def test_missing_year_is_none_not_keyerror(paths: Paths, hits):
    """A local PDF has no year, so `rag.embed.chunk_metadata` omits the key entirely."""
    store = FakeStore([hits[4]])
    (result,) = retrieve(paths, "q", k=1, embedder=OneVector(), store=store)
    assert result.year is None and result.title == "" and result.section == ""


def test_per_paper_cap_applies_through_retrieve(paths: Paths, hits):
    """Pure relevance would take three chunks of paper A; the cap trades the third for the
    best chunk of another paper, and still returns k."""
    store = FakeStore(hits)
    uncapped = retrieve(paths, "q", k=3, max_per_paper=0, lambda_mult=1.0,
                        embedder=OneVector(), store=store)
    assert [r.paper_id for r in uncapped] == ["A", "A", "A"]

    capped = retrieve(paths, "q", k=3, max_per_paper=2, lambda_mult=1.0,
                      embedder=OneVector(), store=store)
    assert [r.paper_id for r in capped] == ["A", "A", "B"]


def test_default_lambda_diversifies(paths: Paths, hits):
    """At the default lambda the near-duplicate second chunk of A loses to a chunk that is
    less relevant but points somewhere else — the whole reason MMR is here."""
    results = retrieve(paths, "q", k=2, lambda_mult=0.6, embedder=OneVector(), store=FakeStore(hits))
    assert [r.chunk_id for r in results] == ["A:0", "C:0"]

    # ...and the shipped default keeps the same behaviour on this candidate set, which is
    # what pins DEFAULT_LAMBDA to a value that actually diversifies.
    shipped = retrieve(paths, "q", k=2, embedder=OneVector(), store=FakeStore(hits))
    assert [r.paper_id for r in shipped] != ["A", "A"]


def test_filters_go_into_the_fetch(paths: Paths, hits):
    """Post-filtering would silently under-fill k. Assert the clause reached the store."""
    store = FakeStore(hits)
    retrieve(paths, "q", k=2, paper_id="A", year_min=2022, embedder=OneVector(), store=store)
    call = store.calls[-1]
    assert call["where"] == {"$and": [{"paper_id": {"$in": ["A"]}}, {"year": {"$gte": 2022}}]}
    assert call["include_vectors"] is True
    assert call["k"] >= 2  # fetch_k, not k


def test_fetch_k_never_below_k(paths: Paths, hits):
    store = FakeStore(hits)
    retrieve(paths, "q", k=5, fetch_k=2, embedder=OneVector(), store=store)
    assert store.calls[-1]["k"] == 5


def test_empty_query_raises(paths: Paths, hits):
    with pytest.raises(IngestError, match="empty query"):
        retrieve(paths, "   ", embedder=OneVector(), store=FakeStore(hits))


def test_no_candidates_returns_empty(paths: Paths):
    assert retrieve(paths, "q", embedder=OneVector(), store=FakeStore([])) == []


# --------------------------------------------------------------------------------------
# eval harness
# --------------------------------------------------------------------------------------


def fixture_results(paper_ids: list[str]) -> list[Result]:
    return [
        Result(
            chunk_id=f"{paper_id}:{index}",
            text="",
            score=0.9 - 0.1 * index,
            rank=index,
            paper_id=paper_id,
            title="",
            year=2023,
            section="",
            page_start=1,
            page_end=1,
            topic_label="",
        )
        for index, paper_id in enumerate(paper_ids)
    ]


EVAL_CASES = [
    EvalQuery("q1", "first", ("A",)),
    EvalQuery("q2", "second", ("B",)),
    EvalQuery("q3", "third", ("C",)),
]


def test_perfect_retriever_scores_one():
    answers = {"first": ["A", "X"], "second": ["B", "Y"], "third": ["C", "Z"]}
    report = evaluate(lambda q, k: fixture_results(answers[q]), EVAL_CASES, k=5, ks=(1, 3, 5))
    assert report["recall"][1] == pytest.approx(1.0)
    assert report["mrr"] == pytest.approx(1.0)
    assert report["misses"] == []
    assert report["distinct_papers"] == pytest.approx(2.0)


def test_wrong_answer_key_scores_zero():
    """The test that separates a metric from a function that always passes."""
    report = evaluate(lambda q, k: fixture_results(["Z", "Y"]), EVAL_CASES, k=5, ks=(1, 3, 5))
    assert report["recall"][5] == 0.0 and report["mrr"] == 0.0
    assert report["misses"] == ["q1", "q2", "q3"]


def test_recall_at_k_respects_rank():
    """The right paper at chunk 3 is a hit at k=3 and k=5, never at k=1."""
    report = evaluate(
        lambda q, k: fixture_results(["X", "Y", "A", "B", "C"]),
        [EvalQuery("q1", "first", ("A",))],
        k=5,
        ks=(1, 3, 5),
    )
    assert report["recall"][1] == 0.0
    assert report["recall"][3] == 1.0
    assert report["mrr"] == pytest.approx(1 / 3)


def test_probes_are_scored_separately():
    cases = [EvalQuery("q1", "first", ("A",)), EvalQuery("p1", "probe", ())]
    report = evaluate(lambda q, k: fixture_results(["A"]), cases, k=5)
    assert report["n_queries"] == 1 and report["n_probes"] == 1
    assert report["misses"] == []  # the probe is not a miss
    assert report["probes"][0]["query_id"] == "p1"
    assert "probe" in format_report(report).lower()


def test_shipped_queries_are_well_formed():
    """A typo'd arXiv ID would otherwise show up forever as a mysterious miss."""
    corpus = set(
        (Paths.default().data_dir.parent / "scripts" / "arxiv_ids.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    queries = load_queries()
    assert len(queries) >= 20
    assert len({query.query_id for query in queries}) == len(queries)
    for query in queries:
        assert query.query.strip(), query.query_id
        for paper_id in query.expected_papers:
            assert paper_id in corpus, f"{query.query_id}: {paper_id} not in the corpus"


# --------------------------------------------------------------------------------------
# store-backed (needs chromadb; still no corpus and no torch)
# --------------------------------------------------------------------------------------


def test_end_to_end_against_a_real_collection(paths: Paths, stub_embedder):
    pytest.importorskip("chromadb", reason="Step 3 dependency")
    from rag.embed import embed_papers
    from rag.ingest import Manifest
    from rag.store import ChunkStore
    from rag.tests.conftest import write_paper

    manifest = Manifest.load(paths.manifest_path)
    write_paper(paths, manifest, "1111.00001", [("Method", "tiling attention in on chip memory")])
    write_paper(paths, manifest, "2222.00002", [("Method", "four bit weight quantisation")])
    embed_papers(paths, embedder=stub_embedder)

    store = ChunkStore(paths.chroma_dir)
    assert store.count() == 2

    results = retrieve(
        paths,
        "tiling attention in on chip memory",
        k=2,
        embedder=stub_embedder,
        store=store,
    )
    # The stub embeds identical text identically, so the exact chunk must come back first.
    assert results[0].chunk_id == "1111.00001:0"
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert results[0].rank == 0


def test_store_returns_vectors_for_mmr(paths: Paths, stub_embedder):
    pytest.importorskip("chromadb", reason="Step 3 dependency")
    from rag.embed import embed_papers
    from rag.ingest import Manifest
    from rag.store import ChunkStore
    from rag.tests.conftest import write_paper

    manifest = Manifest.load(paths.manifest_path)
    write_paper(paths, manifest, "1111.00001", [("Method", "alpha text"), ("Method", "beta text")])
    embed_papers(paths, embedder=stub_embedder)

    store = ChunkStore(paths.chroma_dir)
    hits = store.query(stub_embedder.vector("alpha text"), k=2, include_vectors=True)
    assert hits[0]["vector"] is not None
    assert hits[0]["vector"].shape == (stub_embedder.dim,)
    assert float(np.linalg.norm(hits[0]["vector"])) == pytest.approx(1.0, abs=1e-5)
    # No vectors unless asked: the smoke path stays a cheap call.
    assert "vector" not in store.query(stub_embedder.vector("alpha text"), k=1)[0]
