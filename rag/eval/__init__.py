"""Retrieval eval — docs/18-rag-retrieval.md § Eval set.

The point of Step 4 is not MMR; it is that retrieval leaves Phase 3 with a **number attached
to it**, measured over queries written by hand and frozen once measured. Phase 4 is built on
whatever this prints.

    queries.json  ->  retrieve(k)  ->  recall@1/3/5, MRR, distinct papers, the miss list

`evaluate` takes a `retriever` callable rather than reaching for the store itself, so the
harness is testable against a fixture with a known answer key — including the test that
proves a *wrong* answer key scores 0.0, which is the only thing separating a metric from a
function that always passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..ingest import IngestError

QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"
DEFAULT_KS = (1, 3, 5)


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    query: str
    expected_papers: tuple[str, ...]
    note: str = ""

    @property
    def is_probe(self) -> bool:
        """A near-miss: in-domain, answered by no paper in the corpus.

        Scored separately rather than as a miss — its job is to show whether a low top score
        is a usable "I don't know" signal for Phase 4, not to drag recall down.
        """
        return not self.expected_papers


def load_queries(path: Path | None = None) -> list[EvalQuery]:
    payload = json.loads((path or QUERIES_PATH).read_text(encoding="utf-8"))
    queries = [
        EvalQuery(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            expected_papers=tuple(str(value) for value in row.get("expected_papers") or ()),
            note=str(row.get("note") or ""),
        )
        for row in payload["queries"]
    ]
    seen = {query.query_id for query in queries}
    if len(seen) != len(queries):
        raise IngestError("queries.json: duplicate query_id")
    return queries


def _ranked_papers(results: Sequence) -> list[str]:
    """Papers in the order they first appear in the result list. Recall is measured over
    papers, not chunks: two chunks from the right paper is one correct answer."""
    ordered: list[str] = []
    for result in results:
        if result.paper_id not in ordered:
            ordered.append(result.paper_id)
    return ordered


def evaluate(
    retriever: Callable[[str, int], Sequence],
    queries: Sequence[EvalQuery],
    *,
    k: int = 5,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Run every query once and reduce to the reported metrics.

    Recall@k counts a query correct when *any* expected paper appears in the top k chunks —
    the question Phase 4 actually cares about is whether the evidence was in front of it.
    """
    ks = tuple(sorted({int(value) for value in ks if int(value) <= k}))
    hits = {value: 0 for value in ks}
    reciprocal = 0.0
    distinct = 0
    scored = 0
    misses: list[str] = []
    probes: list[dict] = []

    for query in queries:
        results = list(retriever(query.query, k))
        papers = _ranked_papers(results)

        if query.is_probe:
            probes.append(
                {
                    "query_id": query.query_id,
                    "top_score": float(results[0].score) if results else 0.0,
                    "top_paper": papers[0] if papers else "",
                }
            )
            continue

        scored += 1
        distinct += len({result.paper_id for result in results})
        # Rank of the first correct *chunk*, so MRR reflects what the agent reads first.
        rank = next(
            (index + 1 for index, r in enumerate(results) if r.paper_id in query.expected_papers),
            0,
        )
        if rank:
            reciprocal += 1.0 / rank
        else:
            misses.append(query.query_id)
        for value in ks:
            # Recall@k is over the top *k chunks*: an expected paper found at chunk 4 is not
            # a hit at k=3, however few distinct papers happened to precede it.
            if rank and rank <= value:
                hits[value] += 1

    return {
        "n_queries": scored,
        "n_probes": len(probes),
        "k": k,
        "recall": {value: (hits[value] / scored if scored else 0.0) for value in ks},
        "mrr": reciprocal / scored if scored else 0.0,
        "distinct_papers": distinct / scored if scored else 0.0,
        "misses": misses,
        "probes": probes,
    }


def format_report(report: dict, *, label: str = "") -> str:
    recall = "   ".join(f"recall@{value}  {score:.2f}" for value, score in report["recall"].items())
    head = f"eval{' ' + label if label else ''}"
    lines = [
        f"{head}  ({report['n_queries']} queries, {report['n_probes']} probes, k={report['k']})",
        f"  {recall}",
        f"  mrr       {report['mrr']:.2f}   distinct papers per query  {report['distinct_papers']:.1f}",
    ]
    misses = report["misses"]
    lines.append(f"  misses: {len(misses)}" + (f"  ({', '.join(misses)})" if misses else ""))
    if report["probes"]:
        detail = ", ".join(f"{p['query_id']} {p['top_score']:.2f}" for p in report["probes"])
        lines.append(f"  probes (no correct paper exists) top score: {detail}")
    return "\n".join(lines)


def run_eval(
    paths,
    *,
    k: int = 5,
    lambda_mult: float | None = None,
    queries_path: Path | None = None,
    device: str | None = None,
    embedder=None,
    store=None,
) -> dict:
    """The corpus-backed run behind `--eval`. One embedder and one store for every query."""
    from ..retrieve import DEFAULT_LAMBDA, retrieve

    if embedder is None:
        from ..embed import MiniLMEmbedder

        embedder = MiniLMEmbedder(device=device)
    if store is None:
        from ..store import ChunkStore

        store = ChunkStore(paths.chroma_dir)

    lambda_mult = DEFAULT_LAMBDA if lambda_mult is None else lambda_mult

    def retriever(query: str, top_k: int):
        return retrieve(
            paths,
            query,
            k=top_k,
            lambda_mult=lambda_mult,
            embedder=embedder,
            store=store,
        )

    return evaluate(retriever, load_queries(queries_path), k=k)


__all__ = [
    "DEFAULT_KS",
    "QUERIES_PATH",
    "EvalQuery",
    "evaluate",
    "format_report",
    "load_queries",
    "run_eval",
]
