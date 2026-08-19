"""MMR retrieval over the chunk store — docs/18-rag-retrieval.md.

    question  ->  HNSW top-`fetch_k` (filters applied in Chroma)  ->  exact MMR  ->  k Results

Two stages, and the split is the shape of the step. Chroma answers "what is close to this
question", which is approximate and cheap; MMR answers "which of those should the agent
actually see", which is exact and runs over 40 vectors in numpy. Diversifying inside the
index is not possible, and diversifying over all 4148 chunks would mean a similarity matrix
per query for a selection that only ever returns five rows.

Everything here is a pure function of (query vector, candidate rows) except the one call
into `ChunkStore`, so the whole selection policy — MMR, the per-paper cap, the filter
translation, the result shape — is testable with no Chroma, no torch and no corpus.

This module is the entire retrieval surface Phase 4's agent depends on: it imports
`retrieve()` and never touches Chroma, numpy or the manifest itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .ingest import IngestError, Paths

# What the caller gets back.
DEFAULT_K = 5
# Candidates pulled from the index before MMR selects among them. 8x the default k: enough
# room for diversity to mean something, still one HNSW call.
DEFAULT_FETCH_K = 40
# 1.0 = pure relevance (identical to plain top-k), 0.0 = pure diversity. 0.7 is measured,
# not guessed: on the Step 4 eval set it beats plain top-k on recall@3 and MRR while raising
# distinct papers per query, and every value below it starts costing recall@5. See
# docs/18-rag-retrieval.md § Results — it was chosen on that set, so it is a fitted default.
DEFAULT_LAMBDA = 0.7
# MMR diversifies in vector space, which is not the same as diversifying across papers: a
# paper with 200 chunks can fill the result with passages that are genuinely different from
# each other. 0 disables the cap.
MAX_PER_PAPER = 2


@dataclass(frozen=True)
class Result:
    """One retrieved chunk, in the shape Phase 4 formats citations from.

    A frozen dataclass rather than the raw dict `ChunkStore.query` returns: a typo in a key
    should be an `AttributeError` at the call site, not an empty citation inside an answer.
    """

    chunk_id: str  # "<paper_id>:<index>" — what cite_sources resolves
    text: str  # the chunk, verbatim, for quoting
    score: float  # cosine similarity to the query — NOT the MMR objective
    rank: int  # 0-based position after selection
    paper_id: str
    title: str
    year: int | None
    section: str
    page_start: int
    page_end: int
    topic_label: str

    @classmethod
    def from_hit(cls, hit: dict, rank: int) -> "Result":
        metadata = hit.get("metadata") or {}
        chunk_id = str(hit.get("chunk_id") or "")
        year = metadata.get("year")
        return cls(
            chunk_id=chunk_id,
            text=str(hit.get("text") or ""),
            score=float(hit.get("score") or 0.0),
            rank=rank,
            # `year` is absent from the metadata of a paper with no year (a local PDF), so
            # it reads back as None rather than raising.
            paper_id=str(metadata.get("paper_id") or chunk_id.rsplit(":", 1)[0]),
            title=str(metadata.get("title") or ""),
            year=int(year) if year is not None else None,
            section=str(metadata.get("section") or ""),
            page_start=int(metadata.get("page_start") or 0),
            page_end=int(metadata.get("page_end") or 0),
            topic_label=str(metadata.get("topic_label") or ""),
        )

    def citation(self) -> str:
        pages = f"p{self.page_start}" if self.page_start == self.page_end else f"pp{self.page_start}-{self.page_end}"
        year = f" ({self.year})" if self.year else ""
        return f"{self.title or self.paper_id}{year}, {pages} [{self.chunk_id}]"


# --------------------------------------------------------------------------------------
# MMR
# --------------------------------------------------------------------------------------


def mmr(
    relevance: Sequence[float],
    vectors: np.ndarray,
    *,
    k: int = DEFAULT_K,
    lambda_mult: float = DEFAULT_LAMBDA,
    groups: Sequence[str] | None = None,
    max_per_group: int = 0,
) -> list[int]:
    """Maximal marginal relevance (Carbonell & Goldstein) -> selected candidate indices.

        score(c) = λ · sim(q, c) − (1 − λ) · max sim(c, s)   over s already selected

    `relevance` is passed in rather than recomputed from `vectors` so the number that ranks
    a chunk is the same number the store reported and the caller sees.

    Greedy, O(fetch_k · k · dim), which at 40 candidates is microseconds — there is no
    reason to approximate it. Vectors are unit-norm by contract (`rag.embed` normalizes at
    write time), so every similarity here is a plain dot product.

    Determinism matters more than it looks: ties break by candidate rank, never by argmax
    over float noise, because a retriever that reorders between identical runs cannot be
    evaluated.
    """
    relevance = np.asarray(relevance, dtype=np.float64).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(relevance)
    k = min(int(k), n)
    if k <= 0:
        return []
    if not 0.0 <= lambda_mult <= 1.0:
        raise IngestError(f"lambda must be in [0, 1], got {lambda_mult}")
    if len(vectors) != n:
        raise IngestError(f"mmr: {n} relevance scores but {len(vectors)} vectors")

    counts: dict[str, int] = {}
    selected: list[int] = []
    # The cap is a preference, not a quota. If every remaining candidate belongs to a paper
    # that has filled it, the cap is dropped for the rest of the selection rather than
    # returning fewer than k — under-filling k is the exact failure that post-filtering
    # causes, and it looks like bad retrieval instead of like a knob.
    capped = groups is not None and max_per_group > 0
    # Best similarity of each candidate to anything already selected. -inf until the first
    # pick, so the first selection is pure relevance whatever λ is.
    redundancy = np.full(n, -np.inf, dtype=np.float64)
    taken = np.zeros(n, dtype=bool)

    # A while loop, not `for _ in range(k)`: relaxing the cap has to re-pick the slot it
    # failed on rather than consuming it.
    while len(selected) < k:
        best_index, best_score = -1, -np.inf
        for index in range(n):
            if taken[index]:
                continue
            if capped and counts.get(groups[index], 0) >= max_per_group:
                continue
            if not selected:
                score = relevance[index]
            else:
                score = lambda_mult * relevance[index] - (1.0 - lambda_mult) * redundancy[index]
            # Strictly greater: the first candidate of a tied pair wins, and candidates
            # arrive in the store's relevance order.
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0:
            if capped:
                capped = False  # relax and re-pick this slot
                continue
            break

        selected.append(best_index)
        taken[best_index] = True
        if groups is not None:
            counts[groups[best_index]] = counts.get(groups[best_index], 0) + 1
        similarities = vectors @ vectors[best_index]
        redundancy = np.maximum(redundancy, similarities.astype(np.float64))

    return selected


# --------------------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------------------


def topic_labels(paths: Paths) -> list[str]:
    """The labels currently derived on disk, in topic_id order."""
    path = paths.topics_path
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(topic.get("label") or "") for topic in payload.get("topics") or []]


def resolve_topic_label(paths: Paths, label: str) -> str:
    """Match a user-supplied topic against `topics.json`, case- and prefix-tolerantly.

    Labels, never `topic_id`, are the filter currency at this boundary: Step 3 recorded that
    the integer is not stable across corpus growth, so an ID memorized in a Phase 4 prompt
    or an eval fixture would rot invisibly. An unknown label names the known ones instead of
    quietly returning nothing, which is the failure mode that looks like bad retrieval.
    """
    known = topic_labels(paths)
    wanted = label.strip().lower()
    for candidate in known:
        if candidate.lower() == wanted:
            return candidate
    matches = [candidate for candidate in known if candidate.lower().startswith(wanted)]
    if len(matches) == 1:
        return matches[0]
    known_text = "; ".join(repr(candidate) for candidate in known) or "none derived yet"
    raise IngestError(f"unknown topic {label!r}. Derived topics: {known_text}")


def build_where(
    *,
    paper_id: str | Sequence[str] | None = None,
    topic_label: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    section: str | None = None,
) -> dict | None:
    """Translate the filter arguments into one Chroma `where` clause.

    These go into the *fetch*, never applied to the selected results: post-filtering silently
    returns fewer than `k` rows, which reads as a retrieval-quality problem and is not one.
    """
    clauses: list[dict] = []
    if paper_id:
        ids = [paper_id] if isinstance(paper_id, str) else list(paper_id)
        clauses.append({"paper_id": {"$in": [str(value) for value in ids]}})
    if topic_label:
        clauses.append({"topic_label": str(topic_label)})
    if year_min is not None:
        clauses.append({"year": {"$gte": int(year_min)}})
    if year_max is not None:
        clauses.append({"year": {"$lte": int(year_max)}})
    if section:
        clauses.append({"section": str(section)})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# --------------------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------------------


def retrieve(
    paths: Paths,
    query: str,
    *,
    k: int = DEFAULT_K,
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_LAMBDA,
    max_per_paper: int = MAX_PER_PAPER,
    paper_id: str | Sequence[str] | None = None,
    topic_label: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    section: str | None = None,
    embedder=None,
    device: str | None = None,
    store=None,
) -> list[Result]:
    """The retrieval surface. Everything downstream of Phase 3 calls this and nothing else.

    `embedder` and `store` are injectable for the same reason they are everywhere else in
    `rag/`: the selection policy must be testable without torch, chromadb or the corpus.
    """
    if not query or not query.strip():
        raise IngestError("retrieve: empty query")
    k = max(1, int(k))
    fetch_k = max(k, int(fetch_k))

    if topic_label:
        topic_label = resolve_topic_label(paths, topic_label)
    where = build_where(
        paper_id=paper_id,
        topic_label=topic_label,
        year_min=year_min,
        year_max=year_max,
        section=section,
    )

    if embedder is None:
        from .embed import MiniLMEmbedder

        embedder = MiniLMEmbedder(device=device)
    if store is None:
        from .store import ChunkStore

        store = ChunkStore(paths.chroma_dir)

    from .embed import _l2_normalize

    vector = _l2_normalize(np.asarray(embedder.encode([query]), dtype=np.float32))[0]
    hits = store.query(vector, k=fetch_k, where=where, include_vectors=True)
    if not hits:
        return []

    relevance = [float(hit.get("score") or 0.0) for hit in hits]
    vectors = np.stack(
        [
            np.asarray(hit.get("vector"), dtype=np.float32)
            if hit.get("vector") is not None
            else np.zeros_like(vector)
            for hit in hits
        ]
    )
    groups = [str((hit.get("metadata") or {}).get("paper_id") or "") for hit in hits]

    order = mmr(
        relevance,
        vectors,
        k=k,
        lambda_mult=lambda_mult,
        groups=groups,
        max_per_group=max_per_paper,
    )
    return [Result.from_hit(hits[index], rank) for rank, index in enumerate(order)]


def format_results(query: str, results: Sequence[Result]) -> str:
    """The `--search` rendering. Kept next to the dataclass so the CLI has no formatting."""
    lines = [f"\nsearch: {query!r}  ({len(results)} result(s))"]
    for result in results:
        lines.append(
            f"  {result.rank + 1}. {result.score:.3f}  {result.chunk_id}  "
            f"p{result.page_start}-{result.page_end}  "
            f"[{result.section or 'n/a'}]  {result.title}"
        )
        lines.append(f"       {' '.join(result.text.split())[:200]}")
    papers = {result.paper_id for result in results}
    lines.append(f"  ({len(papers)} distinct paper(s))")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_FETCH_K",
    "DEFAULT_K",
    "DEFAULT_LAMBDA",
    "MAX_PER_PAPER",
    "Result",
    "build_where",
    "format_results",
    "mmr",
    "resolve_topic_label",
    "retrieve",
    "topic_labels",
]
