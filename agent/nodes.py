"""The node functions — docs/19-agent-bringup.md.

Step 1 has two: `retrieve` and `synthesize`. `evaluate_relevance` (Step 2) and
`cite_sources` (Step 3) land next to them.

Nodes are built by factories that close over their dependencies (`make_retrieve(retriever)`,
`make_synthesize(engine)`) rather than reaching for a module-level singleton, so a test
compiles a real graph with a fake retriever and a `FakeEngineClient` and never touches a GPU,
a server or the corpus. Same injection discipline as `rag/retrieve.py`'s `embedder=`/`store=`.

Every node returns a **partial** dict. LangGraph merges it into the state; returning the
whole state is how a node silently clobbers a key it never thought about.
"""

from __future__ import annotations

from typing import Callable, Sequence

from rag.ingest import Paths
from rag.retrieve import Result

from .engine import AgentError, EngineClient
from .prompts import NO_EVIDENCE_ANSWER, PROMPT_TOKEN_BUDGET, budget_docs
from .state import AgentState

__all__ = [
    "DEFAULT_K",
    "DEFAULT_MAX_NEW_TOKENS",
    "Retriever",
    "corpus_retriever",
    "make_retrieve",
    "make_synthesize",
]

# What a node needs from retrieval, and nothing more.
Retriever = Callable[[str, int], Sequence[Result]]

# The agent's defaults, NOT rag's (rag.retrieve.DEFAULT_K is 5). Both are set by the
# measurement in docs/19 § Results rather than by taste: with no KV cache the run cost is
# dominated by prompt length, and on the A6000 k=5/64 tokens takes 80 s while k=3/32 takes
# 29 s -- and the k=3 answer was the better-grounded of the two. `--k`/`--max-new-tokens`
# raise them when a question is worth the wait.
DEFAULT_K = 3
DEFAULT_MAX_NEW_TOKENS = 32


def corpus_retriever(paths: Paths | None = None, **kwargs) -> Retriever:
    """The real retriever: `rag.retrieve.retrieve` over `data/`.

    Imported inside the closure so that importing `agent.nodes` does not drag chromadb and
    sentence-transformers into a `--dry-run` on a checkout that has neither.
    """
    resolved = paths or Paths.default()

    def _retrieve(query: str, k: int) -> Sequence[Result]:
        from rag.retrieve import retrieve as rag_retrieve

        return rag_retrieve(resolved, query, k=k, **kwargs)

    return _retrieve


def _describe(docs: Sequence[Result]) -> str:
    if not docs:
        return "no chunks"
    papers = {doc.paper_id for doc in docs}
    return f"{len(docs)} chunk(s) from {len(papers)} paper(s), top score {docs[0].score:.3f}"


def make_retrieve(retriever: Retriever | None = None):
    """`retrieve(state) -> dict`: pull evidence, and do nothing else with it."""
    resolve = retriever if retriever is not None else corpus_retriever()

    def retrieve(state: AgentState) -> dict:
        query = state.get("query") or ""
        k = int(state.get("k") or DEFAULT_K)
        attempts = int(state.get("retrieval_attempts") or 0) + 1
        try:
            docs = list(resolve(query, k))
        except Exception as exc:
            # A retrieval failure is recorded state, not a traceback out of the graph: the
            # run still has a query, a trace, and now an error the CLI can print.
            return {
                "retrieved_docs": [],
                "retrieval_attempts": attempts,
                "error": f"retrieve: {exc}",
                "reasoning_steps": [f"retrieve(k={k}) failed: {exc}"],
            }
        # An empty result is not an exception. Step 2's edge decides what it means.
        return {
            "retrieved_docs": docs,
            "retrieval_attempts": attempts,
            "reasoning_steps": [f"retrieve(k={k}) -> {_describe(docs)}"],
        }

    return retrieve


def make_synthesize(engine: EngineClient, *, limit: int = PROMPT_TOKEN_BUDGET):
    """`synthesize(state) -> dict`: budget the evidence, render the prompt, generate."""

    def synthesize(state: AgentState) -> dict:
        query = state.get("query") or ""
        docs = list(state.get("retrieved_docs") or [])
        max_new_tokens = int(state.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)

        if not docs:
            # No engine call. Asking an ungrounded 1.1B model an open question is how a RAG
            # system starts confidently inventing papers.
            return {
                "output": NO_EVIDENCE_ANSWER,
                "used_docs": [],
                "prompt": "",
                "reasoning_steps": ["synthesize: no evidence retrieved; engine not called"],
            }

        budget = budget_docs(
            query,
            docs,
            engine.count_tokens,
            max_new_tokens=max_new_tokens,
            limit=limit,
        )
        trace = [
            f"synthesize: {len(budget.used)}/{len(docs)} chunk(s) fit the "
            f"{budget.limit}-token budget ({budget.n_tokens} tokens, "
            f"{'exact' if budget.exact else 'ESTIMATED'})"
        ]
        if budget.dropped:
            trace.append(
                "synthesize: dropped lowest-ranked "
                + ", ".join(doc.chunk_id for doc in budget.dropped)
            )

        try:
            output = engine.complete(budget.prompt, max_new_tokens)
        except AgentError as exc:
            return {
                "prompt": budget.prompt,
                "used_docs": budget.used,
                "prompt_tokens": budget.n_tokens,
                "prompt_tokens_exact": budget.exact,
                "output": "",
                "error": str(exc),
                "reasoning_steps": trace + [f"synthesize: engine call failed: {exc}"],
            }

        return {
            "prompt": budget.prompt,
            "used_docs": budget.used,
            "prompt_tokens": budget.n_tokens,
            "prompt_tokens_exact": budget.exact,
            "output": output.strip(),
            "reasoning_steps": trace + [f"synthesize: generated {len(output.split())} word(s)"],
        }

    return synthesize
