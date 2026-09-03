"""Shared fixtures for agent/tests — docs/19-agent-bringup.md.

The Phase 3 rule, unchanged: this suite runs green on a CPU-only checkout with **no ML
deps, no server and no corpus**. Every test here builds `Result` objects by hand and injects
a fake retriever and a `FakeEngineClient`; a missing optional dependency (langgraph) is a
green SKIP rather than a failure.
"""

from __future__ import annotations

import pytest

from rag.retrieve import Result

QUERY = "how does flash attention avoid materializing the attention matrix"


def make_result(
    rank: int,
    *,
    text: str = "flash attention tiles the computation and never writes the full matrix",
    paper_id: str = "2205.14135",
    title: str = "FlashAttention",
    year: int | None = 2022,
    section: str = "Method",
    score: float = 0.67,
) -> Result:
    return Result(
        chunk_id=f"{paper_id}:{rank}",
        text=text,
        score=score,
        rank=rank,
        paper_id=paper_id,
        title=title,
        year=year,
        section=section,
        page_start=3 + rank,
        page_end=4 + rank,
        topic_label="attention",
    )


@pytest.fixture
def results() -> list[Result]:
    """Three chunks from two papers, MMR-ordered, rank 0 first."""
    return [
        make_result(0, score=0.67),
        make_result(1, score=0.61, text="the softmax is computed in a streaming fashion over tiles"),
        make_result(
            2,
            score=0.53,
            paper_id="2309.06180",
            title="PagedAttention",
            year=2023,
            section="Background",
            text="a paged kv cache stores attention keys and values in non-contiguous blocks",
        ),
    ]


@pytest.fixture
def fake_retriever(results):
    """A retriever that records its calls and returns the first `k` fixtures."""

    calls: list[tuple[str, int]] = []

    def _retrieve(query: str, k: int):
        calls.append((query, k))
        return results[:k]

    _retrieve.calls = calls
    return _retrieve
