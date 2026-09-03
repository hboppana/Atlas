"""retrieve and synthesize — docs/19-agent-bringup.md § Nodes and § The prompt.

Node logic in isolation: what each returns, what the prompt looks like, and what the budget
drops. The graph that wires them together is test_graph.py's job.
"""

from __future__ import annotations

import pytest

from agent.engine import AgentError, FakeEngineClient, TokenCount
from agent.nodes import make_retrieve, make_synthesize
from agent.prompts import MAX_POSITION, NO_EVIDENCE_ANSWER, budget_docs, build_prompt
from agent.state import initial_state
from agent.tests.conftest import QUERY, make_result


# ---------------------------------------------------------------------------- retrieve


def test_retrieve_returns_docs_attempts_and_a_trace_line(fake_retriever) -> None:
    state = initial_state(QUERY, k=2, max_new_tokens=8)
    out = make_retrieve(fake_retriever)(state)

    assert fake_retriever.calls == [(QUERY, 2)]
    assert [doc.rank for doc in out["retrieved_docs"]] == [0, 1]
    assert out["retrieval_attempts"] == 1  # Step 1 never loops
    assert len(out["reasoning_steps"]) == 1
    assert "0.670" in out["reasoning_steps"][0]  # the top score is in the trace
    assert "error" not in out


def test_retrieve_returns_a_partial_dict() -> None:
    """A node that returns the whole state is one that can clobber a key it never saw."""
    out = make_retrieve(lambda query, k: [])(initial_state(QUERY, k=5, max_new_tokens=8))
    assert set(out) == {"retrieved_docs", "retrieval_attempts", "reasoning_steps"}


def test_empty_retrieval_is_not_an_error() -> None:
    """Step 2's edge decides what an empty result means; the node just reports it."""
    out = make_retrieve(lambda query, k: [])(initial_state(QUERY, k=5, max_new_tokens=8))
    assert out["retrieved_docs"] == []
    assert out.get("error") is None
    assert "no chunks" in out["reasoning_steps"][0]


def test_retrieval_failure_lands_in_state_rather_than_raising() -> None:
    def boom(query, k):
        raise RuntimeError("chroma is not on this box")

    out = make_retrieve(boom)(initial_state(QUERY, k=5, max_new_tokens=8))
    assert "chroma is not on this box" in out["error"]
    assert out["retrieved_docs"] == []


# --------------------------------------------------------------------------- synthesize


def _synthesized(results, *, engine=None, max_new_tokens=48, limit=1536):
    engine = engine or FakeEngineClient("Flash attention tiles the computation.")
    state = initial_state(QUERY, k=len(results), max_new_tokens=max_new_tokens)
    state["retrieved_docs"] = list(results)
    return make_synthesize(engine, limit=limit)(state), engine


def test_synthesize_calls_the_engine_and_records_what_it_sent(results) -> None:
    out, engine = _synthesized(results)
    assert out["output"] == "Flash attention tiles the computation."
    assert engine.prompts == [out["prompt"]]
    assert out["used_docs"] == results
    assert out["prompt_tokens_exact"] is True


def test_prompt_carries_the_zephyr_markers_the_chat_weights_expect(results) -> None:
    """The server applies no chat template, so the agent owns the real one."""
    out, _ = _synthesized(results)
    prompt = out["prompt"]
    assert prompt.startswith("<|system|>\n")
    assert "<|user|>\n" in prompt
    assert prompt.endswith("<|assistant|>\n")
    assert prompt.count("</s>") == 2  # one per closed turn; the assistant turn is open


def test_prompt_carries_numbered_sources_and_the_query(results) -> None:
    out, _ = _synthesized(results)
    prompt = out["prompt"]
    for index, doc in enumerate(results, start=1):
        assert f"[{index}] " in prompt  # what Step 3's cite_sources resolves
        assert " ".join(doc.text.split()) in prompt
    assert "FlashAttention (2022), Method, pp3-4" in prompt
    assert f"Question: {QUERY}" in prompt


def test_budget_keeps_a_rank_ordered_prefix_and_drops_the_tail() -> None:
    """12 chunks of ~200 tokens against a 1536-token budget."""
    chunks = [make_result(rank, text=" ".join(f"w{rank}x{i}" for i in range(200))) for rank in range(12)]
    engine = FakeEngineClient()
    out, _ = _synthesized(chunks, engine=engine)

    used = out["used_docs"]
    assert 0 < len(used) < len(chunks)
    assert used == chunks[: len(used)]  # a prefix in MMR rank order, never a cherry-pick
    assert out["prompt_tokens"] <= 1536
    assert engine.count_tokens(out["prompt"]).n_tokens == out["prompt_tokens"]
    assert "dropped lowest-ranked" in " ".join(out["reasoning_steps"])
    assert chunks[-1].chunk_id in " ".join(out["reasoning_steps"])


def test_budget_leaves_room_for_the_completion_in_the_2048_window() -> None:
    """The prompt and the completion share the window; whichever limit binds first wins."""
    counter = lambda text: TokenCount(n_tokens=len(text.split()), exact=True)  # noqa: E731
    budget = budget_docs(QUERY, [], counter, max_new_tokens=600, limit=1536)
    assert budget.limit == MAX_POSITION - 600


def test_estimated_counts_are_flagged_in_the_state_and_the_trace(results) -> None:
    """A budget decision is never silently based on a guess."""

    class Estimating(FakeEngineClient):
        def count_tokens(self, text):
            return TokenCount(n_tokens=len(text.split()), exact=False)

    out, _ = _synthesized(results, engine=Estimating())
    assert out["prompt_tokens_exact"] is False
    assert "ESTIMATED" in out["reasoning_steps"][0]


def test_no_evidence_means_no_engine_call() -> None:
    """A RAG system answering from parametric memory is worse than one that says it can't."""
    engine = FakeEngineClient()
    out, _ = _synthesized([], engine=engine)
    assert out["output"] == NO_EVIDENCE_ANSWER
    assert engine.prompts == []
    assert out["used_docs"] == []


def test_engine_failure_lands_in_state_with_the_prompt_preserved(results) -> None:
    class Broken(FakeEngineClient):
        def complete(self, prompt, max_new_tokens):
            raise AgentError("/generate returned 500: engine died")

    out, _ = _synthesized(results, engine=Broken())
    assert "500" in out["error"]
    assert out["output"] == ""
    assert out["prompt"]  # kept, so the failing run is reproducible by hand


def test_generation_budget_is_passed_through_and_clamped(results) -> None:
    engine = FakeEngineClient(max_new_tokens=16)
    _synthesized(results, engine=engine, max_new_tokens=64)
    assert engine.budgets == [16]


def test_build_prompt_without_docs_still_asks_the_question() -> None:
    prompt = build_prompt(QUERY, [])
    assert f"Question: {QUERY}" in prompt
    assert "[1]" not in prompt
