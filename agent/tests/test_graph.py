"""The compiled graph — docs/19-agent-bringup.md § The graph.

The *real* topology, compiled, against a fake retriever and a FakeEngineClient. Testing the
wiring by hand-calling the nodes in order would test the test, not the graph.
"""

from __future__ import annotations

import pytest

from agent.engine import FakeEngineClient
from agent.prompts import NO_EVIDENCE_ANSWER
from agent.state import initial_state
from agent.tests.conftest import QUERY

pytest.importorskip("langgraph", reason="Phase 4 dependency; pip install -r requirements.txt")

from agent.graph import build_graph  # noqa: E402


def test_retrieve_then_synthesize_produces_a_grounded_answer(fake_retriever) -> None:
    engine = FakeEngineClient("It tiles the computation and never writes the matrix.")
    graph = build_graph(engine=engine, retriever=fake_retriever)

    final = graph.invoke(initial_state(QUERY, k=2, max_new_tokens=32))

    assert final["output"] == "It tiles the computation and never writes the matrix."
    assert len(final["retrieved_docs"]) == 2
    assert final["used_docs"] == final["retrieved_docs"]
    assert final["retrieval_attempts"] == 1
    assert final["error"] is None


def test_node_visit_order_is_readable_from_the_trace(fake_retriever) -> None:
    """The reducer at work: both nodes' lines survive, in order."""
    final = build_graph(engine=FakeEngineClient(), retriever=fake_retriever).invoke(
        initial_state(QUERY, k=1, max_new_tokens=8)
    )
    steps = final["reasoning_steps"]
    assert steps[0].startswith("retrieve(")
    assert any(step.startswith("synthesize:") for step in steps[1:])
    assert len(steps) >= 2  # nothing overwrote the first node's line


def test_the_graph_hands_the_engine_the_budgeted_prompt(fake_retriever) -> None:
    engine = FakeEngineClient()
    final = build_graph(engine=engine, retriever=fake_retriever).invoke(
        initial_state(QUERY, k=3, max_new_tokens=16)
    )
    assert engine.prompts == [final["prompt"]]
    assert f"Question: {QUERY}" in engine.prompts[0]
    for index in (1, 2, 3):
        assert f"[{index}] " in engine.prompts[0]


def test_empty_corpus_short_circuits_without_touching_the_engine() -> None:
    engine = FakeEngineClient()
    final = build_graph(engine=engine, retriever=lambda query, k: []).invoke(
        initial_state(QUERY, k=5, max_new_tokens=32)
    )
    assert final["output"] == NO_EVIDENCE_ANSWER
    assert engine.prompts == []


def test_a_node_error_ends_the_run_in_state_not_in_a_traceback() -> None:
    def boom(query, k):
        raise RuntimeError("no chroma on this box")

    final = build_graph(engine=FakeEngineClient(), retriever=boom).invoke(
        initial_state(QUERY, k=5, max_new_tokens=32)
    )
    assert "no chroma on this box" in final["error"]
    assert final["output"] == NO_EVIDENCE_ANSWER  # nothing retrieved => nothing invented
