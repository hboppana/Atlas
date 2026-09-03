"""The state contract — docs/19-agent-bringup.md § State schema.

The reducer on `reasoning_steps` is tested *before* the loop that needs it exists (Step 2),
because with plain assignment the bug is invisible until the loop makes a node run twice.
"""

from __future__ import annotations

import operator
from typing import Annotated, get_args, get_type_hints

import pytest

from agent.state import AgentState, initial_state
from agent.tests.conftest import QUERY, make_result


def test_initial_state_is_the_input_and_nothing_else() -> None:
    state = initial_state(QUERY, k=3, max_new_tokens=48)
    assert state["query"] == QUERY
    assert state["k"] == 3
    assert state["max_new_tokens"] == 48
    assert state["retrieval_attempts"] == 0
    assert state["reasoning_steps"] == []
    assert state["error"] is None
    # Nothing downstream is invented up front: no output, no docs, no citations.
    assert "output" not in state and "retrieved_docs" not in state


def test_every_key_is_optional() -> None:
    """`total=False` is what lets a node return a partial dict."""
    assert AgentState.__total__ is False
    assert not AgentState.__required_keys__


def test_reasoning_steps_is_the_only_reducer_field() -> None:
    hints = get_type_hints(AgentState, include_extras=True)
    reducers = {
        key: get_args(value)[1]
        for key, value in hints.items()
        if getattr(value, "__metadata__", None)
    }
    assert reducers == {"reasoning_steps": operator.add}


def test_retrieved_docs_holds_result_objects_not_dicts() -> None:
    """Phase 3's argument: a typo should be an AttributeError, not an empty citation."""
    result = make_result(0)
    state = initial_state(QUERY, k=1, max_new_tokens=8)
    state["retrieved_docs"] = [result]
    assert state["retrieved_docs"][0].chunk_id == "2205.14135:0"
    assert "2205.14135:0" in state["retrieved_docs"][0].citation()
    with pytest.raises(AttributeError):
        state["retrieved_docs"][0].tittle  # noqa: B018 — the point is that it raises


def test_partial_returns_merge_and_traces_concatenate() -> None:
    """The merge semantics the graph relies on, exercised without langgraph.

    Two successive nodes: the second's partial dict overwrites `output` (last-write-wins)
    while `reasoning_steps` accumulates both lines (the reducer).
    """
    state = dict(initial_state(QUERY, k=1, max_new_tokens=8))
    first = {"output": "draft", "reasoning_steps": ["retrieve -> 1 chunk"]}
    second = {"output": "final", "reasoning_steps": ["synthesize -> 12 words"]}

    for partial in (first, second):
        for key, value in partial.items():
            if key == "reasoning_steps":
                state[key] = operator.add(state[key], value)
            else:
                state[key] = value

    assert state["output"] == "final"
    assert state["reasoning_steps"] == ["retrieve -> 1 chunk", "synthesize -> 12 words"]
    assert state["query"] == QUERY  # untouched keys survive a partial return
