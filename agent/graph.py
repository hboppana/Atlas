"""The compiled state machine — topology and nothing else. docs/19-agent-bringup.md.

Step 1's graph is deliberately two nodes and one edge:

    retrieve -> synthesize -> END

Everything that makes the agent *an agent* — relevance judging, the loop-back edge, tool
selection, citation rendering — is Steps 2-4, and each of them adds nodes and edges *here*.
The moment a node starts deciding what runs next, the separation the Phase 4 skill demands
is gone.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .engine import EngineClient, HttpEngineClient
from .nodes import Retriever, make_retrieve, make_synthesize
from .state import AgentState

__all__ = ["build_graph"]


def build_graph(engine: EngineClient | None = None, retriever: Retriever | None = None):
    """Compile the Step 1 graph, closing over its dependencies.

    Both are injectable so tests compile the *real* topology against fakes — the wiring under
    test is then the wiring that ships, not a hand-rolled sequence of calls.
    """
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", make_retrieve(retriever))
    builder.add_node("synthesize", make_synthesize(engine if engine is not None else HttpEngineClient()))
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()
