"""Atlas Phase 4 — the LangGraph agent over the Phase 3 corpus and the Phase 2 engine.

See docs/19-agent-bringup.md. Step 1 is on disk; the rest lands step by step:

    agent.state      the AgentState TypedDict every node is written against
    agent.engine     HTTP client for the Phase 2 server (generation + tokenization)
    agent.prompts    the Zephyr template, source markers and the prompt token budget
    agent.nodes      (Step 1) retrieve, synthesize
    agent.graph      the compiled state machine — topology and nothing else
    agent.cli        python -m agent.cli "question"

The agent orchestrates. Retrieval lives in `rag/`, inference lives behind the Phase 2
server, and neither is reimplemented here.
"""
