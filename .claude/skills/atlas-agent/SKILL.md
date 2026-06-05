---
name: atlas-agent
description: Phase 4 — the LangGraph multi-step agent (agent/). Covers the stateful graph (retrieve → evaluate relevance → synthesize → cite sources), the node/tool/state split, and the TypedDict state schema. Use when writing or reviewing agent/ (graph, nodes, tools, state).
---

# Atlas Phase 4 — LangGraph Agent

Orchestrate multi-step reasoning over the RAG corpus as a stateful graph. **Prerequisite:**
the RAG pipeline (Phase 3) and the serving engine work. Read `atlas-architecture` first. The
agent reasons with **Atlas's own inference engine** (via the Phase 2 server), not a hosted LLM.

## The graph (agent/)

A stateful LangGraph state machine. Canonical flow:

```
retrieve → evaluate_relevance → synthesize → cite_sources
```

with conditional edges (e.g. loop back to `retrieve` if relevance is insufficient).

| File | Responsibility |
|------|----------------|
| `graph.py` | the LangGraph state machine — node registration, transitions, conditional edges |
| `nodes.py` | the node functions: `retrieve`, `evaluate_relevance`, `synthesize`, `cite_sources` |
| `tools.py` | tool definitions: `search_papers`, `query_corpus`, `summarize`, `compare` |
| `state.py` | **TypedDict** state schema: `query`, `retrieved_docs`, `reasoning_steps`, `output` |
| `__init__.py` | package marker |

## Conventions

- **State is an explicit TypedDict** threaded through every node — `query`,
  `retrieved_docs`, `reasoning_steps`, `output`. Nodes read and extend it; keep mutations
  explicit and serializable.
- **Separation of concerns:** `graph.py` = topology, `nodes.py` = step logic, `tools.py` =
  callable capabilities, `state.py` = the data contract. Don't collapse these.
- **Citations are first-class.** `cite_sources` relies on the metadata the Phase 3 chunker
  attached (source paper, section). Synthesis must remain traceable to retrieved evidence —
  no unsupported claims.
- Retrieval and tools call into the existing `rag/` pipeline and the Phase 2 server; the
  agent layer orchestrates, it does not reimplement retrieval or inference.
- Uncomment the Phase 4 deps in `requirements.txt`: `langchain`, `langgraph`.
