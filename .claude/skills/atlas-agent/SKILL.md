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
| `prompts.py` | templates, source-marker rendering, the prompt token budget (data, not logic) |
| `engine.py` | `EngineClient` protocol + `HttpEngineClient`/`Fake`/`DryRun` — the link to Phase 2 |
| `tools.py` | tool definitions: `search_papers`, `query_corpus`, `summarize`, `compare` |
| `state.py` | **TypedDict** state schema: `query`, `retrieved_docs`, `reasoning_steps`, `output` |
| `cli.py` | `python -m agent.cli "question"` — one question, one run |
| `__init__.py` | package marker |

**Step 1 is done** (docs/19-agent-bringup.md): `retrieve → synthesize`, 34 tests, and a
real grounded answer off the A6000. Read that doc before touching `agent/`; the facts below
are the ones a later step would otherwise pay to rediscover.

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
- **Nodes are built by factories that close over their dependencies** (`make_retrieve(retriever)`,
  `make_synthesize(engine)`), and `build_graph(engine=, retriever=)` passes them through. That
  is what lets a test compile the shipping topology against a `FakeEngineClient`. Same
  injection discipline as `rag/retrieve.py`'s `embedder=`/`store=`.
- **A node failure lands in `state["error"]`**, never as a traceback out of the graph. The run
  still has a query, a trace, and something the CLI can print.
- `langgraph` is the only Phase 4 dep in `requirements.txt`; `langchain` lands when a step
  actually uses it.

## Facts Step 1 measured (don't re-derive these)

- **Cost is a function of prompt length, not a per-token constant.** No KV cache, 2048-token
  window. On the A6000: 165 prompt tokens → ~247 ms/token, 529 → ~700, 834 → ~1144. k=5 with
  64 new tokens is an **80 s** run; k=3 with 32 is **29 s**.
- **The agent's defaults are `k=3`, `max_new_tokens=32`** (`agent/nodes.py`), deliberately not
  `rag.retrieve.DEFAULT_K` (5). More evidence also made the *answer* worse — at k=5 TinyLlama
  started reciting a retrieved title and inventing authors.
- **The agent owns the chat template.** The server applies none; the weights are Zephyr-tuned,
  so `agent/prompts.py` emits `<|system|>…</s>\n<|user|>…</s>\n<|assistant|>\n`.
- **Token counts come from `POST /tokenize`** on the Phase 2 server, never a heuristic
  (docs/17 measured what guessing costs). When the server is unreachable the count is an
  over-estimate and is flagged `exact=False` all the way into the state and the trace.
- **Budgeting drops whole low-ranked chunks** to `PROMPT_TOKEN_BUDGET = 1536`, never truncates
  one mid-sentence.
- **Empty retrieval means no engine call.** A 1.1B model answering from parametric memory is
  fluent and wrong.
- **`--dry-run` swaps the client, not a branch inside `synthesize`**, so it exercises the
  shipping graph.

## Running it on this box

Two interpreters, and that is the design working: `.venv` (3.10) matches the extension
module's ABI and runs the server (`PYTHON=.venv/bin/python scripts/run_server.sh`); the
system python3 (3.11) has chromadb/sentence-transformers/langgraph and runs the agent
(`python3 -m agent.cli`). They meet over HTTP and neither needs the other's deps.
