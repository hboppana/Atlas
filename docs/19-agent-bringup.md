# Phase 4 · Step 1 — `agent/` bring-up: state contract, engine client, `retrieve → synthesize`

> Status: **designed 2026-08-24** — not yet implemented.
> Predecessor: Phase 3 Step 4 — MMR retrieval + eval set — **done**
> ([18-rag-retrieval.md](18-rag-retrieval.md)). recall@5 **0.94** on 36 queries, 43 papers,
> 4148 chunks in Chroma.
> Successor: Step 2 — `evaluate_relevance` + the conditional loop-back edge.

## Goal

Stand up `agent/` and prove one thing end to end: **a question goes in, and an answer
grounded in the corpus comes out of Atlas's own engine.**

```
question → retrieve (rag/) → synthesize (TinyLlama via the Phase 2 server) → answer
```

Two nodes, one linear graph, no branching. Everything that makes the agent *an agent* —
relevance judging, the retry loop, tool selection, citation rendering — is Steps 2–4. What
this step buys is the three contracts every later step is written against:

1. **`AgentState`** — the TypedDict threaded through every node, defined once, in full.
2. **The engine client** — how a node calls TinyLlama, and what it does about a 1.1B model
   with a 2048-token window and no KV cache.
3. **The graph shell** — compile, invoke, and get a typed result back.

Get these wrong and Steps 2–4 pay for it in churn. That is why a two-node graph is a whole
step.

### Scope boundary (what Step 1 is *not*)

- **Not `evaluate_relevance`, not the loop-back edge.** The conditional edge is the reason
  LangGraph is here at all, and it needs a threshold. Phase 3 measured what scores mean
  (docs/18: well-answered questions top out ≈0.67, near-miss probes ≈0.42–0.50) and
  explicitly warned against trusting a hard cut. Choosing that policy is Step 2's whole job.
- **Not `cite_sources`.** Step 1's synthesis prompt carries source markers and the node
  returns the `Result` objects it used, so the evidence is *present*; turning it into
  rendered, verified citations — and failing an answer that cites what it wasn't given — is
  Step 3.
- **Not `tools.py`.** `search_papers` / `query_corpus` / `summarize` / `compare` are Step 4.
  Step 1 calls `rag.retrieve.retrieve()` directly from the node.
- **Not re-implementing retrieval or inference.** The agent layer orchestrates. If retrieval
  is wrong the fix is in `rag/`; if generation is wrong the fix is in `engine/`.
- **Not a chat surface.** No history, no multi-turn, no sessions. One question, one run.
- **Not sampling.** Decoding is greedy (docs/13), which is what makes an agent run
  reproducible. A temperature knob lands when the engine has one, not before.

## The operation

```
python -m agent.cli "how does flash attention avoid materializing the attention matrix"
python -m agent.cli --k 3 --max-new-tokens 48 --json "what is a paged kv cache"
python -m agent.cli --dry-run "..."      # retrieve + build the prompt, never call the engine
```

`--dry-run` matters more than it looks: it is the only way to exercise the whole graph on a
checkout with no GPU and no running server, and it is how the prompt budget below gets
inspected without paying 30 s of generation for every look.

The server is a prerequisite for a real run, exactly as documented in docs/14:

```
scripts/run_server.sh              # uvicorn server.serve:app, card pinned via CUDA_VISIBLE_DEVICES
curl localhost:8000/health         # {"status":"ok","device":"cuda",...}
```

## State schema (`agent/state.py`)

The full Phase 4 contract is defined **now**, in one place, even though Step 1 populates
only part of it. A TypedDict that grows a key per step is a contract nobody can read.

```python
class AgentState(TypedDict, total=False):
    # --- input ---
    query: str                       # the user's question, verbatim
    k: int                           # retrieval breadth for this run
    max_new_tokens: int              # generation budget for this run

    # --- retrieval (Step 1) ---
    retrieved_docs: list[Result]     # rag.retrieve.Result, MMR-ordered, rank 0 first
    retrieval_attempts: int          # Step 2's loop guard; Step 1 always leaves it 1

    # --- relevance (Step 2) ---
    relevance: float | None          # judged sufficiency of retrieved_docs
    needs_retry: bool

    # --- synthesis (Step 1) ---
    output: str                      # the answer text
    prompt: str                      # the exact prompt sent to the engine
    used_docs: list[Result]          # the subset that survived the token budget

    # --- citations (Step 3) ---
    citations: list[str]

    # --- always ---
    reasoning_steps: Annotated[list[str], operator.add]
    error: str | None
```

Three decisions are load-bearing:

- **`reasoning_steps` is a reducer field** (`Annotated[list[str], operator.add]`). Every
  other key is last-write-wins, which is what you want for `output`; the trace is the one
  thing nodes must *append* to. Step 2's loop visits `retrieve` twice, and with plain
  assignment the second visit would erase the first's trace — the bug is invisible until the
  loop exists, so the reducer goes in before the loop does.
- **`retrieved_docs` holds `rag.retrieve.Result` objects, not dicts.** Phase 3 made `Result`
  a frozen dataclass for a stated reason: "a typo in a key should be an `AttributeError` at
  the call site, not an empty citation inside an answer." Re-wrapping it in the agent layer
  would throw that away and duplicate the schema. `Result` already carries everything
  `cite_sources` needs (`title`, `year`, `page_start/end`, `chunk_id`, `section`) and already
  has `.citation()`.
- **`total=False` everywhere, and nodes return partial dicts.** LangGraph merges what a node
  returns into the state; a node that returns the whole state is a node that can silently
  clobber a key it never thought about.

## The engine client (`agent/engine.py`)

The skill is explicit that the agent reasons with Atlas's own engine **via the Phase 2
server**, not by importing `server.bridge`. So the client is HTTP:

```python
class EngineClient(Protocol):
    def complete(self, prompt: str, max_new_tokens: int) -> str: ...

class HttpEngineClient:
    def __init__(self, base_url: str = ..., timeout: float = 300.0) -> None: ...
```

- **Base URL from `ATLAS_AGENT_ENGINE_URL`**, default `http://127.0.0.1:8000`, same
  `ATLAS_`-prefixed-env posture as `server/config.py`.
- **`stream=False`.** The agent needs the finished text before the next node runs; SSE
  parsing buys nothing here. Streaming to a *user* is a Phase 5 concern, and the endpoint
  already supports it when someone wants it.
- **The client reads the server's cap at construction** from `/health`
  (`max_new_tokens`, default 64) and clamps its requests to it. The server returns 422 for
  an over-cap request — correct behaviour, but a graph that dies on a config mismatch it
  could have read is a graph that wastes a lab session.
- **A 503 from `/generate` means the engine is still loading**, and is retried once after a
  short pause; anything else is raised as `AgentError` and lands in `state["error"]`.
- **The timeout is 300 s and that is not paranoia** — see the cost section below.
- **`FakeEngineClient`** returns a canned completion and records the prompts it was given.
  Every graph test uses it. This is the same injection discipline as `rag/retrieve.py`'s
  `embedder=` / `store=` parameters, and for the same reason: the orchestration logic must be
  testable without a GPU, a server, or the corpus.

## The prompt (`agent/prompts.py`)

Two facts collide here, and Step 1 exists partly to write down what falls out.

**Fact one: the server applies no chat template.** `GenerateRequest.prompt` is documented as
"Raw prompt text; no chat template is applied." The weights are
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, which was tuned on the Zephyr format. So the agent owns
the template, and it must be the real one:

```
<|system|>
You answer questions using only the excerpts provided. If they do not contain
the answer, say so.</s>
<|user|>
[1] {title} ({year}), {section}, pp{a}-{b}
{chunk text}

[2] ...

Question: {query}</s>
<|assistant|>
```

The bracketed markers are what Step 3's `cite_sources` will resolve back to `chunk_id`s.
They cost a handful of tokens now and save the whole "which chunk did that sentence come
from" problem later.

**Fact two: there is no KV cache, and the window is 2048.** `max_position = 2048`
(`engine/include/model.h`), and docs/14 measured ~49 ms/token at sequence length ~38 with the
cost growing quadratically — the KV cache is named there as the outstanding fix. Chunks are
packed to `TARGET_TOKENS = 200` (`rag/chunk.py`), so the default `k=5` puts ~1000 tokens of
evidence in front of a model that re-runs the full forward pass for every token it emits.

That is the central engineering fact of this step: **prompt length is the dominant cost of a
run, and nothing upstream of the agent is protecting it.** So the agent budgets:

```
PROMPT_TOKEN_BUDGET = 1536      # leaves headroom under 2048 for max_new_tokens + template
```

Chunks are added in MMR rank order until the next one would breach the budget; the rest are
dropped, and the drop is recorded in `reasoning_steps`. Dropping the *lowest-ranked* evidence
is the only defensible truncation — cutting a chunk mid-sentence produces exactly the kind of
mangled context that makes a 1.1B model hallucinate.

### Counting the tokens honestly

The budget is worthless if the count is a guess. Phase 3 Step 3 already paid for this lesson:
a word-count heuristic *failed* its truncation check (334/3489 chunks over MiniLM's real
window) and the corpus had to be re-chunked with the real tokenizer. Repeating the heuristic
here would be repeating a known mistake.

The tokenizer that matters lives behind the server (`Engine.encode`, `server/bridge.py`), so
Step 1 adds one small endpoint to Phase 2:

```
POST /tokenize  {"text": "..."}  →  {"n_tokens": 1274}
```

~15 lines over an already-loaded tokenizer, no new state, no new dependency. The agent counts
by asking the thing that will actually do the encoding. When the server is unreachable —
`--dry-run` on a laptop — the client falls back to a deliberately **conservative**
chars/3.2 estimate and the state records that the count was estimated, so a budget decision
is never silently based on a guess.

## Nodes (`agent/nodes.py`)

**`retrieve(state) -> dict`** — calls `rag.retrieve.retrieve(Paths.default(), query, k=k)`
and nothing else. Returns `retrieved_docs`, `retrieval_attempts`, and one
`reasoning_steps` line naming what came back (count, papers, top score). Empty result is not
an exception: it returns an empty list and lets Step 2's edge decide what that means.

**`synthesize(state) -> dict`** — budgets the docs, renders the template, calls
`EngineClient.complete`, returns `output`, `prompt`, `used_docs`, and a trace line. With
`retrieved_docs` empty it short-circuits to a fixed "no relevant passages in the corpus"
answer **without calling the engine** — asking an ungrounded 1.1B model an open question is
how a RAG system starts confidently inventing papers.

## The graph (`agent/graph.py`)

```python
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("synthesize", synthesize)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "synthesize")
builder.add_edge("synthesize", END)
graph = builder.compile()
```

`build_graph(engine=None, retriever=None)` takes its dependencies as arguments and closes
over them, so tests compile a real graph with fakes. The topology stays in this file and
nothing else; the moment a node starts deciding what runs next, the separation the skill
demands is gone.

## Validation

`agent/tests/`, pytest, no GPU, no server, no corpus:

- **State contract.** A node returning a partial dict merges; `reasoning_steps` from two
  successive nodes **concatenate** rather than replace (the reducer, tested before the loop
  that needs it exists).
- **Graph wiring.** `build_graph` with a fake retriever and `FakeEngineClient` runs
  `retrieve → synthesize` and produces a populated `output`; the node visit order is asserted
  from the trace.
- **Prompt shape.** The prompt handed to the fake contains the Zephyr markers, every budgeted
  chunk's text, numbered `[n]` source lines, and the query — asserted on the recorded prompt,
  not on the answer.
- **Budget.** 12 synthetic 200-token chunks against a 1536-token budget yields a prompt under
  budget, keeps a rank-ordered prefix, drops the tail, and records the drop.
- **Empty retrieval.** Zero docs ⇒ the fixed answer and **zero** engine calls.
- **Engine client.** Against a stub HTTP server (httpx/ASGI, already a Phase 2 test dep):
  clamping to `/health`'s cap, one retry on 503, `AgentError` on 500, and the error landing in
  `state["error"]` instead of raising out of the graph.
- **`/tokenize`.** Added to `server/tests/`: round-trips a known string to the same count
  `Engine.encode` gives.

Then, on the lab box with the server up — the numbers that go in *Results*:

- One real question, end to end, answer pasted in verbatim, right or wrong.
- **Prompt tokens, TTFT-equivalent, and total wall time at k=1, 3, 5.** This is the
  measurement the step is really for: it turns "no KV cache" from a note in docs/14 into the
  number that sets the default `k`. If k=5 costs more than ~30 s a run, the default drops and
  the reason is recorded here rather than rediscovered in Step 4.

## Files created / touched

```
agent/__init__.py
agent/state.py                   AgentState TypedDict + reducer
agent/engine.py                  EngineClient protocol, HttpEngineClient, FakeEngineClient
agent/prompts.py                 Zephyr template, source-marker rendering, token budget
agent/nodes.py                   retrieve, synthesize
agent/graph.py                   build_graph
agent/cli.py                     python -m agent.cli, --k/--max-new-tokens/--json/--dry-run
agent/tests/test_state.py
agent/tests/test_engine.py
agent/tests/test_nodes.py
agent/tests/test_graph.py
server/serve.py                  + POST /tokenize
server/tests/test_serve.py       + its test
requirements.txt                 uncomment langchain, langgraph
docs/19-agent-bringup.md         this file
```

`agent/prompts.py` is the one file the skill's table does not list. Prompt text is data,
it changes far more often than node logic, and Steps 2–4 each add templates of their own —
keeping it out of `nodes.py` is the same separation argument the skill makes everywhere else.

## Done when

- `pytest agent/tests` green on a CPU-only checkout **with no ML deps and no server** — the
  Phase 3 rule, unchanged.
- `python -m agent.cli "..."` against the live server returns a grounded answer, and
  `--json` emits the full final state.
- `--dry-run` works with the server down.
- Prompt-token and wall-time numbers for k=1/3/5 are pasted into *Results*, and the default
  `k` is justified by them.
- Phase 1, 2 and 3 suites stay green (10/10 CPU CTest, 7/7 CUDA CTest, `pytest server/tests`
  now +1, `pytest rag/tests` 123). Phase 4 adds a directory and one endpoint.

## Design decisions

- **HTTP to the server, not `import server.bridge`.** The in-process import is faster and
  tempting. It also means the agent can only run where a 4.4 GB model and a built `.so` are,
  which breaks the CPU-only test rule and quietly makes Phase 5's MCP server the *third*
  thing that has to load the engine. One process owns the weights.
- **The full state schema now, populated incrementally.** Cheaper than four contract
  revisions.
- **The reducer on `reasoning_steps` before the loop exists.** It is the correct semantics
  either way, and adding it later means debugging a vanished trace.
- **`Result` objects in state, not dicts.** Phase 3's argument, unchanged.
- **`/tokenize` on the server rather than a client-side heuristic.** Docs/17 measured what
  heuristic token counting costs. The tokenizer is already loaded; ask it.
- **Budget by dropping whole low-ranked chunks.** Truncating text mid-chunk hands a small
  model a mangled sentence and no signal that anything was removed.
- **No engine call when retrieval is empty.** A RAG system that answers from parametric
  memory when the corpus has nothing is worse than one that says "not in the corpus" — and
  TinyLlama-1.1B is exactly the size where that failure is fluent.
- **Greedy only.** Determinism is what will make Step 2's relevance threshold and any later
  agent eval mean anything.

## Next (Step 2)

`evaluate_relevance` + the conditional edge: judge whether `retrieved_docs` can answer
`query`, and loop back to `retrieve` with a widened `k` or a relaxed filter when they can't.
Docs/18 already supplies the empirical prior — ≈0.67 for a well-answered question, 0.42–0.50
for the near-miss probes, and the explicit warning that the gap is "usable but not wide", so
a top score under ~0.5 is weak evidence rather than a hard threshold. The 36-query eval set
is the obvious place to measure whether the loop helps or just doubles the latency.
