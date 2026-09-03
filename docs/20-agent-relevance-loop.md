# Phase 4 · Step 2 — `evaluate_relevance`, the conditional edge, and knowing when to shut up

> Status: **designed 2026-09-03** — not yet implemented.
> Predecessor: Phase 4 Step 1 — `agent/` bring-up — **done**
> ([19-agent-bringup.md](19-agent-bringup.md)). `retrieve → synthesize`, 34 tests, defaults
> `k=3` / `max_new_tokens=32` at 29 s a run on the A6000.
> Successor: Step 3 — `cite_sources`.

## Goal

Put a decision between retrieval and generation. Today every question reaches the engine and
pays 20–80 s for an answer, whether or not the corpus contains one. This step gives the graph
a third option — **say the corpus does not cover this, and don't generate** — and the
conditional edge that chooses between the three.

```
retrieve → evaluate_relevance ─┬─(sufficient)──→ synthesize → END
                               ├─(retry)───────→ retrieve  (once)
                               └─(insufficient)→ abstain   → END
```

The conditional edge is the reason LangGraph is in this project at all, and Step 1 built the
`reasoning_steps` reducer specifically so a node could run twice without erasing its own
trace. This is the step that uses both.

It is also the step where a number gets chosen from data rather than taste. Docs/18 measured
what retrieval scores mean on this corpus — well-answered questions top out ≈0.67, the
near-miss probes land 0.42–0.50 — and then explicitly warned that the gap is "usable but not
wide" and that a hard cut should not be trusted on the strength of it. Two probes is not a
distribution. Fixing that is part of this step.

### Scope boundary (what Step 2 is *not*)

- **Not `cite_sources`.** The abstain path returns the closest passages it found, so the
  evidence is present; rendering and verifying citations is Step 3.
- **Not `tools.py`.** Filters (`paper_id`, `topic_label`, `section`) arrive with the tools in
  Step 4. This matters more than it sounds — see *The retry has almost no lever* below.
- **Not an LLM judge.** Asking TinyLlama whether excerpts answer a question costs a full
  generation. Step 1 measured what that is: 20–80 s, on a 1.1B model that had already been
  caught inventing an author list. Paying the most expensive operation in the system to
  decide whether to perform the most expensive operation in the system needs an argument
  nobody has made yet.
- **Not query rewriting.** The one retry lever that could genuinely change the evidence, and
  it needs either an engine call (see above) or a scoring trap explained below.
- **Not a confidence score in the answer text.** The state records the number; what the user
  sees is Step 3's problem.
- **Not re-tuning retrieval.** λ, `fetch_k` and the per-paper cap were fitted in docs/18 and
  are not reopened here.

## The operation

```
python -m agent.cli "how does flash attention avoid materializing the attention matrix"
python -m agent.cli "how does CRISPR gene editing work"        # expected: abstain, ~7 s
python -m agent.cli --relevance-floor 0.6 "..."                # override the fitted τ
python -m agent.cli --no-gate "..."                            # Step 1's behaviour, exactly
python -m agent.eval --sweep                                   # fit τ on rag/eval/queries.json
```

`--no-gate` is not a courtesy flag. It is how a regression in the gate gets separated from a
regression in retrieval or synthesis, and it is the control arm of the sweep.

`python -m agent.eval` needs the corpus and MiniLM but **never the engine**: the whole sweep
is 36 encodes and 36 HNSW queries, seconds, no generation. That is the point of choosing a
signal retrieval already computes.

## The fact that reshapes this step

Before designing a retry, measure whether a retry can do anything. It mostly cannot, and the
reason is structural.

`rag.retrieve.retrieve` pulls `fetch_k=40` candidates and runs MMR over them. MMR's *first*
selection is pure relevance whatever λ is — `redundancy` is `-inf` until something has been
picked — and `fetch_k` does not depend on `k`. So rank 0 is the argmax of the same 40
candidates no matter what `k` is. Docs/18 noticed the λ half of this ("recall@1 is identical
at every λ — MMR cannot change the first pick"); the `k` half is the one that matters here.

Measured, 2026-09-03, `--dry-run` on the live corpus, one query, four values of `k`:

| k | rank-0 chunk | top1 score | mean@k |
|---|---|---|---|
| 1 | `2205.14135:60` | 0.6579 | 0.6579 |
| 3 | `2205.14135:60` | 0.6579 | 0.6529 |
| 8 | `2205.14135:60` | 0.6579 | 0.5809 |
| 20 | `2205.14135:60` | 0.6579 | 0.5690 |

Bit-identical top1 across a 20× range of `k`, and mean@k **falling monotonically** because a
wider `k` can only reach further down the same ranked list.

The obvious retry — "loop back and retrieve more" — is therefore provably useless on the
first signal and actively counterproductive on the second. A graph that widens `k` on a weak
result buys nothing but a longer prompt, and Step 1 measured that a longer prompt is the
dominant cost of a run: 529 → 834 prompt tokens took 51 s → 80 s. **Widening `k` on a retry
makes the run slower and the answer no better.**

### Then what *can* a retry change?

| lever | can it move top1? | cost | verdict |
|---|---|---|---|
| widen `k` | **no** (measured above) | +prompt tokens, slower synthesis | rejected |
| drop the per-paper cap | no — the cap never constrains pick 0 | free | rejected |
| widen `fetch_k` 40 → 200 | only if HNSW's approximation missed a true neighbour | one extra HNSW call, ~ms | **the one available lever** |
| relax a filter | yes — a different candidate pool, same query vector | free | **Step 4**; no filters exist yet |
| rewrite/expand the query | yes | see the trap below | rejected here |

Query expansion deserves its own paragraph, because it looks like the obvious answer and it
hides a scoring trap. Expanding the query with terms taken from the top-ranked chunk
(pseudo-relevance feedback) produces a *different query vector*, and cosine similarities to
two different query vectors are not comparable numbers. Worse, appending a chunk's own
vocabulary mechanically raises similarity **to that chunk** — so the gate's number would go
up while retrieval got no better, and the system would talk itself into answering exactly the
questions it should have refused. Any query-changing retry needs the gate to re-score against
the *original* query vector, which needs chunk vectors that `Result` does not carry. That is
a real design, but it is not a free one, and it should not be bolted on next to a threshold
that is itself being fitted for the first time.

So the retry branch ships with `fetch_k` widening as its only lever, and with an honest
prediction attached: **it will almost never change anything.** Which brings us to the reason
it is built anyway.

### A pre-registered kill criterion

The retry branch is an experiment, and it is written down before the measurement so the
result cannot be rationalised afterwards:

> If, over the eval set, widening `fetch_k` to 200 changes the rank-0 chunk for **fewer than
> 10%** of below-floor queries, the retry branch is **deleted** in this step — the router
> becomes two-way, `MAX_RETRIEVAL_ATTEMPTS` goes away, and the loop-back edge returns in Step
> 4 with filter relaxation, which is the lever that actually works.

Docs/18 set this precedent: λ = 0.6 was proposed in a design doc, measured, found wrong, and
changed. A decorative loop kept because the design promised one would be worse than no loop
at all — it doubles the code paths Step 3 has to reason about in exchange for nothing.

## The signal (`agent/relevance.py`)

```python
RELEVANCE_FLOOR = 0.55          # PROPOSED — fitted by the sweep below before it ships
MAX_RETRIEVAL_ATTEMPTS = 2
RETRY_FETCH_K = 200             # the one lever; DEFAULT_FETCH_K is 40

def relevance(docs: Sequence[Result]) -> float | None:
    """Top-1 cosine similarity to the query — None when nothing was retrieved."""
    return float(docs[0].score) if docs else None
```

**The signal is top1, and the argument for it is the table above: it is the only candidate
that does not move when the user changes `k`.** `mean@k` drops from 0.658 to 0.569 as `k` goes
1 → 20 on an unchanged query against an unchanged corpus; a gate built on it would abstain
more readily at `k=8` than at `k=3` for reasons that have nothing to do with the question. A
policy whose behaviour depends on a breadth knob is not a policy.

mean@k is still computed and written into `reasoning_steps` — it is a useful diagnostic when
reading a bad run, it is just not allowed to decide anything.

The comparison is `>=`, not `>`: a score exactly at the floor answers. This is arbitrary and
therefore has to be written down and tested, because the alternative is discovering it from a
flaky boundary test in Step 3.

## Fitting the floor

### First, the eval set has to be able to answer the question

`rag/eval/queries.json` is 36 hand-built queries: 34 with expected papers, and **2 near-miss
probes** — in-domain questions no paper in the corpus answers. The file's own note says the
probes exist "to show whether a low top score is a usable 'I don't know' signal for Phase 4".
That is exactly this step, and two of them is not enough to place a threshold: with n=2 a
single unlucky probe moves the measured abstain rate by 50 points.

So Step 2 raises the probe count to **12**, keeping the file's stated discipline — hand-built,
judged by reading, phrased the way someone who has not read the papers would ask, frozen once
measured — and splitting them into two kinds, because they fail differently:

- **near-miss** (~8): in-domain, plausibly *about* the corpus, answered by none of its 43
  papers. The existing two are this kind. These are the hard cases and the ones the floor is
  really fitted against.
- **out-of-domain** (~4): questions from another field entirely. These should be trivially
  below any floor, and if they are not, the signal is broken and no threshold will save it.

`version` goes to 2. The 34 scored queries are untouched, and probes are already scored
separately, so **docs/18's recall@5 = 0.94 is unaffected and is not re-measured here.**

### The sweep

`python -m agent.eval --sweep` retrieves once per query at the shipping `k=3` and reports, per
candidate floor:

| τ | answered (of 34) | **false abstain** | probes abstained (of 12) | near-miss | out-of-domain |
|---|---|---|---|---|---|
| 0.45 | | | | | |
| 0.50 | | | | | |
| 0.55 | | | | | |
| 0.60 | | | | | |
| 0.65 | | | | | |

Plus the two things a single table hides — the **distributions**: min / median / max of top1
over the 34 answerable queries, and over the 12 probes. If the lowest answerable top1 sits
below the highest probe top1, no threshold separates them cleanly, and the honest output of
this step is the size of that overlap rather than a number that conceals it.

**The selection rule, fixed in advance:** choose the largest τ whose **false-abstain rate over
the 34 answerable queries is ≤ 5%** (i.e. at most one), then report whatever probe-abstain
rate falls out. The asymmetry is deliberate. Refusing a question the corpus *does* answer is
the failure a user experiences as "this tool is broken"; hedging on one it does not is a
mild, correct annoyance — and TinyLlama has already been instructed to say when the excerpts
do not contain the answer, so the un-gated path is not silent either.

If no τ clears both usefully, the finding is that top1 is not a usable gate on this corpus,
and Step 2 ships with the gate **off by default** (`--gate` opt-in) and that sentence in
*Results*. That is a legitimate outcome and it is cheaper to admit here than to discover in
Phase 5 when the MCP server is returning confident refusals.

## State (`agent/state.py`)

`relevance: float | None` and `needs_retry: bool` were declared in Step 1 and are populated
now, as designed. One key is genuinely new:

```python
    abstained: bool          # the gate refused; `output` is the fixed answer, not generated
```

Step 1 claimed the full Phase 4 schema was written down up front to avoid four contract
revisions. It cost one key across the first two steps, which is roughly the intended hit
rate. `abstained` earns its place by being the thing Step 3 branches on and the thing an eval
counts — reconstructing it from `output == ABSTAIN_ANSWER` would be string-matching a
constant, which is the kind of thing that survives until someone edits the string.

Nothing else changes. `retrieved_docs` stays last-write-wins: a retry's wider fetch supersedes
the first attempt rather than accumulating, which is what makes "the second attempt is the
answer" true without a merge rule. `reasoning_steps` accumulates across both visits, which is
what the reducer was for.

## Nodes (`agent/nodes.py`)

**`evaluate_relevance(state) -> dict`** — computes `relevance` from `retrieved_docs`, compares
it to the floor, sets `needs_retry`, and appends a trace line carrying top1, mean@k and the
verdict. It makes **no** engine call and **no** store call; it reads numbers retrieval already
produced. Empty `retrieved_docs` gives `relevance = None` and `needs_retry = False` — with no
filters in play, an empty result means an empty or missing store, and no retry fixes that.

**`abstain(state) -> dict`** — returns the fixed `ABSTAIN_ANSWER`, sets `abstained=True`,
carries `retrieved_docs` through as `used_docs` so the user (and Step 3) can see what the
closest passages were, and **does not call the engine**. It is a sibling of the empty-evidence
short-circuit Step 1 put inside `synthesize`, and it is a separate node rather than another
branch inside `synthesize` for the reason Step 1 gave for `DryRunEngineClient`: a node with
two exits is a node whose tests stop telling you which path ran.

**`retrieve`** gains one behaviour: on attempt 2 it passes `fetch_k=RETRY_FETCH_K`. That is
the whole retry. It reads `retrieval_attempts` — already in the state since Step 1, already
incremented correctly, already tested.

**`synthesize`** is untouched. Worth stating plainly: the gate changes what reaches
synthesis, not how synthesis works, so Step 1's prompt, budget and 34 tests stand.

## The graph (`agent/graph.py`)

```python
builder.add_node("retrieve", make_retrieve(retriever))
builder.add_node("evaluate_relevance", make_evaluate_relevance(floor))
builder.add_node("synthesize", make_synthesize(engine))
builder.add_node("abstain", make_abstain())
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "evaluate_relevance")
builder.add_conditional_edges(
    "evaluate_relevance",
    route_after_relevance,
    {"retrieve": "retrieve", "synthesize": "synthesize", "abstain": "abstain"},
)
builder.add_edge("synthesize", END)
builder.add_edge("abstain", END)
```

The router is a pure function of the state and it lives **here**, in the topology file, not in
`nodes.py`. The node decides the number and the boolean; the graph decides where a boolean
goes. Step 1's rule — "the moment a node starts deciding what runs next, the separation the
skill demands is gone" — is exactly what a conditional edge is for.

```python
def route_after_relevance(state: AgentState) -> str:
    if not state.get("needs_retry"):
        return "synthesize" if state.get("retrieved_docs") else "abstain"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "abstain"
    return "retrieve"
```

Three guards, and the loop terminates on the third whatever the retriever does. The graph is
also compiled with LangGraph's `recursion_limit` left at its default as a backstop — but the
attempt counter is the actual guarantee, because a limit that fires is a crash and a counter
that fires is an answer.

`build_graph(engine=None, retriever=None, floor=RELEVANCE_FLOOR, gate=True)`. With
`gate=False` the graph is Step 1's exactly — `retrieve → synthesize → END`, no
`evaluate_relevance` node registered — so `--no-gate` is a genuine control arm and not a
gate that always says yes.

## Validation

`agent/tests/`, pytest, no GPU, no server, no corpus — the rule since Phase 3, unchanged.

- **Signal.** `relevance` is top1; identical for a retriever returning 1, 3 and 8 docs whose
  rank-0 is the same chunk (the k-stability the whole choice rests on). `None` for empty.
- **Floor boundary.** A score exactly equal to the floor answers; one float below it does not.
- **Routing table.** `route_after_relevance` is a pure function and gets a parametrized test
  over the whole cross-product: above/below floor × attempts 1/2 × docs present/empty. This
  is the cheapest test in the step and the one most likely to catch a Step 4 regression.
- **The loop actually loops.** A fake retriever that always returns low scores runs `retrieve`
  **exactly twice**, ends in `abstain`, and leaves **two** `retrieve(...)` lines in
  `reasoning_steps` — the reducer Step 1 built for a loop that did not exist yet, now tested
  against the loop.
- **The retry widens the fetch.** The fake retriever records its `fetch_k`; attempt 1 gets 40,
  attempt 2 gets 200.
- **Abstain never generates.** `FakeEngineClient.prompts == []` on an abstained run, and
  `used_docs` still carries the closest passages.
- **`gate=False` reproduces Step 1.** Same output, same prompt, same trace shape, on the same
  fixtures — asserted against Step 1's own test expectations rather than a copy of them.
- **Sufficient evidence is untouched.** An above-floor run produces the identical prompt the
  ungated graph produces. The gate must be a router, not a filter with side effects.

Then, with the corpus (no engine needed):

- `python -m agent.eval --sweep` — the table above, filled in, plus the two distributions.
- The `fetch_k` 40 → 200 experiment, and the kill-criterion verdict, recorded either way.

Then, on the lab box with the server up — the numbers for *Results*:

- **One real abstain, end to end**, on an out-of-domain question, with its wall time. The
  claim this step is making is ~29 s → ~7 s (retrieval only, no generation), and it should be
  measured rather than asserted.
- **The second retrieval's cost** in a warm process, against the 20–80 s generation it is
  deciding about. The prediction is that it is negligible — one encode plus one HNSW query,
  with MiniLM already loaded — and that is the reason a retry-then-abstain path is affordable
  at all. If it is not negligible, the retry goes even if it passes the kill criterion.

## Files created / touched

```
agent/relevance.py                 signal, floor, MAX_RETRIEVAL_ATTEMPTS, RETRY_FETCH_K
agent/state.py                     + abstained
agent/nodes.py                     + evaluate_relevance, + abstain; retrieve widens on retry
agent/graph.py                     + the two nodes, the conditional edges, route_after_relevance
agent/cli.py                       + --relevance-floor, --no-gate
agent/eval.py                      python -m agent.eval --sweep, over rag/eval/queries.json
agent/tests/test_relevance.py      signal, floor boundary, routing table
agent/tests/test_nodes.py          + evaluate_relevance, + abstain
agent/tests/test_graph.py          + routing, the loop guard, gate=False
rag/eval/queries.json              version 2: 2 probes -> 12, scored queries untouched
requirements.txt                   unchanged — no new dependency
docs/20-agent-relevance-loop.md    this file
```

`agent/relevance.py` is a new file rather than constants in `nodes.py` for the reason
`prompts.py` is one: a fitted threshold is *policy data*, it will be re-fitted when the corpus
grows, and the diff that changes it should not be a diff that touches node logic.

## Done when

- `pytest agent/tests` green on a CPU-only checkout with no ML deps and no server.
- The sweep table and both top1 distributions are pasted into *Results*, and `RELEVANCE_FLOOR`
  is the number the stated selection rule picked — or the gate ships off by default and the
  overlap is written down.
- The `fetch_k` experiment is recorded and the kill criterion is honoured, in whichever
  direction it points.
- `python -m agent.cli "how does CRISPR gene editing work"` abstains, names its closest
  passages, makes zero `/generate` calls, and its wall time is in *Results*.
- `--no-gate` reproduces Step 1's output on a question the corpus answers.
- Phases 1–3 stay green: 10/10 CPU CTest, 17/17 CUDA CTest, `pytest server/tests` 29,
  `pytest rag/tests` 123. Step 2 adds no dependency and touches one Phase 3 data file.

## Design decisions

- **Score-based gate, not an LLM judge.** The judge costs the thing it is deciding whether to
  spend, on the model that was caught inventing authors at k=5. The score is already computed,
  free, deterministic, and — unlike a 1.1B judgement — has a measured distribution behind it
  from docs/18.
- **top1, not mean@k.** The only candidate signal that does not move when the user turns `k`.
  Measured: mean@k falls 0.658 → 0.569 over k = 1 → 20 on an unchanged query.
- **The floor is fitted, with the rule fixed first.** Docs/18's λ = 0.6 was a design-doc guess
  that measurement overturned; a threshold guessed here would be the same mistake with a worse
  failure mode. The selection rule is written before the numbers exist so it cannot be chosen
  to flatter them.
- **Bound false abstains first.** Wrongly refusing a covered question breaks the tool.
  Wrongly answering an uncovered one is a hedge the system prompt already asks for.
- **12 probes, not 2.** A threshold placed on two points is a threshold placed on noise.
- **The retry ships with a kill criterion, not a promise.** Widening `k` is measurably
  useless; widening `fetch_k` is the only lever available before Step 4's filters, and it is
  predicted to do nothing. Building it, measuring it and deleting it is cheaper than carrying
  a decorative branch through Steps 3 and 4.
- **No query expansion.** It changes the query vector, which makes the gate's number
  incomparable across attempts and biased upward toward whatever chunk supplied the terms —
  the system would talk itself into answering the questions it should refuse.
- **`abstain` is its own node.** A second exit inside `synthesize` would make "did it
  generate?" unanswerable from the graph's shape, which is the one question this whole step
  exists to answer.
- **`gate=False` removes the node rather than short-circuiting it.** A control arm that still
  runs the thing under test is not a control arm.

## Next (Step 3)

`cite_sources`: resolve the `[n]` markers Step 1 put in the prompt back to `chunk_id`s, render
citations from the `Result` metadata already in `used_docs`, and fail an answer that cites a
source it was not given. The abstain path lands there too — "not in the corpus, and here is
the closest thing I found" is a citation-shaped output — and `abstained` is the flag it
branches on.
