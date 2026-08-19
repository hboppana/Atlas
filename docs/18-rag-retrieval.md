# Phase 3 · Step 4 — MMR retrieval + a retrieval eval set

> Status: **done** — designed and implemented 2026-08-19. See *Results* below.
> Predecessor: Step 3 — local embeddings + ChromaDB store + derived topics — **done**
> ([17-rag-embeddings-store.md](17-rag-embeddings-store.md)). 43/43 papers `embedded`,
> 4148 chunks in `atlas_chunks`, k=2 derived topics.
> Successor: Phase 4 — the LangGraph agent, which imports this module and nothing else
> from `rag/`.

## Goal

Turn the store into a **retriever with a number attached to it**: one `retrieve()` entry
point that takes a question and returns a small, diverse, citation-ready set of chunks —
top-k by cosine, then **MMR** for diversity, with optional metadata filters — plus a
hand-built eval set that measures recall@k so the number is real and regressions are
visible.

Step 3 made the corpus searchable. Step 4 decides *what comes back*, and is the last piece
of Phase 3: after it, Phase 4's agent calls `rag.retrieve.retrieve()` and never touches
Chroma, numpy or the manifest directly.

### Scope boundary (what Step 4 is *not*)

- **Not a re-embed, not a re-chunk.** Retrieval reads the cached `.npz` matrix and the
  Chroma collection. If a chunk is wrong, the fix lives in Step 2 or Step 3.
- **Not an LLM reranker.** Cross-encoder reranking and query rewriting are the obvious next
  quality lever, and they are deliberately out — they change the latency profile and need
  the engine in the loop. MMR is a pure-numpy selection policy over vectors we already have.
  The eval set built here is what would justify adding a reranker later, and measure it.
- **Not the agent's retrieval policy.** How many hops, when to re-query, how to decide a
  result is irrelevant — that is Phase 4's graph (`atlas-agent`). Step 4 hands it one
  deterministic function.
- **Not a server.** No HTTP surface. Phase 4 and Phase 5 import `rag/` directly.
- **Not an answer-quality eval.** The eval set measures *retrieval* (did the right paper
  come back), not generation. End-to-end answer eval belongs to Phase 4.

## The operation

```
scripts/ingest_papers.py --search "how does MMR trade relevance for diversity" -k 5
scripts/ingest_papers.py --search "flash attention tiling" -k 5 --lambda 0.3   # more diverse
scripts/ingest_papers.py --search "kv cache quantization" --topic "quantization, gpu, bit"
scripts/ingest_papers.py --search "..." --paper 2307.08691 --year-min 2023
scripts/ingest_papers.py --eval                                   # recall@k over the eval set
```

`--query` from Step 3 stays exactly as it is — plain top-k cosine, the store smoke path.
`--search` is the new flag and runs the real retriever. Keeping both is deliberate: when a
result looks wrong, the first question is always "is this the store or the selection
policy?", and two flags answer it without a debugger.

## Retrieval

`rag/retrieve.py`. Pure functions over numpy plus one thin call into `ChunkStore`.

```python
DEFAULT_K          = 5      # what the caller gets back
DEFAULT_FETCH_K    = 40     # candidates pulled before MMR selects from them
DEFAULT_LAMBDA     = 0.7    # 1.0 = pure relevance, 0.0 = pure diversity; measured, see Results
MAX_PER_PAPER      = 2      # cap on chunks from any one paper (0 = no cap)
```

Two stages, and the split matters:

1. **Candidate fetch** — embed the query with the *same* `MiniLMEmbedder` the corpus used,
   then `ChunkStore.query(vector, k=fetch_k, where=...)`. Filters are applied here, in
   Chroma, not after selection: filtering post-MMR would silently return fewer than `k`.
2. **Selection** — MMR over the candidate vectors, in numpy, to pick `k` of the `fetch_k`.

`fetch_k` is the only knob that costs anything, and 40 is chosen so MMR has room to be
diverse (8× the default k) while staying one HNSW call. If `fetch_k <= k` MMR is a no-op and
the code says so rather than pretending to diversify.

### MMR

Standard Carbonell & Goldstein formulation, over unit vectors so every similarity is a dot
product:

```
score(c) = λ · sim(q, c) − (1 − λ) · max sim(c, s)   for s already selected
```

Greedy: seed with the highest-relevance candidate, then repeat `argmax` until `k` are
selected. `O(fetch_k · k · dim)` — with 40 candidates that is microseconds, so there is no
reason to approximate it.

Three properties worth fixing in the tests:

- **λ = 1.0 must reproduce plain top-k exactly.** That is the assertion that proves the MMR
  path is not quietly reordering things when diversity is switched off.
- **λ = 0.0 selects near-orthogonal chunks** and, on a corpus with duplicate passages,
  never returns two of them.
- **Ties break by candidate rank**, not by numpy's argmax over float noise, so a repeated
  run of the same query returns the same list. A retriever that is not deterministic cannot
  be evaluated.

### Where the vectors come from

Candidate vectors are needed for the chunk–chunk similarities, and Chroma's `query` does not
return them by default. Two options, and the choice is the reason the `.npz` cache exists:

- `include=["embeddings"]` on the Chroma query — simplest, one call, and the vectors are
  guaranteed to be the rows actually stored.
- Load the cached matrix and look up candidates by `chunk_id`.

**Take the Chroma `include` path**, and keep the cache for offline work (the eval harness,
future clustering) rather than the hot path. Reading vectors from a different source than
the one being searched is exactly the kind of skew Step 3's `ExplicitVectorsOnly` exists to
prevent, and one call beats two plus an index.

### Filters

Translated to a Chroma `where` clause, all optional and combinable with `$and`:

| Argument | Clause | Note |
|---|---|---|
| `paper_id` | `{"paper_id": {"$in": [...]}}` | scope to specific papers |
| `topic_label` | `{"topic_label": ...}` | resolved from `topics.json` at call time |
| `year_min` / `year_max` | `{"year": {"$gte": ...}}` | `year` is an int in metadata |
| `section` | `{"section": ...}` | e.g. only `Method` sections |

`topic_id` is **not** exposed as a filter argument. Step 3 recorded that the integer is
unstable across corpus growth; accepting it here would let Phase 4 memorize one in a prompt
and rot invisibly. Labels are resolved through `topics.json` on every call, and an unknown
label is an error naming the available ones, never a silent empty result.

### Per-paper cap

MMR diversifies in vector space, which is *not* the same as diversifying across papers — a
paper with 200 chunks can still take 4 of 5 slots with genuinely different passages.
`MAX_PER_PAPER = 2` is applied inside the greedy loop (a candidate whose paper is full is
skipped), not as a post-filter, so the selection still fills `k`. And the cap is a
**preference, not a quota**: if every remaining candidate belongs to a paper that has filled
it — which happens on a narrow question the corpus answers in two papers — the cap is
dropped for the remaining slots rather than returning fewer than `k`. Under-filling `k` is
the exact failure post-filtering causes, and a cap that quietly did it would be no better. It is a separate knob from
λ because they solve different problems, and Phase 4's "cite sources" reads much better with
three papers than with one.

## Result shape

```python
@dataclass(frozen=True)
class Result:
    chunk_id: str          # "<paper_id>:<index>" — what cite_sources resolves
    text: str              # the chunk, verbatim, for quoting
    score: float           # cosine similarity to the query, NOT the MMR score
    rank: int              # 0-based position after selection
    paper_id: str
    title: str
    year: int | None
    section: str
    page_start: int
    page_end: int
    topic_label: str
```

`score` is deliberately the raw relevance, not the MMR objective: the MMR score is relative
to whatever else got selected and is meaningless to a caller (or to a threshold in Phase 4's
relevance-evaluation node). A frozen dataclass rather than the raw dict Step 3 returns —
Phase 4 formats citations off these fields, and a typo in a dict key should be an
`AttributeError` at the call site, not an empty citation in an answer.

## Eval set

`rag/eval/queries.json` — hand-built, versioned in git, **the deliverable of this step**.
36 entries over the 43-paper corpus (34 scored + 2 near-miss probes):

```jsonc
{
  "query_id": "mmr-tradeoff",
  "query": "how does MMR trade off relevance against diversity",
  "expected_papers": ["<arxiv_id>"],       // 1-3 papers that genuinely answer it
  "note": "phrased as a user would, not as the paper's title"
}
```

Rules for writing them, because a bad eval set is worse than none:

- **Never copy a title or an abstract sentence.** That measures string overlap surviving an
  encoder, which MiniLM will always pass. Queries are phrased the way someone who has *not*
  read the paper would ask.
- **`expected_papers` is judged by reading the paper**, and a query with no clearly correct
  paper is deleted rather than assigned a plausible one.
- A handful of **multi-paper queries** (a topic several papers cover) and a couple of
  **near-miss queries** (in-domain, answered by no paper in the corpus) — the second kind is
  what tells Phase 4 whether a low top score is a usable "I don't know" signal.
- The set is **frozen once measured**. Tuning λ against it and then editing it is how a
  number becomes a lie.

### Metrics

`scripts/ingest_papers.py --eval` prints, over the whole set:

```
recall@1  0.68   recall@3  0.84   recall@5  0.92     (25 queries, 43 papers)
mrr       0.77   distinct papers per query @5  3.1
misses: 3  (mmr-tradeoff, kv-cache-eviction, rope-scaling)
```

- **recall@k** — fraction of queries where ≥1 expected paper appears in the top k. The
  headline number, and the one Phase 4 is built on.
- **MRR** — rank of the first correct paper, which recall@5 alone hides.
- **Distinct papers per query** — the number MMR is supposed to move. Reporting it next to
  recall makes the tradeoff visible instead of theoretical: if diversity is up and recall is
  flat, λ is doing its job.
- **The miss list is printed by `query_id`**, always. An aggregate with no way to look at
  the failures is a number nobody can act on.

Baseline is run twice — λ=1.0 (plain top-k) and the default λ — so MMR's cost in recall is
recorded, not assumed. The expectation is that recall@5 is roughly unchanged and distinct
papers rises; if recall drops materially, λ is wrong and the default changes here.

## Validation

`rag/tests/test_retrieve.py` — pytest, no network, no corpus, no ML deps. Same quarantine as
Step 3: the query embedder is `conftest.StubEmbedder`, and the store is a tmp_path Chroma
collection populated with hand-built vectors (`chromadb` is `importorskip`-guarded; the pure
MMR tests need neither).

- **MMR unit tests.** λ=1.0 equals top-k order exactly; λ=0.0 on three duplicates plus one
  outlier picks the outlier second; selection is deterministic across runs; `k > n_candidates`
  returns everything without raising; `fetch_k <= k` degrades to top-k.
- **Per-paper cap.** A candidate set of 10 chunks from one paper and 2 from another returns
  at most `MAX_PER_PAPER` from the first and still fills `k`; a candidate set drawn entirely
  from one paper relaxes the cap rather than returning fewer than `k`.
- **Filter translation.** Each argument produces the expected `where` clause; combinations
  nest under `$and`; an unknown `topic_label` raises naming the known labels; filters are in
  the *fetch* call, asserted by a store double that records its arguments.
- **Result shape.** `page_start`/`page_end` survive as ints, `score` is the cosine to the
  query and matches the store's, `rank` is 0..k-1, and a missing optional metadata key
  (`year` on a local PDF) yields `None` rather than a `KeyError`.
- **Eval harness.** Runs against a synthetic 3-paper fixture with a known answer key:
  recall@1 = 1.0 when the retriever is fed exact chunk text, and a deliberately wrong answer
  key produces 0.0 — the test that proves the metric is measuring rather than always passing.
- **`queries.json` well-formedness.** Every `expected_papers` entry is a `paper_id` that
  exists in `scripts/arxiv_ids.txt`; `query_id`s are unique. Catches a typo'd ID that would
  otherwise show up as a permanent, mysterious miss.

Plus the **corpus run by hand**: `--eval` over the real store, with the numbers pasted into
*Results* below at both λ settings.

## Files created / touched

```
rag/retrieve.py                  MMR, filters, Result, retrieve()
rag/eval/__init__.py             harness: recall@k, MRR, distinct-papers
rag/eval/queries.json            the hand-built eval set (tracked in git)
rag/tests/test_retrieve.py
scripts/ingest_papers.py         --search, --lambda, --topic, --paper, --year-min/max, --eval
docs/18-rag-retrieval.md         this file
```

No new dependency. MMR is ~30 lines of numpy, and the metrics are arithmetic — the same
argument that kept `scikit-learn` out of Step 3's k-means.

## Done when

- `pytest rag/tests` green (Step 1–3's 93 plus the new ones) on a checkout with no ML deps. ✅ 123
- `--search` returns 5 results in the documented shape, and λ=1.0 reproduces `--query`.
- `--eval` runs the full set and prints recall@1/3/5, MRR, distinct-papers and the miss list,
  at both λ=1.0 and the default, with both pasted into *Results*.
- recall@5 ≥ 0.85 at the default λ, or the shortfall is explained here rather than tuned away. ✅ 0.94
- `queries.json` has ≥20 queries, every `expected_papers` ID in the corpus. ✅ 36 entries
- Phase 1 / Phase 2 suites stay green.

## Design decisions

- **MMR over the fetch set, not over the corpus.** Exact MMR on 4148 chunks would mean a
  4148×4148 similarity pass per query for a selection that only ever picks 5. HNSW top-40
  then exact MMR gets the same answer for a fraction of the work, and the approximation is
  in the *candidate* stage where it belongs.
- **Vectors from Chroma's `include`, cache for offline.** One source of truth on the hot
  path. The `.npz` cache stays what Step 3 built it for: not re-running the model.
- **Filters go to the store, not to the selection.** Post-filtering silently under-fills `k`,
  which looks like a retrieval quality problem and is not.
- **`score` is relevance, never the MMR objective.** Phase 4 will threshold on it.
- **Labels, not `topic_id`, at the API boundary.** Step 3 recorded the instability; this is
  the step that has to honour it.
- **The eval set is hand-built and small.** A synthetic set generated by asking a model to
  write questions about each paper measures whether the generator and the encoder agree —
  which is not the thing under test. 25 honest queries beat 500 circular ones.
- **The eval set is frozen once measured**, and lives in git next to the code it grades.

## Results (2026-08-19)

`pytest rag/tests` — **123 passed** (Steps 1–3's 93, 27 new here, 3 more with the chunker
fix below), CPU, no GPU needed.

All numbers below are on the **v3 corpus** (3656 chunks). The eval was first run against v2
and drove a chunker fix mid-step; both sets of numbers are kept, because the difference
between them is the most useful thing this step produced.

### The λ sweep, and an honest note about it

`--eval` over the real store, 34 scored queries + 2 probes, k=5, per-paper cap 2:

| λ | recall@1 | recall@3 | recall@5 | MRR | distinct papers |
|---|---|---|---|---|---|
| 1.0 (plain top-k) | 0.71 | 0.85 | 0.91 | 0.78 | 3.6 |
| 0.9 | 0.71 | 0.88 | 0.94 | 0.79 | 3.9 |
| 0.8 | 0.71 | 0.85 | 0.94 | 0.80 | 3.8 |
| **0.7 (shipped)** | 0.71 | **0.91** | **0.94** | **0.80** | 3.9 |
| 0.6 (originally proposed) | 0.71 | 0.85 | 0.88 | 0.77 | 4.0 |
| 0.5 | 0.71 | 0.85 | 0.88 | 0.77 | 4.2 |

The design proposed λ = 0.6 as a guess, and it was **wrong**: it costs 6 points of recall@5
and 3 points of MRR to buy 0.4 of a paper. λ = 0.7 dominates plain top-k outright — 3 points
of recall@5, 6 of recall@3, 2 of MRR, and 3.6 → 3.9 distinct papers — so the default changed
here, exactly as the design said it would if the drop was material. (The same ordering held
on the v2 corpus, at 0.91/0.88 instead of 0.94/0.91.)

**The 0.7 is fitted to this eval set, not held out from it.** Nothing was tuned except this
one scalar, and the set was not edited after being measured — but a number chosen on the set
it is scored against is worth less than the table makes it look, and the honest reading is
"0.7 is not worse than top-k on 34 queries", not "0.7 generalises". recall@1 is identical at
every λ, which is the expected sanity check: MMR cannot change the first pick.

### Acknowledgements sections were eating the corpus — two chunker bugs, now fixed

The first eval run (v2 corpus) missed three queries, and two of them were topped by chunks
whose section was **Acknowledgements**, from papers with nothing to do with the question.
`699 / 4148 chunks — 17% of the corpus, 25 papers` carried that title. Rerunning the eval
with those chunks excluded predicted **recall@5 0.91 → 0.94**, so it was worth chasing.

Two independent defects in Step 2, both written up in
[16-rag-chunking.md](16-rag-chunking.md) § *Superseded by chunker v3*:

1. **A contents page hijacks the monotonic-run gate.** PaLM's table of contents carries the
   paper's real section numbers in ascending order, so the run was consumed on page 2 and
   every real header afterwards was rejected — 2510 of 2595 lines in one section.
2. **`REFERENCES_MIN_POSITION = 0.3` rejected three real References headers**, on the papers
   whose appendix is three times their body. The bibliography then landed inside whatever
   section preceded it.

Fixed at `CHUNKER_VERSION = 3` (contents entries dropped ahead of detection, anchored on dot
leaders; floor lowered to 0.05), corpus re-chunked and re-embedded:

| | v2 corpus | v3 corpus |
|---|---|---|
| Chunks | 4148 | 3656 |
| Acknowledgements-titled chunks | 699 (17%) | 98 (2.7%) |
| recall@1 / @3 / @5 (λ=0.7) | 0.71 / 0.88 / 0.91 | 0.71 / **0.91** / **0.94** |
| MRR | 0.79 | **0.80** |
| Misses | 3 | **2** |

The predicted 0.94 is exactly what the rebuilt corpus delivered, and λ = 0.7 is still the
best setting after the rebuild (recall@3 0.91 and recall@5 0.94 against 0.85 / 0.91 for
plain top-k) — a default that survives a 12% change in the corpus is worth slightly more
than one fitted to a single snapshot, though it is still fitted.

**This is the argument for the eval set existing at all.** Both bugs sat in a corpus that
looked healthy on every statistic Step 2 reports — 43/43 papers on the sections path, no
over-budget chunks, sane chunk counts — and neither was visible until something downstream
measured whether the right paper came back.

### The remaining two misses are the encoder, not the pipeline

- `self-attention` ("how can a model work out which words in a sentence relate to each other
  without processing them one at a time") does not retrieve *Attention Is All You Need*. The
  paper never phrases the mechanism that way; the query is a description of the idea and the
  paper is a description of an architecture. This is the query set doing its job — it was
  written deliberately without reusing the title, and a bi-encoder with a 384-d MiniLM is
  what a paraphrase gap looks like.
- `token-level-matching` (ColBERT) surfaces DPR and BERT instead — semantically adjacent
  papers on the same task, and unchanged by the chunker fix. A cross-encoder rerank over the top 40 is the obvious fix, and it
  is out of scope by design. **This is the measurement that would justify adding one.**

### Probes behave as hoped

Both near-miss probes — MMR itself, and graph-based ANN indexes, neither in the corpus —
top out at **0.50** and **0.42**, against 0.67 for a well-answered question. The gap is
usable but not wide, so Phase 4 should treat a top score under ~0.5 as weak evidence rather
than trusting a hard threshold.

### Small things that changed under contact

- **The per-paper cap under-filled `k`.** The first real `--search` returned 4 results for
  "how is the kv cache paged during serving": only two papers cover it, cap 2, so the greedy
  loop ran out of legal candidates. Fixed by relaxing the cap once it blocks every remaining
  candidate — and the `for _ in range(k)` loop became a `while len(selected) < k` so that
  relaxing re-picks the slot it failed on instead of consuming it.
- **`ChunkStore.query` grew `include_vectors`.** Chroma does not return stored embeddings
  unless asked, and returns `None` rather than an empty list when not asked. The smoke path
  stays cheap; only retrieval pays for the vectors.
- **The CLI defers `import numpy`.** `--fetch-k` / `--lambda` / `--max-per-paper` default to
  `None` at the argparse layer and resolve to `rag.retrieve`'s constants at call time, so
  `--report` still runs on a checkout with no ML deps — the Step 3 rule, which a top-level
  `from rag.retrieve import DEFAULT_K` had quietly broken.

## Next

Phase 3 is complete: ingest → chunk → embed → store → retrieve, with **recall@5 0.94** on 34
hand-written queries. Phase 4 (`agent/`) builds the LangGraph agent on `retrieve()`.

The one piece of Phase 3 work left on the table is a **cross-encoder reranker** over the top
40 candidates, justified by the `token-level-matching` class of miss and measurable against
this eval set unchanged. It is out of scope here for the reason given in the scope boundary:
it changes the latency profile and puts a second model in the loop.
