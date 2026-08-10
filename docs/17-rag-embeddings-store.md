# Phase 3 · Step 3 — local embeddings + ChromaDB store + derived topics

> Status: **design** — 2026-08-10.
> Predecessor: Step 2 — section-aware chunking — **done**
> ([16-rag-chunking.md](16-rag-chunking.md)). 43/43 papers `sections`, 3489 chunks.
> Successor: Step 4 — MMR retrieval + a retrieval eval set.

## Goal

Turn `data/chunks/<paper_id>.json` into a **queryable local vector store**: every chunk
embedded once by `all-MiniLM-L6-v2` on an A6000, cached to disk, and upserted into a
persistent ChromaDB collection with citation-grade scalar metadata — plus **derived topics**,
clustered from paper vectors, so Step 4 and Phase 4 can scope retrieval without anyone
declaring a taxonomy.

After this step the corpus is searchable end to end. Step 4 only changes *how* results are
selected (MMR, reranking), not what is stored.

Nothing here keys off subject matter. The topic labels are computed from the corpus that
happens to be on disk; adding 200 papers from another field re-derives them.

### Scope boundary (what Step 3 is *not*)

- **Not retrieval policy.** `retrieve.py`, MMR, reranking, and the eval set are Step 4.
  Step 3 ships exactly one query path: a `--query` smoke flag that does plain top-k cosine,
  so the store can be proven to work before the selection logic exists.
- **Not re-chunking, not re-extraction.** Embedding reads chunk files and nothing else. If a
  chunk is bad, the fix is `chunk.py` and a `chunker_version` bump — *except* for the one
  measurement this step owns (below): whether the word heuristic ever exceeds MiniLM's real
  256-wordpiece window.
- **Not a hosted embedding API.** `all-MiniLM-L6-v2` weights are downloaded once from the
  HF hub, pinned by revision, and cached; every embed run after that is offline. Same
  posture as Phase 1's reference weights.
- **Not an LLM in the loop.** Topic labels come from term statistics, not from asking
  TinyLlama to name a cluster. Engine-generated section notes remain a later step.
- **Not a server.** No HTTP surface for the store; Phase 4's agent and Phase 5's MCP tools
  import `rag/` directly.

## The operation

```
scripts/ingest_papers.py --embed                 # chunks → vectors → Chroma; derives topics
scripts/ingest_papers.py --embed --force         # re-embed everything
scripts/ingest_papers.py --query "flash attention tiling" -k 5
scripts/ingest_papers.py --report                # + embedded counts, collection size, topics
```

One CLI, one manifest, one report — the Step 2 argument unchanged. `--embed` is a new status
transition (`chunked` → `embedded`), so its failures are isolated from chunking's.

`--embed` runs two passes, and the split is the whole shape of this step:

1. **Per paper** — load chunk file, embed its chunks, write `data/embeddings/<paper_id>.npz`,
   mark the manifest. Resumable, skippable, one paper's failure costs one paper.
2. **Corpus-level** — mean-pool every paper's vectors, cluster, label the clusters, then
   upsert *all* chunks into Chroma with topic metadata attached.

Topics cannot be computed per paper: a cluster is a statement about the corpus. So pass 2
runs whenever the set of embedded papers changed, and it rewrites topic metadata for papers
whose vectors did not change (a metadata-only `collection.update`, no re-embedding).

## Embedding

`rag/embed.py`.

```python
MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "<pinned sha>"          # recorded like Phase 1 pins its reference weights
EMBED_DIM      = 384
MAX_SEQ_TOKENS = 256                     # MiniLM truncates silently past this
EMBEDDER_VERSION = 1                     # bump ⇒ re-embed, same role as chunker_version
BATCH_SIZE     = 64
```

- **Vectors are L2-normalized at write time** (`normalize_embeddings=True`), and the Chroma
  collection uses `hnsw:space = "cosine"`. Normalizing once on disk means Step 4's MMR can
  use a plain dot product for cosine similarity and never has to renormalize inside its
  inner loop.
- `float32` on disk. 3489 × 384 × 4 B ≈ **5.4 MB** for the whole corpus — small enough that
  caching vectors as a separate artifact is free, and that is the point: rebuilding the
  Chroma collection, changing the metadata schema, or debugging clustering must never
  require re-running the model.
- **Device**: `cuda` when available, else `cpu`; the lab box is shared, so the CLI takes
  `--device` and the documented invocation pins a card with `CUDA_VISIBLE_DEVICES=0`.
  Model in `eval()`, `torch.inference_mode()`, fp32 (fp16 buys nothing at 3.5 k chunks and
  costs reproducibility).
- **Batching is by chunk, sorted by length within a paper** to cut padding waste, then
  restored to chunk order before writing. Chunk order on disk is always `chunk_index` order.

### The truncation check (the one measurement Step 3 owes Step 2)

Step 2 counted tokens with the word heuristic `ceil(words * 1.35)` and budgeted 200 tokens
against MiniLM's 256, explicitly deferring exactness to this step. Step 3 has the real
tokenizer, so it checks rather than assumes: every chunk's true wordpiece count is computed
during embedding, and the run prints

```
tokens: max 231, p99 214, over 256: 0 / 3489
```

- **0 over 256** → the heuristic's headroom held; nothing to do, and the measured
  distribution goes in *Results* so nobody re-derives it.
- **Any over 256** → those chunks were silently truncated, which is the exact failure Step 2
  exists to prevent. The fix is to inject `rag.embed.minilm_token_counter` into
  `chunk_papers`, bump `CHUNKER_VERSION` to 2, re-chunk, and re-embed. The seam already
  exists; this is why it exists.

Chunking's default counter stays the heuristic either way. Making the chunker's output
depend on whether `sentence-transformers` happens to be installed would be a
non-reproducibility bug, not a feature.

## Embedding cache file

`data/embeddings/<paper_id>.npz` — `np.savez` with:

| Key | Contents |
|---|---|
| `ids` | `<U…` array of `chunk_id`, in `chunk_index` order |
| `vectors` | `(n_chunks, 384)` float32, L2-normalized, row-aligned to `ids` |
| `meta` | 0-d JSON string: `embedder_version`, `model_name`, `model_revision`, `chunker_version`, `source_sha256` (hash of the chunk file), `dim` |

`source_sha256` + `chunker_version` + `embedder_version` form the skip key, mirroring Step 2
exactly: a re-chunked paper re-embeds, a bumped embedder re-embeds the corpus, and
everything else is a no-op.

## ChromaDB store

`rag/store.py`. Thin, deliberately: Chroma is a dependency to be isolated, not a framework to
build on.

```python
client     = chromadb.PersistentClient(path=str(paths.chroma_dir))   # data/chroma/
collection = client.get_or_create_collection(
    name="atlas_chunks",
    metadata={"hnsw:space": "cosine"},
    embedding_function=None,        # we always pass vectors; Chroma must never embed
)
```

`embedding_function=None` is load-bearing. Chroma's default silently downloads its own ONNX
MiniLM and would embed queries with a *different* pipeline than the corpus — an invisible
train/serve skew. Every `add`/`query` in Atlas passes explicit vectors from `rag/embed.py`.

**Document** = the chunk text (so Chroma's own `documents` field is what Phase 4 quotes).
**ID** = `chunk_id` (`<paper_id>:<index>`), already the format `cite_sources` expects.

**Metadata — scalars only**, as Step 2 anticipated:

| Key | Source | Note |
|---|---|---|
| `paper_id` | chunk file | the join key; every filter and delete goes through it |
| `chunk_index` | chunk file | int |
| `section` | chunk file | `""` under the `fixed` strategy |
| `section_index` | chunk file | int |
| `page_start`, `page_end` | `page_span` flattened | Chroma rejects lists |
| `n_tokens_est` | chunk file | int |
| `title`, `year` | extracted doc | looked up at insert time, not stored per chunk on disk |
| `arxiv_categories` | extracted doc | joined `"cs.CL,cs.LG"` — source-asserted |
| `topic_id`, `topic_label` | derived, pass 2 | the only inferred fields, and they are marked |
| `chunker_version`, `embedder_version` | provenance | lets a stale row be identified in place |

**Writes are per paper and destructive-then-additive**: `collection.delete(where={"paper_id":
pid})` before `add`. A re-chunked paper usually has a *different number* of chunks, so an
upsert keyed on `chunk_id` alone would leave orphaned high-index rows from the previous run —
rows that still match queries and cite page numbers that no longer exist.

Chroma calls are batched (≤ 2000 records) and the collection is a `PersistentClient`, so the
store survives the process without an explicit persist call.

## Derived topics

The Phase 3 rule is *topics are derived, never declared*. Concretely, in `rag/embed.py`:

1. **Paper vector** = mean of the paper's chunk vectors, renormalized. (Abstract-only would
   be sharper but 2/43 papers have no detectable abstract; the mean is the robust choice and
   it costs nothing.)
2. **k-means, from scratch in numpy** — k-means++ init with a fixed seed, Lloyd iterations to
   convergence or 100 steps, on unit vectors so Euclidean and cosine order agree. ~40 lines,
   deterministic, no `scikit-learn` in `requirements.txt`. Consistent with the phase's
   from-scratch ethos, and clustering 43 × 384 floats does not need a library.
3. **k is chosen, not fixed**: run k ∈ [2, 8] (capped at `n_papers // 4`), pick the best mean
   silhouette score. A hand-picked k on a corpus that is meant to broaden is a constant that
   goes stale silently.
4. **Labels from distinguishing terms**: for each cluster, score unigrams/bigrams from its
   papers' titles + abstract-section chunks by `tf_in_cluster / (tf_in_corpus + 1)`,
   stopword-filtered, and join the top 3 — e.g. `topic_2 · attention kernels, gpu memory`.
   The label is a human-readable handle for a filter, not a claim about the field.
5. Written to **`data/topics.json`** (`k`, silhouette, per-cluster label + member `paper_id`s
   + top terms) and attached to every chunk of a member paper as `topic_id` / `topic_label`.

Topics are recomputed whenever the embedded paper set changes, so `topic_id` is **not
stable across corpus growth**. That is recorded here and in `topics.json` (which carries a
`derived_at` and the member list) because anything downstream that persists a `topic_id` —
an eval fixture, an agent prompt — would otherwise rot invisibly. Filters should be written
against labels resolved at query time, never against a memorized integer.

## Manifest integration

New status `embedded`, after `chunked` in the same pipeline-position sense:

```jsonc
"status": "embedded",                // fetched | extracted | chunked | embedded | failed
"embedded_at": "2026-08-10T...Z",
"embedder_version": 1,
"embedding_count": 187,
"topic_id": 2
```

Rules, unchanged from Steps 1–2: skip when `source_sha256` + `chunker_version` +
`embedder_version` all match; `--force` re-embeds; the manifest is written after every paper;
a per-paper failure is `status: "failed"` with a populated `error` and no `.npz`, and the
remaining 42 papers still finish.

## Validation

`rag/tests/test_embed.py`, `rag/tests/test_store.py` — pytest, no network, no corpus.

The heavy deps are quarantined the same way Step 2 quarantined the tokenizer: **all logic
that is not literally "call the model" runs against an injected embedder**, a deterministic
stub that hashes text to a unit vector. So the cache format, the skip rules, the metadata
flattening, k-means, silhouette, labelling, and the Chroma write path are all covered on a
checkout with no `sentence-transformers` installed. Only two tests need the real model, and
they are `pytest.importorskip`-guarded.

- **Cache round-trip.** Embed a synthetic 2-paper corpus with the stub → `.npz` has
  `ids` aligned to `vectors`, dim 384, rows unit-norm to 1e-6, `meta` complete.
- **Skip / invalidation.** Second run is a pure no-op; touching a chunk file re-embeds only
  that paper; bumping `embedder_version` re-embeds all; `--force` re-embeds all.
- **Re-chunk shrinkage.** A paper re-embedded with *fewer* chunks leaves no orphan rows in
  the collection — the delete-then-add test, and the reason that rule exists.
- **Metadata contract.** Every value in every metadata dict is `str | int | float | bool`;
  `page_span` arrives as `page_start`/`page_end`; `arxiv_categories` is a comma string;
  `title`/`year` come from the extracted doc, not the chunk file.
- **Query path.** With the stub embedder, querying with the exact text of chunk *i* returns
  chunk *i* first; a `where={"paper_id": …}` filter returns only that paper's chunks.
- **Clustering.** Three well-separated synthetic vector groups recover 3 clusters with the
  right membership; k-means is deterministic across runs with the same seed; a corpus of 3
  papers does not crash the k-selection loop.
- **Labelling.** A cluster whose papers repeat a distinctive term gets it in the label; a
  term common to the entire corpus does not appear in any label.
- **Failure isolation.** One paper with a corrupt chunk file → `status: "failed"` with an
  error, the other papers `embedded`, exit code non-zero.
- **Real-model tests (skipped without deps):** MiniLM output is 384-d and unit-norm, and
  `"the cat sat on the mat"` is closer to `"a cat is on a rug"` than to
  `"gradient checkpointing reduces activation memory"`. Not a bit-exactness assertion —
  GPU reductions differ from CPU at ~1e-6, so identity is asserted as cosine > 0.9999.

Plus a **corpus smoke run** by hand: `--embed` over all 43 papers, then `--report`, then
3–4 `--query` probes whose expected paper is obvious (a query about MMR should surface the
retrieval papers, one about attention kernels the FlashAttention papers). This is a sanity
check, not the eval — the real retrieval eval set is Step 4's deliverable.

## Files created / touched

```
rag/embed.py                     model load, batch embed, cache, k-means + topic labels
rag/store.py                     Chroma collection, upsert, delete-by-paper, query
rag/tests/test_embed.py
rag/tests/test_store.py
rag/ingest.py                    Paths.embeddings_dir / chroma_dir / topics_path;
                                 EMBEDDED status; report() embed + topic sections
scripts/ingest_papers.py         --embed, --device, --query/-k flags
requirements.txt                 uncomment chromadb, sentence-transformers
data/.gitignore                  embeddings/, chroma/, topics.json
docs/17-rag-embeddings-store.md  this file
```

`sentence-transformers` pulls `torch` + `transformers`, both already pinned for Phase 1.
`chromadb` is the only genuinely new tree.

## Done when

- `pytest rag/tests` green (Step 1–2's 46 plus the new ones) on a CPU-only checkout with no
  ML deps installed — the real-model tests skip, everything else runs.
- `CUDA_VISIBLE_DEVICES=0 scripts/ingest_papers.py --embed` embeds 43/43 papers, 0 failures,
  and reports the true-wordpiece histogram with **0 chunks over 256** (or the chunker is
  bumped to v2 and the corpus re-chunked, per the rule above).
- `data/chroma/` holds a collection whose count equals the corpus chunk count (3489 today),
  and `--report` shows it alongside the manifest counts.
- `--embed` re-run is a pure no-op; `embedder_version` bump re-embeds all 43.
- `data/topics.json` exists with k chosen by silhouette, every paper assigned, and labels
  that a human reading the corpus would recognize as approximately right.
- 3–4 `--query` probes return an obviously-correct paper in the top 3.
- Phase 1 / Phase 2 suites stay green.

## Design decisions

- **Vectors are a separate on-disk artifact, not only Chroma rows.** Same reasoning that
  split chunking from extraction: iterating on the metadata schema, the collection layout, or
  the clustering must not re-run the model, and a 5 MB cache buys that outright. It also
  makes Step 4's MMR implementable over a numpy matrix without going through Chroma's API.
- **`embedding_function=None`, always explicit vectors.** Letting Chroma embed queries with
  its bundled ONNX model would put a *different* encoder on the query side than the corpus
  side. It would mostly work, which is what makes it dangerous.
- **Normalize at write time, cosine space.** One decision, made once, that removes a whole
  class of "why is this similarity 4.7" bugs and makes MMR a dot product.
- **Delete-by-paper before insert.** Upsert-by-id silently leaks orphan chunks when a paper
  re-chunks to fewer chunks, and an orphan is worse than a missing row: it retrieves, and it
  cites a page span that no longer exists.
- **k-means in numpy rather than scikit-learn.** 40 lines against a large transitive
  dependency tree, for a clustering problem of 43 points. The from-scratch ethos and the
  dependency budget point the same way here; if clustering ever needs HDBSCAN-class behaviour
  that calculus changes.
- **k chosen by silhouette, not pinned.** The corpus is explicitly meant to broaden. A pinned
  k is a constant that becomes wrong without ever failing.
- **Topic labels from term statistics, not from the engine.** An LLM-generated label would be
  nicer prose and would be non-deterministic, unauditable, and a new dependency of the
  ingestion path on the inference path. Term scores are inspectable in `topics.json`.
- **`topic_id` is explicitly unstable.** Deriving topics means they move when the corpus
  moves. Saying so here, and stamping `derived_at` + members into `topics.json`, is cheaper
  than the alternative of pretending stability and having a Step 4 fixture rot silently.
- **The truncation check runs at embed time, not as a one-off script.** It is the assertion
  that Step 2's whole token budget was correct; it costs one tokenizer pass over text the
  model is about to tokenize anyway, and it turns a silent quality loss into a printed number
  on every run.
- **A flag on the existing CLI, not a new script** — third time, same reason: one manifest,
  one report, one entry point, with separate status transitions doing the failure isolation.

## Next (Step 4)

Retrieval: `rag/retrieve.py` over the cached matrix + Chroma — top-k by cosine, then **MMR**
for diversity, optional `paper_id` / `topic_label` / `year` filters, and a small hand-built
retrieval eval set (query → expected paper(s)) with recall@k measured, so Phase 4's agent is
built on a retriever with a known number attached to it.
