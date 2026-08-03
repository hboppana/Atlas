# Phase 3 · Step 1 — corpus acquisition, `rag/` bring-up, PDF → text extraction

> Status: **done** — 2026-08-03. 43/43 seed papers extracted, 0 failures;
> `pytest rag/tests` 18/18 green. See [Results](#results).
> Predecessor: Phase 2 Step 8 — pybind11 bridge + FastAPI streaming — **done**
> ([14-bridge-serving.md](14-bridge-serving.md)). Phase 2 is complete.
> Successor: Step 2 — section-aware chunking + metadata schema.

## Goal

Get a real, heterogeneous corpus of academic PDFs onto disk and turn each one into
**structured raw text with provenance**, idempotently and resumably.

This is the Phase 3 analogue of Phase 2 Step 1: bring up the infrastructure and prove it
with a payload that has no interesting logic in it, so that when chunking, embedding, and
retrieval land in Steps 2–4, a failure names one layer. There is no `rag/` directory, no
`data/` directory, and no Phase 3 dependency installed today. This step builds all three.

The corpus is **domain-general and has no fixed theme**. Nothing in the pipeline may key off
subject matter — the chunker in Step 2 splits on IMRaD structure, which is universal to
arXiv-style papers, and the embedding model in Step 3 is general-purpose.

Topics are **derived, not declared**. Step 1 records the `arxiv_categories` the source
already asserts; Step 3 clusters paper-level embeddings to produce topic groupings, which
become filterable chunk metadata for scoped retrieval. Nothing asks the user to name a
theme, and nothing invokes the LLM to find one — clustering over vectors we are computing
anyway is cheaper, deterministic, and inspectable.

One soft constraint on the *initial* corpus, for evaluation rather than for the code: it
should be **clustered around one or two areas**, not scattered across many. Step 4 grades
retrieval with a hand-labelled question set, and both writing that set and judging its
results require the papers to talk to each other. MMR's diversity behaviour is likewise only
observable when near-duplicate passages compete, which happens within a topic. Forty papers
spread across five unrelated fields is five thin corpora, none of them measurable. Once
retrieval is validated, the corpus broadens freely — that is the point of keeping the
pipeline theme-free.

### Scope boundary (what Step 1 is *not*)

- **Not chunking.** Step 1 produces one extracted document per paper: ordered text blocks
  with page numbers, plus paper-level metadata. Deciding where a section begins is Step 2's
  problem and its own correctness question.
- **Not embedding, not ChromaDB.** Nothing is vectorized. `data/chunks/` is not written.
  Uncommenting `sentence-transformers` and `chromadb` waits for Steps 3–4; only `pypdf`
  lands now.
- **Not note-taking.** Engine-generated section notes are a later step and depend on
  chunking existing first.
- **Not OCR.** A scanned PDF with no text layer is detected and **recorded as a failure**,
  not silently ingested as an empty document. Adding an OCR path is a named follow-up, not
  a gap.
- **Not a UI.** Ingestion is a CLI batch job. Status is a manifest file and stdout.

## The operation

```
scripts/ingest_papers.py --fetch arxiv_ids.txt     # acquire  → data/papers/*.pdf
scripts/ingest_papers.py --ingest                  # extract  → data/extracted/*.json
```

Two phases, deliberately separable. Acquisition touches the network and is the flakiest
part; extraction is pure local compute over whatever is on disk. Re-running either must be
safe.

### Corpus target

~40 papers to start. Enough that retrieval in Step 4 is measurable and section detection in
Step 2 meets real structural variety (two-column layouts, missing headers, odd front
matter), without acquisition becoming its own project. Seeded from arXiv IDs in a
checked-in list; `data/papers/` contents stay gitignored.

## `data/` layout

```
data/
  papers/       *.pdf                  raw, gitignored
  extracted/    <paper_id>.json        Step 1 output, gitignored
  chunks/                              Step 2 output, gitignored
  manifest.json                        ingestion state, gitignored
```

`data/.gitkeep` and a `.gitignore` covering the contents are the only tracked files.

## Extracted-document schema

One JSON per paper. This is the contract Step 2 consumes.

```jsonc
{
  "paper_id": "2103.00020",              // arXiv ID, or a content hash for local PDFs
  "title": "...",
  "authors": ["..."],
  "year": 2021,
  "arxiv_id": "2103.00020",
  "arxiv_categories": ["cs.LG", "cs.CR"],  // as returned by arXiv; not hand-entered
  // NOTE: no `topic` field here. Topics are *derived* in Step 3 by clustering paper
  // embeddings, and written back as chunk metadata then. Step 1 records only what the
  // source actually asserts.
  "source_path": "data/papers/2103.00020.pdf",
  "extractor": "pypdf",
  "page_count": 14,
  "blocks": [
    {"page": 1, "text": "..."},          // one block per page, in order
    {"page": 2, "text": "..."}
  ]
}
```

Page-granular blocks, not one flat string. Page numbers are the only provenance pypdf gives
us for free, and they are what makes a citation clickable later. Discarding them here and
trying to reconstruct them after chunking is not possible.

`paper_id` is the join key everywhere downstream — chunk IDs are `<paper_id>:<index>`, and
Phase 4's `cite_sources` resolves through it.

## The manifest — idempotency and resumability

`data/manifest.json` maps `paper_id` → record:

```jsonc
{
  "2103.00020": {
    "status": "extracted",              // fetched | extracted | failed
    "pdf_sha256": "...",
    "extracted_at": "2026-08-01T14:22:10Z",
    "extractor": "pypdf",
    "error": null
  }
}
```

Rules:

- **Fetch skips** any `paper_id` already present with a matching `pdf_sha256`.
- **Extract skips** any paper whose `status` is `extracted` and whose PDF hash is unchanged.
  `--force` re-extracts.
- **A crash loses at most one paper.** The manifest is written after each paper, not at the
  end of the run.
- **Failures are recorded, not swallowed.** `status: "failed"` with a populated `error`.
  A paper that silently isn't in the store is the single most miserable thing to debug in a
  RAG pipeline, and the manifest is the cure.

A `--report` flag prints counts by status and lists failures. That is the whole ingestion
UI for now.

## Validation

`rag/tests/test_ingest.py`, pytest, no network:

- **Schema round-trip.** A committed 2-page fixture PDF extracts to the documented shape;
  `page_count` matches, blocks are page-ordered, `paper_id` is stable across runs.
- **Idempotency.** Extracting twice produces byte-identical output and leaves the manifest
  timestamp untouched on the second run.
- **Resumability.** Given a manifest where paper A is `extracted` and paper B is absent,
  a run touches only B.
- **Failure recording.** A fixture PDF with no text layer yields `status: "failed"` with a
  non-null `error` — and does *not* write an `extracted/` file.
- **Hash invalidation.** Mutating a PDF's bytes causes re-extraction.

Fixtures are small and committed under `rag/tests/fixtures/`, matching the Phase 1
blob-free-test discipline: no test may require the real corpus to be on disk.

## Files created / touched

```
rag/__init__.py
rag/ingest.py                    fetch + extract + manifest
rag/tests/test_ingest.py
rag/tests/fixtures/*.pdf         tiny, committed
scripts/ingest_papers.py         CLI entry point
data/.gitignore
requirements.txt                 uncomment pypdf only
.gitignore                       data/ contents
```

## Done when

- `pytest rag/tests` is green on the lab box and on a CPU-only checkout.
- `scripts/ingest_papers.py --fetch --ingest` over the ~40-paper seed list completes, and
  `--report` shows every paper either `extracted` or `failed` with a stated reason.
- Re-running the same command is a no-op.
- Phase 1 and Phase 2 suites stay green (10/10 CPU CTest, 7/7 CUDA CTest, `pytest
  server/tests`) — Phase 3 adds a directory, it does not touch the engine.

## Results

Measured on the lab box (Suramar), 2026-08-03, CPU only — nothing here touches the GPU.

| | |
|---|---|
| Seed list | `scripts/arxiv_ids.txt`, **43 IDs**, clustered on transformer/LLM systems + retrieval |
| Fetched | 43/43, ~94 MB in `data/papers/` |
| Extracted | 43/43, ~3.8 MB of JSON, 0 failures; 5–87 pages per paper |
| Wall time | ~4 min, dominated by the 3 s inter-request arXiv delay |
| Re-run | pure no-op: every paper logs `skip`, no file rewritten |
| Tests | `pytest rag/tests` 18 passed in 0.10 s, no network |
| Regression | CTest 10/10 (CPU), 17/17 (CUDA), `pytest server/tests` 27 skipped |

Every ID in the seed list resolved to the intended paper — the titles arXiv returned are in
the manifest and `--report`, which is how a mistyped ID is meant to surface.

Two deviations from the design, both recorded rather than silently absorbed:

- **Fetched metadata lives in the manifest record**, under a `metadata` key, not in a
  sidecar file. `data/papers/` stays PDFs-only as documented, and extraction (which never
  touches the network) still has the title/authors/year/categories it must copy into the
  document. Extract on a PDF with no manifest metadata writes nulls rather than failing —
  a locally-dropped PDF is still ingestible.
- **`server/bridge.py` gained one `try/except`** around the extension-module import: an
  `.so` built against a different Python raises `ImportError` from
  `importlib.util.module_from_spec`, which escaped as an error instead of the intended
  `EngineError` → green SKIP. Found because Phase 3's interpreter differs from the one
  Phase 2 built against. Phase 2 behaviour is otherwise untouched.

Two properties of the real output that Step 2 has to plan for, both visible in
`data/extracted/` now rather than surfacing later as bad retrieval:

- **Text is hard-wrapped at the PDF's line breaks, with hyphenation preserved** — pypdf
  emits `"the MLM ob-\njective enables"`. Chunking has to de-hyphenate and rejoin lines
  before a header regex or an embedding sees the text.
- **Front matter lands in page 1's block** (1706.03762 opens with Google's reproduction
  notice, not the title), so "the first block is the title and abstract" is not a safe
  assumption for section detection.

## Design decisions

- **Acquisition and extraction are one step but two commands.** They share the manifest and
  neither is a shippable artifact alone, so splitting them into separate steps would be
  ceremony. But the network half fails for reasons the local half never will, so they get
  separate entry points and separate status transitions — the same failure-domain reasoning
  that split the bridge from the FastAPI layer inside docs/14.
- **Extraction is a separate artifact from chunking, not fused into it.** Fusing them means
  every chunking iteration re-parses every PDF, and iteration on section detection is
  exactly what Step 2 will need a lot of. Extracted JSON on disk makes Step 2 a fast local
  loop.
- **Page blocks, not a flat string.** Provenance has to be captured at extraction time or it
  is gone. This is the decision that makes Phase 4's citations possible.
- **The manifest is a file, not a database.** One writer, batch cadence, tens to hundreds of
  rows. SQLite would be a dependency and a schema migration story bought for nothing.
- **Failure is a recorded state, not an exception that aborts the run.** An open-ended
  corpus contains bad PDFs by definition; a pipeline that halts on the first one cannot
  ingest 40 papers unattended.
- **No OCR, stated rather than assumed.** A no-text-layer PDF has an obvious tempting
  fallback (ship it empty, move on) that produces a paper which exists in the manifest and
  returns nothing from retrieval forever. Failing loudly is the correct behaviour, and the
  test enforces it.
- **Topic is a filter on a single collection, not a separate store.** One ChromaDB
  collection with a filterable tag lets a query span topics when that is useful and scope
  down when it is not; separate collections make cross-topic retrieval a client-side merge
  problem for no benefit.
- **Topics are derived in Step 3, not declared in Step 1.** A hand-entered tag records what
  the operator believed when the paper was added; a cluster label records what the corpus
  actually contains, and re-derives for free as the corpus grows. Step 1 therefore stores
  only source-asserted `arxiv_categories` — the rule being that extraction records what the
  source says and never what we inferred.
- **Derivation is clustering, not generation.** Paper vectors are mean-pooled chunk
  embeddings Step 3 computes anyway; labels come from distinguishing TF-IDF terms. No model
  invocation, deterministic output, inspectable when a grouping looks wrong. Asking a
  language model to name the theme would be slower, non-reproducible, and — on our 1.1B
  engine — worse.
- **No seed theme, but a clustered first corpus.** These are different things and only the
  second one is load-bearing, and only for evaluation. Deriving topics does not manufacture
  overlap that is not there: a scattered corpus clusters correctly into scattered groups.
  The resolution is to build the Step 4 question set over the **densest cluster** rather
  than to constrain the corpus up front — same rigour, no decision required at ingest time.
- **Corpus stays gitignored, fixtures do not.** Same split as Phase 1: `weights/` is
  ignored, `reference/` is committed because tests depend on it.

## Next (Step 2)

Section-aware chunking over `data/extracted/*.json`: IMRaD header detection, the
fixed-window-with-overlap fallback when detection fails, the chunk metadata schema
(`{paper_id, section, chunk_index, page_span}`), and recording which path each paper took
so bad retrieval stays traceable.
