# Phase 3 · Step 2 — section-aware chunking + chunk metadata schema

> Status: **designed** — 2026-08-05.
> Predecessor: Step 1 — corpus acquisition + PDF extraction — **done**
> ([15-rag-ingest-extraction.md](15-rag-ingest-extraction.md)). 43/43 papers extracted.
> Successor: Step 3 — local embeddings + ChromaDB store + derived topics.

## Goal

Turn `data/extracted/<paper_id>.json` — ordered page blocks of raw pypdf text — into
`data/chunks/<paper_id>.json`: **retrieval-sized passages that respect paper structure and
carry enough metadata to cite**.

This is the quality lever of the whole RAG phase. Embedding and retrieval are largely
off-the-shelf; how the text is cut is the part that decides whether the agent in Phase 4
sees a coherent argument or half of one sentence glued to a table caption. Everything
downstream — MMR diversity, the Step 4 eval set, `cite_sources` — reads what this step
writes.

Nothing here keys off subject matter. Section structure (IMRaD, numbered headers, an
abstract, a bibliography) is a property of the arXiv-paper *form*, not of its field.

### Scope boundary (what Step 2 is *not*)

- **Not embedding, not ChromaDB.** No vectors, no `sentence-transformers`, no `chromadb`.
  Chunking is pure local text processing over JSON, and stays testable with zero ML deps
  installed. Token budgets are honoured through an injectable counter (below), not by
  importing a tokenizer today.
- **Not topic assignment.** Topics are derived in Step 3 by clustering paper embeddings.
  Step 2 writes no `topic` field — same rule as Step 1: record what the source asserts,
  never what we inferred.
- **Not figure/table/equation understanding.** pypdf gives us a text layer with tables
  flattened into prose-shaped garbage. Step 2 does not attempt to reconstruct them; it
  drops what it can identify cheaply (see *Junk removal*) and lets the rest ride.
- **Not re-extraction.** If a chunking bug traces to bad extraction, the fix is in
  `ingest.py` and the doc is re-extracted. Chunking never opens a PDF.
- **Not engine-generated section notes.** Still a later step, still dependent on this one.

## The operation

```
scripts/ingest_papers.py --chunk            # data/extracted/*.json → data/chunks/*.json
scripts/ingest_papers.py --report           # now also prints chunk counts + strategy split
```

One CLI, one manifest, one report. Fetch, extract, and chunk are stages of a single
pipeline over a single state file, so they are flags on the existing entry point rather
than a second script that would need its own manifest loading, its own path handling, and
its own `--report`. They keep separate status transitions for the same reason Step 1 kept
`--fetch` and `--ingest` separate: distinct failure domains.

## Chunk-file schema

One JSON per paper. This is the contract Step 3 consumes.

```jsonc
{
  "paper_id": "1706.03762",
  "chunker_version": 1,
  "source_sha256": "...",            // hash of the extracted JSON this was built from
  "strategy": "sections",            // sections | fixed  — which path this paper took
  "sections_detected": 7,
  "chunks": [
    {
      "chunk_id": "1706.03762:0",
      "chunk_index": 0,
      "text": "...",                 // normalized, de-hyphenated, ready to embed
      "section": "Introduction",     // "" when strategy == "fixed"
      "section_index": 1,            // ordinal of the section within the paper
      "page_span": [2, 3],           // inclusive first/last source page
      "n_chars": 1180,
      "n_tokens_est": 214
    }
  ]
}
```

Chunk metadata that ends up in ChromaDB must be **scalars** — Chroma rejects lists in a
metadata dict. `page_span` is therefore stored as a pair here and flattened to
`page_start` / `page_end` at insert time in Step 3, and `arxiv_categories` is joined to a
comma string there. Paper-level fields (`title`, `year`, `arxiv_categories`) are **not**
copied into every chunk: they are looked up from the extracted document at insert time.
Duplicating them 200× per paper on disk buys nothing and makes a metadata fix a re-chunk.

`chunk_id` is `<paper_id>:<chunk_index>` — the format Phase 4's `cite_sources` already
expects, and `chunk_index` is monotonic across the whole paper, not per-section, so it
sorts into reading order.

## Normalization — before anything else looks at the text

Step 1's Results recorded the two properties that force this, both visible in
`data/extracted/` today:

- text is hard-wrapped at the PDF's line breaks with **hyphenation preserved** —
  `"the MLM ob-\njective enables"`;
- **front matter lands in page 1's block**, so "the first block is title + abstract" is not
  a safe assumption.

`normalize(text)`, applied per page block before detection or splitting:

1. **De-hyphenate** across a line break: `(\w)-\n(\w)` → `\1\2`. Only when both sides are
   word characters and the prefix is not itself a full hyphenated compound already ending a
   line (`state-\nof-the-art` keeps its hyphen — handled by requiring the next line to start
   lowercase).
2. **Rejoin soft-wrapped lines.** A line break that does not end a sentence and is not
   followed by a header candidate or a list marker becomes a space. Paragraph breaks
   (blank line, or a line ending in `.`/`?`/`:` followed by an indent) survive.
3. **Collapse whitespace**, normalize ligatures (`ﬁ`→`fi`) and the Unicode dashes/quotes
   pypdf passes through.
4. **Preserve line structure for the header pass.** Detection runs on the pre-rejoin line
   list; rejoining runs on the section bodies. Doing it the other way round destroys the
   only signal a header has — that it occupies its own line.

Normalization is a pure function with its own tests. It is the single highest-leverage
30 lines in the phase: an embedding of `"ob- jective"` is a different vector than
`"objective"`, and no amount of MMR tuning recovers from that.

## Section detection

Measured over the real corpus (all 43 extracted papers, 2026-08-05):

| Signal | Coverage |
|---|---|
| ≥3 distinct top-level numbered headers (`1 Introduction`, `2 Related Work`, …) | **43/43** |
| An `Abstract` line | 41/43 |
| A `References` line | 43/43 |

So the **numbered-header regex is the primary detector**, and it is not a guess — it hits
every paper we have. The design still keeps a fallback, because the corpus is meant to
broaden and the first unnumbered-header paper must degrade rather than crash.

**Primary pass.** A line is a header candidate when it matches
`^\s{0,3}(\d{1,2})(\.\d+)*\.?\s+([A-Z].{2,60})$` and is ≤8 words. Candidates are accepted
only if the top-level numbers form a **monotonically non-decreasing run starting at 1** —
that single constraint is what rejects the false positives an unanchored regex produces in
bulk (table cells, `4 Model`, figure captions, reference-list entries). Sub-numbered
headers (`3.1 Encoder`) attach to their parent section and do **not** open a new chunk
group; a paper's retrievable unit is a top-level section.

**Unnumbered heads** are matched by an explicit small vocabulary regardless of numbering —
`abstract`, `references`, `bibliography`, `acknowledg(e)ments`, `appendix`. `Abstract` is
its own section (it is the highest-value passage in the paper for retrieval), and the
others mark the tail.

**Acceptance gate.** A paper takes the `sections` strategy when ≥3 top-level headers are
accepted; otherwise `strategy: "fixed"` and the whole body is windowed blind. **Which path
each paper took is recorded in the chunk file and in `--report`.** This is the debuggability
requirement Step 1 established for failures, applied to a silent quality regression: when
retrieval is bad for one paper, the first question is "did section detection work on it?",
and that must be answerable without a rerun.

### Junk removal

Applied in both strategies, all cheap and all recorded as a dropped-character count:

- **Everything from `References`/`Bibliography` onward is dropped.** A bibliography embeds
  as a dense cloud of title-like text that matches every query about any of the cited works
  and answers none of them — the classic RAG corpus poison. 43/43 papers have a detectable
  references line, so this is nearly free. Appendices are **kept** (they carry real method
  detail).
- Repeated header/footer lines (the same short line on >60% of pages), page numbers, and
  arXiv stamp lines.
- Blocks under `MIN_CHUNK_CHARS` after normalization are merged into the neighbour rather
  than emitted as a 40-character chunk.

## Splitting within a section

The embedding model is `all-MiniLM-L6-v2`, which **truncates at 256 wordpiece tokens**.
Anything longer is not "a bit degraded" — the tail is silently discarded. So the target is
**~200 tokens per chunk with ~40 tokens of overlap**, which leaves headroom for the
tokenizer counting more pieces than the estimate.

- A section that fits in one chunk stays one chunk.
- A longer section is split at **paragraph boundaries** first, packing paragraphs greedily
  up to the budget; a single paragraph over budget is split at sentence boundaries; a single
  sentence over budget (an equation-dense line) is split at the token limit.
- **Overlap carries the tail of the previous chunk into the next**, at sentence granularity,
  so a claim spanning a boundary survives in at least one chunk whole.
- Every chunk records the `page_span` it came from. Pages are tracked as character offsets
  during normalization so a chunk that straddles a page boundary reports `[7, 8]` truthfully
  rather than guessing.

Token counting uses an **injectable `token_counter` callable**, defaulting to a
dependency-free word heuristic (`ceil(words * 1.35)`, calibrated against MiniLM's tokenizer
on a sample). Step 3, which has `sentence-transformers` installed, passes the real
tokenizer in and the same code path becomes exact. This is what keeps Step 2's tests
runnable on a CPU-only checkout with no ML deps — the same discipline as Phase 1's blob-free
tests.

## Manifest integration

Chunking adds to each manifest record:

```jsonc
"status": "chunked",                 // fetched | extracted | chunked | failed
"chunked_at": "2026-08-05T...Z",
"chunker_version": 1,
"chunk_count": 187,
"chunk_strategy": "sections"
```

Rules, mirroring Step 1:

- **Skip** any paper already `chunked` whose extracted-JSON hash **and** `chunker_version`
  are unchanged. Bumping `chunker_version` re-chunks the corpus — that is the intended
  iteration loop, and it must not require `--force` or a manual `rm -rf`.
- `--force` re-chunks regardless.
- The manifest is written after each paper; a crash loses at most one.
- A per-paper failure is `status: "failed"` with a populated `error` and **no** chunk file.

## Validation

`rag/tests/test_chunk.py`, pytest, no network, no corpus, no ML deps. Fixtures are small
**JSON documents** (the Step 1 output shape) built inline or committed under
`rag/tests/fixtures/` — chunking's input is JSON, so its fixtures are JSON; no new PDFs.

- **Normalization.** `"ob-\njective"` → `"objective"`; `"state-\nof-the-art"` keeps its
  hyphen; soft-wrapped lines rejoin; paragraph breaks survive; ligatures normalize.
- **Header detection, positive.** A synthetic doc with `1 Introduction … 5 Conclusion`
  yields 5 sections with the right titles, in order.
- **Header detection, negative.** A doc whose only numbered-looking lines are table cells
  (`4 Model`, `12 OOM`) out of sequence yields `strategy: "fixed"` — the monotonic-run gate
  is what this test pins.
- **Fallback.** A header-free doc chunks into windowed passages with `section == ""`,
  `strategy: "fixed"`, and no crash.
- **References truncation.** Text after a `References` line does not appear in any chunk;
  an `Appendix` after it does not resurrect it.
- **Page provenance.** A chunk built from text spanning pages 3–4 reports `[3, 4]`; a chunk
  wholly on page 3 reports `[3, 3]`.
- **Size bounds.** No chunk exceeds the token budget under the injected counter; no chunk is
  under `MIN_CHUNK_CHARS` except a final remainder; consecutive chunks in a section overlap.
- **Determinism / idempotency.** Chunking the same doc twice is byte-identical; a second CLI
  run over an unchanged corpus is a no-op; a bumped `chunker_version` re-chunks.
- **ID format.** `chunk_id == f"{paper_id}:{chunk_index}"`, indices contiguous from 0.

Plus a **corpus smoke check** run by hand at the end (not a pytest): `--report` over all 43
papers shows the strategy split, chunks per paper, and the token-length histogram. A paper
producing 3 chunks or 900 chunks is a detection bug wearing a plausible number.

## Files created / touched

```
rag/chunk.py                     normalize + detect + split + write
rag/tests/test_chunk.py
rag/tests/fixtures/*.json        tiny synthetic extracted-docs
rag/ingest.py                    manifest fields: chunked/chunker_version/chunk_count
scripts/ingest_papers.py         --chunk flag; --report shows chunk + strategy columns
docs/16-rag-chunking.md          this file
```

No new dependency. `chromadb` and `sentence-transformers` stay commented out until Step 3.

## Done when

- `pytest rag/tests` green (Step 1's 18 tests plus the new ones) on a CPU-only checkout with
  no ML deps installed.
- `scripts/ingest_papers.py --chunk` over all 43 papers completes with 0 failures, and
  `--report` shows every paper `chunked` with its strategy.
- The strategy split is **43/43 `sections`** on the current corpus — the measurement above
  says it should be, so anything less is a detection bug to fix, not a fallback to accept.
- Re-running `--chunk` is a pure no-op; bumping `chunker_version` re-chunks everything.
- Spot-check: 5 chunks read as coherent prose with correct section labels and page spans.
- Phase 1 / Phase 2 suites stay green.

## Design decisions

- **Section-aware over fixed-size, with fixed-size as the recorded fallback.** Blind
  512-token windows are the default in every RAG tutorial and they cut mid-argument; a
  paper's sections are semantic units its author already drew for us. But detection can
  fail on a corpus meant to broaden, so the fallback exists, is exercised by a test, and is
  *recorded per paper* rather than being an invisible degradation.
- **The monotonic-run gate, not a bigger regex.** An unanchored header regex over real
  extracted text returns `PaLM`, `OOM`, `FlashAttention`, `Sequence Length` — measured, not
  hypothesized. Sequence validation across the whole document is a stronger and much simpler
  signal than any per-line pattern refinement, because a table cell is not part of an
  ascending run that starts at 1.
- **Normalize before detect, but detect on lines.** Rejoining wrapped lines destroys the
  "header sits alone on a line" signal. The order — de-hyphenate, detect on lines, then
  rejoin within bodies — is load-bearing and easy to get backwards.
- **Drop the bibliography.** It is the highest-density source of false-positive retrievals
  in an academic corpus and it is trivially detectable here (43/43). Keeping appendices is
  the matching call in the other direction: they read like methods because they are.
- **Paper metadata is not copied into chunks on disk.** It joins through `paper_id` at
  insert time. Denormalizing now means every metadata correction is a full re-chunk.
- **Token budget via an injectable counter.** Hard-importing a tokenizer would drag
  `sentence-transformers` into Step 2 and make chunking untestable without ML deps, to buy
  precision that only matters at the boundary the overlap already protects. Step 3 injects
  the real one and the estimate stops mattering.
- **`chunker_version` in the skip condition.** Chunking is the parameter that will be tuned
  most, and the failure mode of a hash-only skip is a corpus half-chunked by two different
  algorithms — invisible, and poisonous to the Step 4 eval. A version bump is the cheap
  invalidation.
- **Chunking is a separate on-disk artifact, not fused into embedding.** Same reasoning that
  split extraction from chunking in Step 1: iterating on the chunker must not re-embed the
  corpus, and inspecting chunks as text is how detection bugs get found.
- **A flag on the existing CLI, not a new script.** One manifest, one report, one entry
  point for the whole pipeline. Separate *status transitions* give the failure isolation;
  a separate script would only duplicate plumbing.

## Next (Step 3)

Local embeddings and the vector store: `all-MiniLM-L6-v2` over `data/chunks/*.json` on an
A6000, ChromaDB persistence with the scalar-flattened metadata described above, and derived
topics — mean-pooled paper vectors clustered into groups labelled by distinguishing TF-IDF
terms, written back as filterable chunk metadata.
