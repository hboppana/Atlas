---
name: atlas-rag
description: Phase 3 — the RAG pipeline (rag/), from PDF ingestion to retrieval. Covers section-aware chunking, local sentence-transformer embeddings, the ChromaDB vector store, and MMR retrieval over a federated-learning / differential-privacy paper corpus. Use when writing or reviewing rag/ (ingest, embed, store, retrieve) or scripts/ingest_papers.py.
---

# Atlas Phase 3 — RAG Pipeline

Turn a corpus of academic PDFs into retrievable evidence. **Prerequisite:** the inference
engine (Phases 1–2) works. Read `atlas-architecture` first. Everything is **local** — local
embeddings, a local vector store — consistent with the from-scratch, no-hosted-API ethos.

Corpus theme: **federated learning / differential privacy** papers (arXiv / Semantic Scholar).

## Pipeline (rag/)

| Stage | File | What it does |
|-------|------|--------------|
| Ingest | `ingest.py` | PDF parsing + **section-aware** chunking — split on abstract / methods / results rather than blind fixed-size windows; carry metadata |
| Embed | `embed.py` | local embeddings via **sentence-transformers `all-MiniLM-L6-v2`**, runs on GPU |
| Store | `store.py` | **ChromaDB** vector store — insert, index, persist to disk |
| Retrieve | `retrieve.py` | semantic retrieval + reranking — top-k chunks with **MMR** (maximal marginal relevance) for diversity |

- `scripts/ingest_papers.py` bulk-ingests arXiv PDFs into this pipeline.
- Raw PDFs live in `data/papers/`, processed chunks in `data/chunks/` — **both gitignored**.
  Chunks carry metadata (source paper, section) so retrieval results stay citable, which the
  Phase 4 agent depends on for `cite_sources`.
- Embeddings can be generated on HiPerGator via `slurm/embed_corpus.sh` for the full corpus.
- Uncomment the Phase 3 deps in `requirements.txt`: `chromadb`, `sentence-transformers`,
  `pypdf`.

## Conventions

- **Section-aware over naïve splitting.** The chunker should respect paper structure; this is
  the deliberate quality lever for scholarly retrieval.
- **MMR, not plain top-k.** Retrieval optimizes for relevance *and* diversity so the agent
  sees complementary evidence rather than near-duplicate passages.
- Keep embedding and retrieval **local and reproducible** — no external embedding APIs.
