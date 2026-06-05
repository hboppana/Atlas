---
name: atlas-architecture
description: Atlas project north star — the from-scratch ethos, the 5-phase build plan, the repository layout, and the non-negotiable conventions (conda envs, TinyLlama target, validate-everything-against-HuggingFace). Use when orienting on Atlas, starting any new piece of work, deciding which directory code belongs in, picking dependencies, or when a request spans more than one phase.
---

# Atlas — Architecture & Conventions

Atlas is a **full-stack AI research assistant for scholarly publications, built from the
metal up**. Every layer is implemented from scratch — nothing is outsourced to a hosted
LLM API. It runs its own transformer inference engine (hand-written C++/CUDA), retrieves
evidence from a vector store of ingested PDFs, orchestrates multi-step reasoning with
LangGraph, and exposes the whole system through the Model Context Protocol (MCP).

```
C++/CUDA inference engine  →  Python serving  →  RAG  →  LangGraph agent  →  MCP server
```

## The 5 phases (~8 weeks)

| Phase | Layer | Directory | Skill |
|-------|-------|-----------|-------|
| 1 | C++ inference engine (CPU baseline) | `engine/` | `atlas-cpp-engine` |
| 2 | CUDA kernels + FastAPI serving | `engine/cuda/`, `server/`, `slurm/` | `atlas-cuda-serving` |
| 3 | RAG pipeline | `rag/` | `atlas-rag` |
| 4 | LangGraph agent | `agent/` | `atlas-agent` |
| 5 | MCP server | `mcp/` | `atlas-mcp` |

When working inside a phase, read that phase's skill for the detailed specs and conventions.

## Non-negotiable conventions

- **From-scratch ethos.** The transformer inference engine itself is built by hand. Do not
  reach for a hosted LLM API or a high-level inference library (vLLM, llama.cpp, ggml,
  ONNXRuntime) to do the model's job. Building it *is* the project. (High-level libraries
  are fine for the surrounding plumbing — FastAPI, ChromaDB, LangGraph, MCP.)
- **Phase by phase, deeply.** Fully build and validate one phase before starting the next.
  Do **not** scaffold all five phases up front. Each phase produces working, tested code
  before the next begins.
- **Validate against HuggingFace ground truth.** Phase 1 Step 0 captured golden oracles in
  `reference/` (`config.json`, `prompt.txt`, `token_ids.npy`, `logits.npy`) from
  TinyLlama-1.1B-Chat in FP32 on CPU. Every from-scratch component is checked against these
  oracles within tolerance. Correctness first, performance second.
- **Model target is TinyLlama-1.1B-Chat-v1.0**, a modern Llama-architecture model
  (RoPE, RMSNorm, SwiGLU, GQA 8:1). **Not GPT-2.** Pinned hyperparameters live in
  `reference/config.json`.
- **Python env management is conda.** Set up environments with a named conda env, then
  `pip install -r requirements.txt` inside it. Do not default to bare venv/pip.
  ```powershell
  conda create -n atlas python=3.11 -y
  conda activate atlas
  pip install -r requirements.txt
  ```
- **`requirements.txt` grows phase by phase.** Phase 2–5 deps are present but commented
  out; uncomment them only when that phase starts.

## Dev environment & hardware

- **Phase 1 is CPU-only and built locally on this Windows machine** (MSVC, C++17, FP32).
- **CUDA work (Phase 2+) targets HiPerGator** — the UF cluster, NVIDIA A6000, SLURM. CUDA
  builds and GPU benchmarks are pushed there via the scripts in `slurm/`.
- The corpus theme is **federated learning / differential privacy** papers (arXiv /
  Semantic Scholar).

## Repository layout

```
atlas/
  engine/         Phase 1 — C++/CUDA inference engine (include/, src/, cuda/, tests/)
  server/         Phase 2 — pybind11 bridge + FastAPI serving
  rag/            Phase 3 — ingest, embed, store, retrieve
  agent/          Phase 4 — LangGraph graph, nodes, tools, state
  mcp/            Phase 5 — MCP server + registered tools
  scripts/        download_weights, convert_weights, validate, benchmark, ingest_papers
  slurm/          HiPerGator SLURM jobs (build_cuda, benchmark, embed_corpus)
  data/           papers/ + chunks/ (gitignored contents)
  weights/        model weights (gitignored, populated by download_weights.py)
  reference/      golden test oracles (NOT gitignored — small, used as test fixtures)
  docs/           per-step design docs (e.g. 01-tensor-foundation.md)
```

The authoritative, annotated layout is `atlas-repo-structure.jsx` at the repo root; the
per-step design docs live in `docs/`. When in doubt about where something goes or what a
file is responsible for, consult those before inventing new structure.
