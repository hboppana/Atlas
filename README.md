# Atlas

An AI research assistant for scholarly publications — built from the metal up.

Atlas is a full-stack AI system where every layer is implemented from scratch, from
hand-written CUDA inference kernels to an autonomous agent that reasons over a corpus
of academic papers. Instead of calling a hosted LLM API, Atlas runs its own transformer
inference engine, retrieves evidence from a vector store of ingested PDFs, orchestrates
multi-step reasoning with LangGraph, and exposes the whole thing to any client through
the Model Context Protocol (MCP).

## Architecture

```
C++/CUDA inference engine  →  Python serving  →  RAG  →  LangGraph agent  →  MCP server
```

| Phase | Layer | What it does |
|-------|-------|--------------|
| 1 | **Inference engine** (`engine/`) | C++ transformer forward pass — BPE tokenizer, tensor ops, INT8 quantization, CPU baseline validated against HuggingFace |
| 2 | **CUDA + serving** (`engine/cuda/`, `server/`) | Fused CUDA kernels (matmul, attention, layernorm) + pybind11 bridge + FastAPI streaming endpoint |
| 3 | **RAG pipeline** (`rag/`) | Section-aware PDF chunking, local sentence-transformer embeddings, ChromaDB vector store, MMR retrieval |
| 4 | **LangGraph agent** (`agent/`) | Stateful multi-step graph: retrieve → evaluate relevance → synthesize → cite sources |
| 5 | **MCP server** (`mcp/`) | Exposes Atlas as tools (`search_papers`, `ask_corpus`, `ingest_document`, `run_inference`) to any MCP client |

## Stack

C++ · CUDA · Python · FastAPI · pybind11 · LangGraph · ChromaDB · sentence-transformers · MCP

Built and benchmarked on HiPerGator (NVIDIA A6000) via SLURM.

## Setup

Create the project environment and install dependencies:

```powershell
conda create -n atlas python=3.11 -y
conda activate atlas
pip install -r requirements.txt
```

### Phase 1: establish ground truth

Download TinyLlama-1.1B-Chat and capture the golden reference forward pass that
the C++ engine is validated against:

```powershell
python scripts/download_weights.py
```

This writes the test oracles (`config.json`, `token_ids.npy`, `logits.npy`) to
`reference/`. A `Paris`-topped top-5 prediction confirms the reference is sane.

## Status

Early scaffolding. See `atlas-repo-structure.jsx` for the full planned repository layout
and phased build plan.
