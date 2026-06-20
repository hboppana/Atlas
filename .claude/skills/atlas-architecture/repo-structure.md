# Atlas — Annotated Repository Structure

The authoritative, per-file map of the Atlas repo. Each phase is tagged so it's clear
which layer a file belongs to. This is the markdown source of truth for the layout
(previously held in `atlas-repo-structure.jsx`, a React visualizer that is now gitignored).

> **Full-stack AI system: C++/CUDA inference → RAG → LangGraph → MCP.**
> From CUDA kernels to autonomous agents — every layer, from scratch.

## Phases

| Phase | Layer | Timeline |
|-------|-------|----------|
| 1 | C++ inference | Weeks 1–3 |
| 2 | CUDA + serving | Weeks 3–5 |
| 3 | RAG pipeline | Weeks 5–6 |
| 4 | LangGraph agent | Weeks 6–7 |
| 5 | MCP server | Weeks 7–8 |

## Tree

```
atlas/                          Full-stack AI system: C++/CUDA inference → RAG → LangGraph → MCP
├── README.md                   Project overview, architecture diagram, build instructions, benchmark results
├── CMakeLists.txt              Build config — handles CPU-only and CUDA paths
├── .gitignore                  Ignore weights/, build/, data/papers/, .env
│
├── engine/                     [Phase 1] C++/CUDA inference engine — the foundation layer
│   ├── include/                Header files for the inference engine
│   │   ├── model.h             Model struct — layer weights, hyperparams, forward() signature
│   │   ├── tokenizer.h         BPE tokenizer — encode/decode, vocab loading
│   │   ├── tensor.h            Lightweight tensor class — shape, strides, data pointer, basic ops
│   │   └── quantize.h          INT8 quantization — scale/zero-point computation, dequant
│   ├── src/                    C++ implementation files
│   │   ├── model.cpp           CPU forward pass — embedding lookup, attention, FFN, layer norm
│   │   ├── tokenizer.cpp       BPE merge logic, special token handling, vocab from binary file
│   │   ├── tensor.cpp          Tensor memory management, reshape, mmap weight loading
│   │   ├── quantize.cpp        Post-training quantization — FP32 → INT8 weight conversion
│   │   └── main.cpp            CLI entry point — load model, run inference, print tokens
│   ├── cuda/                   [Phase 2] CUDA kernels — the performance layer
│   │   ├── matmul.cu           Tiled matrix multiply — shared memory, coalesced access, A6000 tuned
│   │   ├── attention.cu        Fused attention kernel — Q·K^T softmax V in one launch
│   │   ├── layernorm.cu        Fused layer norm — single-pass mean+variance, warp-level reduction
│   │   └── kernels.cu          Utility kernels — GELU, residual add, embedding lookup
│   └── tests/                  Correctness tests against HuggingFace reference
│       ├── test_tokenizer.cpp  BPE encode/decode round-trip, edge cases (empty, unicode)
│       ├── test_forward.cpp    Compare output logits vs HuggingFace within tolerance
│       └── test_quantize.cpp   Quantized vs full-precision accuracy delta
│
├── server/                     [Phase 2] Python bridge + FastAPI serving layer
│   ├── __init__.py
│   ├── bridge.py               pybind11 bridge — exposes C++ engine.generate() to Python
│   ├── serve.py                FastAPI server — /generate endpoint, streaming token output
│   └── config.py               Model path, max tokens, temperature, device selection
│
├── rag/                        [Phase 3] RAG pipeline — document ingestion to retrieval
│   ├── __init__.py
│   ├── ingest.py               PDF parsing + chunking — section-aware splits (abstract, methods, results)
│   ├── embed.py                Local embeddings via sentence-transformers (all-MiniLM-L6-v2), runs on GPU
│   ├── store.py                ChromaDB vector store — insert, index, persist to disk
│   └── retrieve.py             Semantic retrieval + reranking — top-k chunks with MMR diversity
│
├── agent/                      [Phase 4] LangGraph multi-step agent orchestration
│   ├── __init__.py
│   ├── graph.py                LangGraph state machine — node transitions and conditional edges
│   ├── nodes.py                Agent nodes: retrieve, evaluate_relevance, synthesize, cite_sources
│   ├── tools.py                Tool definitions — search_papers, query_corpus, summarize, compare
│   └── state.py                TypedDict state schema — query, retrieved_docs, reasoning_steps, output
│
├── mcp/                        [Phase 5] MCP server — exposes Atlas as a tool to any MCP client
│   ├── __init__.py
│   ├── server.py               MCP entrypoint — tool registration, request handling, stdio transport
│   ├── tools.py                Registered tools: search_papers, ask_corpus, ingest_document, run_inference
│   └── config.py               Server config — model path, corpus path, embedding model, port
│
├── scripts/                    Utility scripts for setup, benchmarking, data prep
│   ├── download_weights.py     Pull model weights from HuggingFace (TinyLlama-1.1B-Chat)
│   ├── convert_weights.py      Convert .safetensors → raw binary for C++ mmap loading
│   ├── validate.py             Compare Atlas output vs HuggingFace for correctness verification
│   ├── benchmark.py            Measure tokens/sec, latency, memory usage across all phases
│   ├── build_cuda.sh           Configure + compile CUDA into build-cuda/ on the lab A6000 box
│   ├── test_cuda.sh            Run the CUDA bring-up test (ctest -R test_device) on the A6000 box
│   └── ingest_papers.py        Bulk ingest arXiv PDFs on federated learning / differential privacy
│
├── data/                       Paper corpus and processed chunks (gitignored except structure)
│   ├── papers/                 Raw PDFs from arXiv / Semantic Scholar
│   └── chunks/                 Processed text chunks with metadata
│
├── weights/                    Model weights — gitignored, populated by download_weights.py
├── requirements.txt            Python deps — fastapi, langchain, langgraph, chromadb, sentence-transformers, mcp
└── pyproject.toml              Project metadata, build config, optional CUDA extras
```

## Stack

C++ · CUDA · Python · FastAPI · pybind11 · LangGraph · ChromaDB · sentence-transformers ·
MCP · dual NVIDIA RTX A6000 (Linux)
