---
name: atlas-cuda-serving
description: Phase 2 — CUDA inference kernels plus the Python serving layer (engine/cuda/, server/, slurm/). Covers the fused A6000-tuned kernels, the pybind11 bridge, the FastAPI streaming endpoint, and the HiPerGator/SLURM build-and-benchmark workflow. Use when writing or reviewing CUDA kernels, the C++↔Python bridge, the FastAPI server, or SLURM job scripts.
---

# Atlas Phase 2 — CUDA Kernels + Serving

Make the proven CPU engine fast on GPU, then expose it over HTTP. **Prerequisite:** Phase 1
is complete and the CPU forward pass matches `reference/logits.npy`. Read
`atlas-architecture` first. CUDA kernels replace CPU ops *without changing results* — every
kernel is validated against the same `reference/` oracles within tolerance.

## CUDA kernels (engine/cuda/) — tuned for NVIDIA A6000

| File | Kernel | Approach |
|------|--------|----------|
| `matmul.cu` | tiled matrix multiply | shared-memory tiling, coalesced global access |
| `attention.cu` | fused attention | Q·Kᵀ → softmax → ·V in a single launch (GQA-aware, 8:1) |
| `layernorm.cu` | fused RMSNorm | single-pass mean+variance, warp-level reduction |
| `kernels.cu` | utility kernels | GELU/SwiGLU activation, residual add, embedding lookup |

- **Correctness before speed.** Match the CPU/HuggingFace reference first, then optimize
  (occupancy, memory coalescing, shared-memory tiling). Benchmark with `scripts/benchmark.py`.
- The CUDA path is **guarded behind a CMake flag** that was off in Phase 1; turn it on here.
- Keep the architecture honest to TinyLlama: RMSNorm (not LayerNorm), RoPE, SwiGLU, GQA.

## Serving layer (server/)

| File | Responsibility |
|------|----------------|
| `bridge.py` | pybind11 bridge — exposes the C++ `engine.generate()` to Python |
| `serve.py` | FastAPI server — `/generate` endpoint with **streaming** token output |
| `config.py` | model path, max tokens, temperature, device selection |
| `__init__.py` | package marker |

- The bridge is **pybind11** (already a planned dep). The Python side never reimplements
  inference — it calls into the C++/CUDA engine.
- `/generate` **streams** tokens as they are produced (don't buffer the whole completion).
- Uncomment the Phase 2 deps in `requirements.txt`: `fastapi`, `uvicorn`, `pybind11`.

## HiPerGator / SLURM workflow (slurm/)

CUDA is **not** built on the local Windows machine — it is pushed to HiPerGator (UF cluster,
A6000, SLURM). Jobs:

| Script | Job |
|--------|-----|
| `build_cuda.sh` | compile CUDA kernels on a GPU node |
| `benchmark.sh` | run the full benchmark suite, log to `results/` |
| `embed_corpus.sh` | (used in Phase 3) embed the paper corpus on GPU |

Typical loop: develop kernels locally → `sbatch slurm/build_cuda.sh` on HiPerGator →
`sbatch slurm/benchmark.sh` → inspect tokens/sec, latency, memory in `results/`.
