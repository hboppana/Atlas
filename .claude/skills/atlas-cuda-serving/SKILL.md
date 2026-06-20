---
name: atlas-cuda-serving
description: Phase 2 — CUDA inference kernels plus the Python serving layer (engine/cuda/, server/, scripts/build_cuda.sh + test_cuda.sh). Covers the fused A6000-tuned kernels, the pybind11 bridge, the FastAPI streaming endpoint, and the local build-and-test workflow on the dual-A6000 lab box (no SLURM). Use when writing or reviewing CUDA kernels, the C++↔Python bridge, the FastAPI server, or the CUDA build/test scripts.
---

# Atlas Phase 2 — CUDA Kernels + Serving

Make the proven CPU engine fast on GPU, then expose it over HTTP. **Prerequisite:** Phase 1
is complete and the CPU forward pass matches `reference/logits.npy`. Read
`atlas-architecture` first. CUDA kernels replace CPU ops *without changing results* — every
kernel is validated against the same `reference/` oracles within tolerance.

## Step 1 — bring-up infra (LANDED, docs/07) — before any kernel

Phase 2 splits infra from the first kernel so failure domains stay apart. Step 1 writes
**no compute kernel** — it stands up the scaffolding every kernel rides on, proven with a
no-math round-trip. What exists in `engine/cuda/`:

- `device_tensor.h` / `.cu` — `DeviceTensor`: the device mirror of `Tensor` (owns-or-view,
  **move-only**, row-major strides). `alloc`/`view`/`numel`; `to_device`/`to_host` copy
  helpers (out-param style; host `Tensor` is the source of truth for shape/stride). Header
  stays host-includable. Also holds the Step-1-only `launch_scale` payload kernel.
- `cuda_check.h` — `CUDA_CHECK(expr)` (assert-don't-handle: report + `std::abort`) and
  `CUDA_CHECK_KERNEL()` (poll `cudaGetLastError` + sync after a launch). CUDA-only, so it
  lives apart from `device_tensor.h`. **Wrap every CUDA runtime call.**
- `tests/test_device.cu` — round-trip/identity (bit-exact), the **reusable diff harness**
  (`compare()` → max-abs/mean-abs; *measure on first green run, then pin* — Step 2 drops in
  `linear()` as oracle unchanged), and a CUDA_CHECK-fires death case (self-subprocess).
- Build is **guarded**: top `CMakeLists.txt` `if(ATLAS_USE_CUDA)` → `enable_language(CUDA)`,
  `CMAKE_CUDA_ARCHITECTURES 86`, `add_subdirectory(engine/cuda)`. CPU build (10/10 CTest)
  stays green with the flag OFF; turning it ON on this Windows box fails fast (no nvcc) — by
  design. `engine/cuda/CMakeLists.txt` builds `atlas_cuda` (links `atlas_engine`) + `test_device`.

## CUDA kernels (engine/cuda/) — tuned for NVIDIA A6000

Step 2 onward. First kernel: tiled **matmul** (`y = x @ Wᵀ`, oracle `atlas::linear()`),
riding the Step-1 infra; design in memory `phase2-matmul-kernel-plan`.

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

## Build & test workflow (scripts/)

CUDA builds and runs **directly on the lab box** — `Suramar`, a self-contained Linux server
with 2x NVIDIA RTX A6000 (Ampere, sm_86) attached, no SLURM/scheduler. Runners:

| Script | Job |
|--------|-----|
| `scripts/build_cuda.sh` | LANDED — configure `-DATLAS_USE_CUDA=ON` + compile into `build-cuda/` |
| `scripts/test_cuda.sh` | LANDED — `nvidia-smi` + `ctest -R test_device` (widens as kernel tests join) |
| benchmark (later) | run the benchmark suite, log to `results/` |

Loop: edit `engine/cuda/` → `scripts/build_cuda.sh` → `scripts/test_cuda.sh` → iterate. The
box is shared and both A6000s are usually busy, so pin a card with `CUDA_VISIBLE_DEVICES=1`
(check `nvidia-smi` first for the freer/cooler one); benchmark numbers will be noisy.
