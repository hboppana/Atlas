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

Steps 2–8 LANDED and validated on the A6000 — every op in TinyLlama's forward pass has a
proven GPU counterpart, Step 6 composed them into the full on-device forward pass
(`forward.cu`, measured 2.04e-4 max-abs vs `reference/logits.npy`), Step 7 put the
greedy decode loop on top (`GpuModel::generate()` + `Model::generate()`/`argmax_last_row()`,
token-exact against the CPU oracle, docs/13), and Step 8 exposed it over pybind11 + FastAPI
(docs/14). **Phase 2 is complete**; CTest 7/7 in `build-cuda/`, 10/10 CPU, `pytest
server/tests` 27 passed. No KV cache and no sampling yet: both are named perf/feature
follow-ups (the KV cache is the headline Phase 2 perf item, held to Step 7's token-exact
oracle).

`atlas_cuda` builds with **`CUDA_SEPARABLE_COMPILATION OFF`** (flipped in Step 8): nothing
needs relocatable device code — `reduce.cuh`'s reductions are function templates — and
leaving it ON emits a device-link object referencing `fatbinData`, which cannot resolve
inside a shared module and broke `import _atlas_engine`. Changing that property requires a
clean `build-cuda/`; stale `-dc` objects survive it and fail at link.

| File | Kernel | Approach |
|------|--------|----------|
| `matmul.cu` | tiled matrix multiply (docs/08) | shared-memory tiling, coalesced global access |
| `rmsnorm.cu` | fused RMSNorm (docs/09) | single-pass block-per-row, warp-shuffle reduction |
| `kernels.cu` | utility kernels (docs/10) | embed gather, residual add, SwiGLU, RoPE |
| `attention.cu` | fused causal GQA attention (docs/11) | Q·Kᵀ → softmax → ·V in one launch, block per (query, head), score row in dynamic smem (seq ≤ 12288); flash-style rewrite is the named perf follow-up |
| `reduce.cuh` | — | shared `block_reduce_sum`/`block_reduce_max`, templated on block size (rmsnorm 256, attention 128) |
| `forward.cu` | `GpuModel`, full forward pass + `generate()` (docs/12, docs/13) | 4.4 GB blob uploaded once at `create()`, weights are zero-copy `DeviceTensor::view`s at their host offsets; launch order held to `Model::forward()`. FP32 only (`qweights` ignored); per-launch syncs + per-call scratch kept deliberately — both are named perf follow-ups. Also home to Step 7's `generate()` greedy decode loop (docs/13): forward→argmax-last-row→append→repeat, no KV cache, EOS-terminated, `on_token` streaming hook |

Each kernel is validated in `tests/test_*.cu` against its CPU oracle with a
measured-then-pinned max-abs tolerance (attention: measured 8.94e-08 worst, pinned 1e-6).

- **Correctness before speed.** Match the CPU/HuggingFace reference first, then optimize
  (occupancy, memory coalescing, shared-memory tiling). Benchmark with `scripts/benchmark.py`.
- The CUDA path is **guarded behind a CMake flag** that was off in Phase 1; turn it on here.
- Keep the architecture honest to TinyLlama: RMSNorm (not LayerNorm), RoPE, SwiGLU, GQA.

## Serving layer (Step 8, LANDED — docs/14)

| File | Responsibility |
|------|----------------|
| `engine/bindings/atlas_engine.cpp` | the pybind11 module `_atlas_engine`. Binds `Tokenizer`, `Model`, and `GpuModel` (the last only when built with `ATLAS_USE_CUDA=ON`), plus `has_cuda`/`BOS_ID`/`EOS_ID`. `forward()` and `quantize_int8()` deliberately unbound |
| `server/bridge.py` | the only Python file that knows the module exists. `Engine.load()` once; `generate()`/`stream()` |
| `server/serve.py` | FastAPI — `POST /generate` (SSE), `GET /health`, engine loaded in `lifespan` |
| `server/config.py` | paths, `device` (`auto\|cuda\|cpu`), `max_new_tokens` cap, host/port — all `ATLAS_*` env-overridable |
| `server/tests/` | `test_bridge.py` (imports **no** FastAPI — the layer boundary) + `test_serve.py` |

- Built by `option(ATLAS_BUILD_PYTHON)` into `build*/python/`. **Which `.so` is imported is
  the device selection**: `build/python/` has no `GpuModel`, `build-cuda/python/` does.
  `bridge.py` prefers the CUDA tree; `ATLAS_ENGINE_MODULE` overrides.
- Three binding hazards, all handled and regression-tested: **GIL** (release around
  `generate()`, re-acquire per token; a `None` callback return means keep going),
  **lifetime** (`keep_alive<0,1>` on `GpuModel.create` — it holds a non-owning `Model*`
  into the mmap), **move-only** types (construct via the static factories only).
- **Never decode token-by-token.** `Tokenizer::decode` strips the leading space and
  reassembles `<0xNN>` byte runs — decode the growing prefix and emit the delta, holding
  back anything that ends mid-UTF-8. `test_stream_equals_batch` is the guard.
- One in-flight generation, serialized by a lock (one model, default stream, no batching).
  Client disconnect trips Step 7's `on_token` early stop — measured: a 64-token request
  abandoned after 3 tokens returns in 0.09 s vs ~3.2 s.
- Env on the lab box: conda is broken there, so the Phase 2 deps live in the repo's `.venv`
  (Python 3.10). `PYTHON=` overrides in both scripts.

## Build & test workflow (scripts/)

CUDA builds and runs **directly on the lab box** — `Suramar`, a self-contained Linux server
with 2x NVIDIA RTX A6000 (Ampere, sm_86) attached, no SLURM/scheduler. Runners:

| Script | Job |
|--------|-----|
| `scripts/build_cuda.sh` | LANDED — configure `-DATLAS_USE_CUDA=ON` (+ `-DATLAS_BUILD_PYTHON=ON` when pybind11 is installed) + compile into `build-cuda/` |
| `scripts/test_cuda.sh` | LANDED — `nvidia-smi` + `ctest` over the 7 CUDA tests |
| `scripts/run_server.sh` | LANDED — `python -m server.serve` (uvicorn); honours `ATLAS_DEVICE`, `CUDA_VISIBLE_DEVICES`, `PYTHON` |
| benchmark (later) | run the benchmark suite, log to `results/` |

The Python suites are a **separate `pytest server/tests` invocation**, not part of
`test_cuda.sh`: they need an interpreter with the deps installed and their failure modes are
unrelated to the kernels'. Both they and the blob-gated CUDA tests SKIP green without the
4.4 GB weight blob.

Loop: edit `engine/cuda/` → `scripts/build_cuda.sh` → `scripts/test_cuda.sh` → iterate. The
box is shared and both A6000s are usually busy, so pin a card with `CUDA_VISIBLE_DEVICES=1`
(check `nvidia-smi` first for the freer/cooler one); benchmark numbers will be noisy.
