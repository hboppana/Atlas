---
name: atlas-cpp-engine
description: Phase 1 — the from-scratch C++ TinyLlama inference engine (engine/). Covers the TinyLlama-1.1B architecture, the Tensor foundation conventions, the MSVC/C++17/CMake CPU-only build, the zero-dependency test harness, and the validate-against-reference/ discipline. Use when writing or reviewing anything in engine/ (tensor, tokenizer, model, quantize, tests) or the CMake build.
---

# Atlas Phase 1 — C++ Inference Engine

Build a CPU-only, FP32 TinyLlama forward pass from scratch in C++, validated against the
HuggingFace golden oracles in `reference/`. This is the foundation everything else sits on.
Read `atlas-architecture` first for the project-wide ethos.

## Build order (do them in this sequence, validate each)

1. **Tensor foundation** + CMake build — the data structure and core ops. *(documented in
   `docs/01-tensor-foundation.md`)*
2. **Tokenizer** — BPE/SentencePiece encode/decode, validated against `reference/token_ids.npy`. *(done)*
3. **Model + weight loading** — `convert_weights.py` (`.safetensors` → raw binary) plus
   mmap loading; the full forward pass. *(**done** — `test_forward` green: top-1 id 3681
   `▁Paris`, logit 13.3885; see Weight blob contract below and `docs/03-model-weights.md`)*
4. **Forward-pass validation** — a C++ `.npy` reader; compare logits to `reference/logits.npy`
   within tolerance.
5. **Quantize** — post-training FP32 → INT8, measure the accuracy delta.

Don't pull steps forward. Each row of the table is its own focused, validated unit.

## TinyLlama-1.1B-Chat-v1.0 architecture (pinned in reference/config.json)

Modern Llama architecture — **not GPT-2**:

| Param | Value | Note |
|-------|-------|------|
| hidden_size | 2048 | |
| intermediate_size | 5632 | SwiGLU FFN |
| num_hidden_layers | 22 | |
| num_attention_heads | 32 | |
| num_key_value_heads | 4 | **GQA 8:1** — 8 query heads share each KV head |
| head_dim | 64 | hidden / heads |
| vocab_size | 32000 | |
| max_position_embeddings | 2048 | |
| rms_norm_eps | 1e-5 | |
| rope_theta | 10000.0 | |

Architectural building blocks to implement: **RoPE** (rotary position embeddings),
**RMSNorm** (not LayerNorm), **SwiGLU** FFN, **GQA** attention. The tokenizer prepends
**BOS (id 1)** by default — the C++ side must match. Validation uses the raw prompt
`"The capital of France is"` (no chat template yet).

## Tensor conventions (the load-bearing decisions)

From `docs/01-tensor-foundation.md` — match these:

- **FP32 only** in Phase 1. No dtype templating until the quantize step.
- **Owns *or* views.** `Tensor` holds `float* data`, `shape`, `strides`, and an `owns`
  flag. Owned = allocated activations (freed in destructor); viewing = a non-owning window
  over mmap'd weights. This is what lets weight loading avoid copying ~2 GB. **Move-only;
  no accidental deep copies.**
- **Row-major, contiguous, explicit strides** — reshape/slice are free (no copy).
- **Core ops are free functions** in `tensor.h` (`matmul`, elementwise `add`/`mul`),
  **out-param style** so allocations are explicit: `void matmul(const Tensor& a, const
  Tensor& b, Tensor& out);`. Model-specific math (RMSNorm, RoPE, attention, SwiGLU) is
  **built from** these primitives and lives in `model.cpp`, not `tensor.cpp`. The split is
  deliberate: tensor = memory/ops, model = the network.

## Build & test

- **Toolchain: MinGW-w64 GCC (MSYS2 UCRT64, `C:\msys64\ucrt64\bin`), C++17, CPU-only.** No
  MSVC on this machine; the CMake files are generator-agnostic so MSVC stays optional. The
  CUDA path is guarded off (`ATLAS_USE_CUDA=OFF`) for all of Phase 1. Full build reference
  and gotchas: `docs/build-setup.md`.
- Top-level `CMakeLists.txt` adds `engine/`; `engine/CMakeLists.txt` builds the
  `atlas_engine` static lib + `test_tensor` (registered with CTest; `-static*` link flags
  under `if (MINGW)` so the exe runs without UCRT64 on PATH at runtime).
- **Zero-dependency test harness** — a tiny `CHECK(...)` assert macro, **no
  GoogleTest/Catch2**. Less build friction, fits the from-scratch ethos.
- Definition of done for a step = `cmake --build` succeeds and the step's test passes via
  `ctest` (checked against hand-computed values and/or the `reference/` oracles).

```powershell
# UCRT64 must be on PATH so g++ can load its runtime DLLs (else CMake says "compiler broken")
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
cmake -S . -B build -G "MinGW Makefiles"   # single-config: no --config
cmake --build build
ctest --test-dir build --output-on-failure
```

- **`CMAKE_BUILD_TYPE` defaults to Release** (set in the top-level CMakeLists): the naive
  FP32 forward needs `-O3` to run in ~11 s instead of minutes. Debug (asserts on) must be
  requested explicitly.
- **MinGW gotcha: 32-bit `stat()` fails on files >2 GB** (the 4.4 GB weight blob) —
  check existence with `std::ifstream(path).good()`, size with `GetFileSizeEx`.

**conda boundary:** build the C++ engine with the **system MSYS2 GCC**, not conda — on
Windows conda only offers an MSVC wrapper (needs VS) or a stale GCC 8.x. conda is for the
**Python** layers (Step 0, and Phase 2+ incl. `pybind11`). See [[atlas-architecture]].

## Files (engine/)

| File | Responsibility |
|------|----------------|
| `include/tensor.h` / `src/tensor.cpp` | Tensor class + core ops; memory mgmt, reshape, mmap weight loading |
| `include/tokenizer.h` / `src/tokenizer.cpp` | **done** — BPE merge logic, special tokens, byte fallback; loads the plain-text `reference/tokenizer/{vocab,merges}.txt` exported by `scripts/export_tokenizer.py` (not a binary blob — deliberate deviation, see `docs/02-tokenizer.md`) |
| `include/model.h` / `src/model.cpp` | **done** — `WeightStore` (Win32 mmap + manifest views), `linear` (y = x@Wᵀ), RMSNorm, RoPE (NeoX half-split), GQA causal attention, SwiGLU; full prefill forward |
| `include/quantize.h` / `src/quantize.cpp` | INT8 quant — scale/zero-point, FP32→INT8 |
| `src/main.cpp` | **done** — CLI (`atlas.exe`): load tokenizer + model, encode prompt (argv[1] or the reference default), forward, greedy-decode, print |
| `tests/test_tensor.cpp` `test_tokenizer.cpp` `test_forward.cpp` `test_quantize.cpp` | Correctness vs `reference/` |

## Weight blob contract (`scripts/convert_weights.py`, Step 3.1 — done)

`convert_weights.py` is **numpy-only** (no torch/safetensors dep): it parses the
safetensors header by hand and upcasts BF16 → FP32 with `(u16 << 16).view(f32)` — exact
for normals, subnormals, inf, nan (verified against an independent ldexp decode). It
writes two gitignored files under `weights/tinyllama-1.1b-chat/`:

- `model.f32.bin` — 201 tensors, FP32 little-endian, concatenated in a fixed order
  (embed → layers 0..21 × 9 → `model.norm` → `lm_head`). 4.40 GB / 4400193536 bytes.
- `model.manifest.txt` — 201 lines, `name byte_offset ndim d0 d1 ...`. Offsets are byte
  offsets into the blob and are **contiguous** (manifest total == blob size). Weights keep
  PyTorch `[out, in]` layout — the C++ `linear` helper computes `y = x @ Wᵀ`, no transpose
  at convert time. The C++ `WeightStore` mmaps the blob and builds non-owning `Tensor::view`s
  by name from the manifest (the `owns=false` payoff). On Windows use Win32
  `CreateFileMappingA`/`MapViewOfFile`, isolated behind one `#ifdef _WIN32` so a POSIX `mmap`
  path drops in for Phase 2. `test_forward` SKIPs green if the blob is absent (it's regenerated
  locally, never committed).

## Reference oracles (the source of truth)

Produced by `scripts/download_weights.py`, consumed by `engine/tests/`. Do not regenerate
casually — they are the contract:

- `reference/config.json` — pinned hyperparameters (table above).
- `reference/token_ids.npy` — oracle for `test_tokenizer.cpp` (BOS id 1 prepended).
- `reference/logits.npy` — `[seq, vocab]` FP32 oracle for `test_forward.cpp`, prompt
  "The capital of France is". A `Paris`-topped top-5 confirms sanity.
