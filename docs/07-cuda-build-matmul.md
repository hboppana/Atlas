# Phase 2 · Step 1 — CUDA bring-up (build path, local GPU loop, `DeviceTensor`, validation harness)

> Status: **done** — infra code landed (`engine/cuda/`, `scripts/build_cuda.sh`, `scripts/test_cuda.sh`); the `test_device` round-trip passes on the lab A6000 box (no SLURM), and the CPU build stays green at 10/10 with `ATLAS_USE_CUDA=OFF`. Step 2 (matmul) is next.
> Predecessor: Phase 1 Addendum — test-suite hardening — **done** ([06-phase1-test-hardening.md](06-phase1-test-hardening.md))
> Successor: Step 2 — the tiled **matmul** kernel, validated against the CPU `linear` oracle (first kernel to ride this infra)

## Goal

Make the proven CPU engine *fast on GPU* — but the first step buys none of the speed, and
writes **no kernel**. It stands up the scaffolding every later kernel rides on, and nothing
more. Splitting infrastructure from the first kernel keeps the two failure domains apart:
when Step 2's matmul is wrong, it's the *kernel*, because the build path, the host↔device
copies, and the validation method were already proven green here on their own.

Step 1 delivers four things:

1. **The CUDA build path** — flip `ATLAS_USE_CUDA` on, enable the `CUDA` language in
   CMake, compile `engine/cuda/` with `nvcc` for the A6000 (sm_86).
2. **The local GPU loop** — edit, build, and test directly on the lab box's A6000s. No
   scheduler, no remote round-trip; the CPU build still works anywhere with the flag off.
3. **A device-memory primitive** (`DeviceTensor`) — `cudaMalloc`/H2D/D2H with the same
   `owns` + move-only discipline as `tensor.h`, so device buffers can't silently leak or
   double-free.
4. **A kernel-validation harness** — the pattern, reused by every kernel after this, for
   checking a GPU result against its CPU oracle within a measured tolerance.

Proving the build/copy/harness stack with **zero kernel math** needs a trivial
load-bearing payload — see "Proving the infra with a no-math payload" below.

### Scope boundary (what Step 1 is *not*)

- **Not** any compute kernel — no matmul, attention, RMSNorm, or activations. The matmul
  is **Step 2** (design captured separately, written up then).
- **Not** a full-model GPU forward pass — a much later step, validated end-to-end against
  `reference/logits.npy`.
- **Not** the pybind11 bridge or FastAPI server — that is the serving sub-phase.
- **Not** any performance work or perf pins. The only Step-1 bar is *the toolchain builds
  and a round-trip of bytes through device memory is bit-exact*.
- **Not** INT8/TF32 on GPU. The Phase 1 INT8 path stays CPU-only for now.

## Device-memory primitive (`engine/cuda/device_tensor.h`)

A minimal mirror of `tensor.h` for device memory — enough to move data in, run a kernel,
and read it back, no more:

- `cudaMalloc` on construction from a shape; `owns == true` frees in the destructor.
- A non-owning view path (`owns == false`) for the eventual case of slicing the mmap'd
  weights' device copy, mirroring `tensor.h`'s `view`.
- **Move-only.** The device pointer is unique; copying must not silently duplicate a
  `cudaMalloc` (the same discipline `WeightStore` enforces for the OS mapping handle).
- `to_device(const Tensor&)` / `to_host(Tensor&)` thin wrappers over `cudaMemcpy`
  H2D/D2H. The CPU `Tensor` stays the single source of truth for shape/stride.
- Every CUDA API call goes through a `CUDA_CHECK(...)` macro that aborts on a non-`Success`
  status — the **assert-don't-handle** ethos from Phase 1 (a failed `cudaMalloc` or a
  kernel launch error is a bug, not a recoverable condition).

## Validation harness — the pattern every later kernel reuses

The model-math building blocks were factored out of `forward()` in Phase 1 (docs/06)
**for the next moment**: each CUDA kernel is checked against its CPU counterpart, one
component at a time. Step 1 establishes the *mechanism* for that comparison so Step 2 only
has to supply the kernel and its oracle.

The harness provides:

- A host-side test driver under `engine/cuda/tests/` that fills inputs with a deterministic
  PRNG, runs the device payload, copies the result back via `DeviceTensor`, and compares
  element-wise against a CPU reference.
- `max-abs` / `mean-abs` diff reporting, and the convention (from Phase 1) of **measuring
  the diff on the first green run and then pinning it** as the gate — never guessing a
  tolerance up front.
- The blob-free discipline: tests run on any GPU node without the 4.4 GB weight blob.

## Proving the infra with a no-math payload

Step 1 ships **no compute kernel**, so the toolchain/copy/harness stack is proven with the
most trivial device payload possible — a **round-trip / identity** check:

1. Build a CPU `Tensor` of known values.
2. `to_device` → (optionally a trivial elementwise copy or `+0` kernel that touches every
   element, just to exercise a launch) → `to_host`.
3. Assert the result is **bit-exact** to the input.

This is enough to prove: the `nvcc` build works on the A6000, `cudaMalloc`/`cudaMemcpy`
round-trips correctly, `CUDA_CHECK` fires on injected failure, a kernel can be launched,
and the harness reports a diff. It deliberately carries **no math**, so a failure can only
mean the *infrastructure*, not a kernel. (A trivial scale-by-constant variant gives the
harness a non-zero expected output to diff against without introducing real algorithmic
risk.)

## Build & test workflow

### CMake (`engine/cuda/CMakeLists.txt`, guarded by `ATLAS_USE_CUDA`)

- Top-level `CMakeLists.txt` already declares `option(ATLAS_USE_CUDA ... OFF)`. Step 1
  adds, under that guard: `enable_language(CUDA)`, `set(CMAKE_CUDA_ARCHITECTURES 86)`
  (A6000 = Ampere, sm_86), and `add_subdirectory(engine/cuda)`.
- The CPU build **stays green with the flag off** — nothing CUDA is referenced unless
  `ATLAS_USE_CUDA=ON`. The flag is turned on only on a CUDA-capable box.
- CUDA compiles with `nvcc`; the host-side glue stays C++17.

### `scripts/build_cuda.sh` / `scripts/test_cuda.sh`

Plain runners (no scheduler) for the lab A6000 box: `build_cuda.sh` configures with
`-DATLAS_USE_CUDA=ON` into `build-cuda/` and `cmake --build`s it; `test_cuda.sh` prints
`nvidia-smi` then runs `ctest -R test_device`. The box is shared and both A6000s are
usually busy, so pin a card: `CUDA_VISIBLE_DEVICES=1 scripts/test_cuda.sh`.

### The loop

```
edit engine/cuda/  →  scripts/build_cuda.sh   # configure + compile into build-cuda/
                   →  scripts/test_cuda.sh     # run the round-trip test, report pass
                   →  iterate
```

The only portable check is that the CPU build/test suite still passes with
`ATLAS_USE_CUDA=OFF`.

## Design decisions

- **Infra and the first kernel are separate steps.** A single GPU run that fails should
  point at exactly one thing. By proving the build path, `DeviceTensor`, copy path, and
  harness with a no-math payload, Step 2's matmul failure is unambiguously a *kernel*
  failure — not a confounded build/copy/kernel mystery.
- **`DeviceTensor` mirrors `tensor.h` conventions** (owns flag, move-only, view path) so
  the GPU side feels like the CPU side and the same leak/double-free guarantees hold for
  device memory.
- **Assert-don't-handle on the CUDA boundary.** `CUDA_CHECK` aborts; a bad `cudaMalloc`
  or launch error is a bug, consistent with the Phase 1 `die()` ethos.
- **Tolerance is measured, then pinned** — the harness bakes in the Phase 1 method now, so
  every kernel step inherits it for free.
- **Guarded build keeps Windows green.** The CUDA path is invisible to the local CPU
  build until the flag is on, so Phase 1's 10/10 suite is never disturbed by Phase 2.

## Files created / touched

| File | Change |
|------|--------|
| `CMakeLists.txt` (top) | under `ATLAS_USE_CUDA`: `enable_language(CUDA)`, `CMAKE_CUDA_ARCHITECTURES 86`, `add_subdirectory(engine/cuda)` |
| `engine/cuda/CMakeLists.txt` | new — builds the CUDA target + its test, guarded |
| `engine/cuda/device_tensor.h` (+ `.cu`) | new — `DeviceTensor`, `to_device`/`to_host`, `CUDA_CHECK` |
| `engine/cuda/tests/test_device.*` | new — round-trip / identity test proving build + copy + harness |
| `scripts/build_cuda.sh` | new — configure + compile CUDA into `build-cuda/` on the lab A6000 box |
| `scripts/test_cuda.sh` | new — run the round-trip test, report pass |
| `docs/07-cuda-build-matmul.md` | this spec |

## Done when

- `ATLAS_USE_CUDA=ON` builds cleanly on the lab A6000 box via `scripts/build_cuda.sh`.
- The round-trip test passes on-device: a CPU `Tensor` copied H2D and back is bit-exact,
  `CUDA_CHECK` fires on injected failure, and the harness reports a diff.
- The Windows CPU build + 10/10 CTest suite still pass with `ATLAS_USE_CUDA=OFF`.

## Next (Step 2)

The tiled **matmul** kernel (`y = x @ Wᵀ`, validated against `atlas::linear()`) is the
first real kernel and rides entirely on this infra. Its design is captured in memory
(`phase2-matmul-kernel-plan`) and gets its own doc when Step 2 begins.
