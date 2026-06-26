# Phase 2 · Step 2 — tiled matmul kernel (`y = x @ Wᵀ`, validated against `linear()`)

> Status: **planned** — design only; no kernel code yet.
> Predecessor: Step 1 — CUDA bring-up infra — **done** ([07-cuda-build-matmul.md](07-cuda-build-matmul.md))
> Successor: Step 3 — the fused **RMSNorm** kernel, validated against `atlas::rmsnorm()` ([09-cuda-rmsnorm.md](09-cuda-rmsnorm.md)); then the remaining kernels (RoPE, SwiGLU activation, attention) ride the same infra + harness, then the full GPU forward pass validated against `reference/logits.npy`.

## Goal

Write the **first real CUDA kernel** and prove it correct. The matmul is the single
heaviest op in the model — every attention projection, every MLP layer, and the LM head are
`linear()` calls — so it is where GPU speed is won, and the right first kernel to stand up.

Step 2 ships a **tiled shared-memory matmul** that computes exactly what the proven CPU
`atlas::linear()` computes (`engine/src/model.cpp:47`), validated element-wise against it
across aligned, non-aligned, and real TinyLlama shapes. It rides entirely on the Step 1
infra (`DeviceTensor`, the copy helpers, `CUDA_CHECK`, the diff harness), so a failure here
is unambiguously a *kernel* bug — the build path, the copies, and the comparison mechanism
were already proven green with the no-math payload.

This is also the **first kernel whose GPU result is not bit-exact** to the CPU oracle: tiled
partial-sum accumulation reorders the FP32 additions relative to the CPU's sequential dot
product, so the diff is small-but-nonzero. Step 2 is therefore where the Phase 1
**measure-the-diff-then-pin-it** discipline first does real work (Step 1's payloads were
exact, so they pinned at 0).

### Scope boundary (what Step 2 is *not*)

- **Not** a fused or fully perf-tuned kernel. Step 2's bar is *correct and tiled*, not
  *fastest*. Register-blocking / `float4` vectorization / larger tiles / TF32 are a named
  **follow-up** (see "Performance follow-up"), each re-validated against the same oracle and
  tolerance. Correctness against the oracle comes first — the project's standing discipline.
- **Not** the INT8 path. `linear_q8` (`engine/src/quantize.cpp:63`) stays CPU-only for now;
  the GPU kernel is FP32-in / FP32-out, matching `linear()`.
- **Not** wired into a model forward pass yet. Step 2 delivers `launch_matmul` + its test;
  swapping `linear()` call sites to the GPU kernel comes with the full GPU forward step.
- **Not** any of the other kernels (RMSNorm, RoPE, attention, SwiGLU) — each is its own
  later step on this same infra.

## The operation (and why the W layout is GPU-friendly)

`linear()` computes `y = x @ Wᵀ` with these shapes (PyTorch `nn.Linear`, **no** transpose at
convert time):

| tensor | shape | layout |
|--------|-------|--------|
| `x`   | `[m, in]`   | row-major; each row contiguous in `in` |
| `w`   | `[out, in]` | row-major; each **row** contiguous in `in` |
| `out` | `[m, out]`  | row-major |

`out[i][o] = Σ_p x[i][p] · w[o][p]`. The contraction index `p` walks the **last
(contiguous) dimension of both operands**. In GEMM terms this is an **NT** product
(`C = A · Bᵀ` with `A = x`, `B = w`): we never need to materialize `Wᵀ` in memory, and tile
loads of both `x` and `w` along `p` are naturally coalesced. This is the layout the kernel
targets directly.

TinyLlama's real matmul shapes (`engine/include/model.h`), all of which the test must cover:

| call site | `in` | `out` | notes |
|-----------|------|-------|-------|
| q_proj / o_proj | 2048 | 2048 | square |
| k_proj / v_proj | 2048 | 256  | GQA kv_dim (8:1) |
| mlp gate/up     | 2048 | 5632 | SwiGLU inner width |
| mlp down        | 5632 | 2048 | wide contraction |
| lm_head         | 2048 | 32000| huge `out` |

`m` is the sequence length: large during prefill, `m = 1` during decode. None of
`{256, 2048, 5632, 32000}` nor `m = 1` are clean tile multiples, so **boundary masking is
mandatory**, not optional.

## Kernel design

A classic **tiled shared-memory** GEMM, written for the NT layout above:

- A `TILE × TILE` output tile per thread block (baseline `TILE = 16`; `32` evaluated in the
  perf follow-up). Each thread computes one `out[i][o]` (baseline; register-blocking — each
  thread owning a small `RT × RT` micro-tile — is the first perf lever, deferred).
- The contraction `in` is walked in `TILE`-wide steps. Each step cooperatively stages a
  tile of `x` (`[TILE rows of m] × [TILE of p]`) and a tile of `w` (`[TILE rows of out] ×
  [TILE of p]`) into `__shared__`, `__syncthreads()`, then accumulates the partial dot
  products from shared memory, `__syncthreads()`, advance.
- **Boundary masking:** threads whose global `(i, p)` or `(o, p)` fall outside the real
  extents load `0.0f` into shared memory instead of reading out of bounds; threads whose
  output `(i, o)` is outside `[m, out]` skip the store. This is what makes the non-multiple
  TinyLlama dims correct.
- **FP32 accumulation** in a register, matching `linear()`'s `float acc`. No TF32, no FP16 —
  same numeric type as the oracle, so the only diff is summation *order*, not precision.
- Launched through a host wrapper that sizes the grid and calls `CUDA_CHECK_KERNEL()` after
  the `<<<>>>` (the Step 1 launch-error discipline).

### Host API (`engine/cuda/matmul.h` / `matmul.cu`)

A new translation unit added to the `atlas_cuda` library, mirroring `linear()`'s contract on
device tensors:

```cpp
// y = x @ Wᵀ on the device. x:[m,in], w:[out,in] (nn.Linear layout, no transpose),
// out:[m,out] — pre-allocated by the caller. Same shape asserts as atlas::linear().
void launch_matmul(const DeviceTensor& x, const DeviceTensor& w, DeviceTensor& out);
```

`matmul.h` is host-includable (declaration only, like `device_tensor.h`); the kernel and
`<cuda_runtime.h>` use stay in `matmul.cu`. The existing `launch_scale` payload stays for
now as the infra smoke test; it can be retired once the kernel suite is established.

## Validation — reuse the Step 1 harness, this time with a real tolerance

`atlas_cuda` already links `atlas_engine` (`engine/cuda/CMakeLists.txt:11`), so the CPU
`linear()` oracle is directly callable from the test. The flow per shape:

1. `fill_prng` deterministic inputs for `x` and `w` (host `Tensor`s).
2. CPU oracle: `atlas::linear(x, w, ref)`.
3. GPU: `to_device` both, `launch_matmul`, `to_host` the result.
4. `compare(got, ref)` → `max-abs` / `mean-abs`; **measure on the first green run, then pin**
   the gate. The expected tolerance scales with the contraction length `in` (more terms →
   more reordered FP32 rounding); the wide `in = 5632` (mlp down) and the long `in = 2048 →
   out = 32000` lm_head are the stress cases.

**Refactor:** the diff harness in `test_device.cu` (`Diff`, `compare`, `fill_prng`, the
`CHECK` macro, and the subprocess death-case helper) is currently file-local. Step 2 lifts
the reusable pieces into a small shared header `engine/cuda/tests/test_harness.h` — this is
literally "the pattern every later kernel reuses" from doc-07, made concrete — and both
`test_device.cu` and the new `test_matmul.cu` include it. No behavior change to
`test_device`; it just stops duplicating the harness.

### Test cases (`engine/cuda/tests/test_matmul.cu`)

- **Tile-aligned** small case (e.g. `m=32, in=32, out=32`) — the simplest path, no masking.
- **Non-aligned** case (e.g. `m=3, in=50, out=17`) — exercises masking on all three dims.
- **Decode** shape `m=1` (e.g. `[1, 2048] @ wᵀ[2048→2048]`) — the single-row degenerate grid.
- **Representative TinyLlama shapes** vs `linear()`: q_proj `[m, 2048]→2048`, mlp gate
  `[m, 2048]→5632`, mlp down `[m, 5632]→2048`, lm_head `[1, 2048]→32000` — with a modest
  `m` (e.g. 4–8) to keep the blob-free test fast.
- Each reports `max-abs`/`mean-abs` and asserts against the pinned tolerance. Blob-free:
  random inputs, no 4.4 GB weight blob.

## Build & test workflow (unchanged loop)

- `engine/cuda/CMakeLists.txt`: add `matmul.cu` to the `atlas_cuda` library sources; add a
  `test_matmul` executable (`tests/test_matmul.cu`, links `atlas_cuda`) and
  `add_test(NAME test_matmul ...)`, exactly like `test_device`.
- `scripts/build_cuda.sh`: no change — it builds the whole `atlas_cuda` target.
- `scripts/test_cuda.sh`: widen the CTest filter from `-R test_device` to
  `-R 'test_device|test_matmul'` (the doc-07 comment already anticipates this).
- The loop is the same: `edit engine/cuda/ → build_cuda.sh → test_cuda.sh → iterate`, pinned
  to one card on the shared box (`CUDA_VISIBLE_DEVICES=1`).
- Portable check unchanged: the CPU build stays green at 10/10 with `ATLAS_USE_CUDA=OFF`
  (`matmul.cu` only compiles under the flag).

## Performance follow-up (named, deferred)

Once correctness is pinned, a perf pass tunes the *same* kernel against the *same* oracle and
tolerance — no correctness regression allowed. Levers, roughly in order: register-blocking
(each thread owns an `RT × RT` micro-tile, raising arithmetic intensity), `float4` vectorized
shared-memory loads, `TILE = 32` / larger block tiles, double-buffered shared memory, and —
only if the accuracy budget allows and re-validated — TF32 on the A6000 tensor cores. This is
where the A6000-tuned numbers come from; Step 2 proper just earns the right to chase them by
being correct first.

## Files created / touched

| File | Change |
|------|--------|
| `engine/cuda/matmul.h` | new — `launch_matmul` declaration (host-includable) |
| `engine/cuda/matmul.cu` | new — tiled NT matmul kernel + launcher |
| `engine/cuda/tests/test_harness.h` | new — `Diff`/`compare`/`fill_prng`/`CHECK` lifted from `test_device.cu` |
| `engine/cuda/tests/test_device.cu` | include the shared harness (drop the local copy) |
| `engine/cuda/tests/test_matmul.cu` | new — validates `launch_matmul` vs `linear()` across shapes |
| `engine/cuda/CMakeLists.txt` | add `matmul.cu` to `atlas_cuda`; add `test_matmul` target + test |
| `scripts/test_cuda.sh` | widen CTest filter to `test_device|test_matmul` |
| `docs/08-cuda-matmul.md` | this spec |

## Done when

- `launch_matmul` builds into `atlas_cuda` via `scripts/build_cuda.sh` on the lab A6000 box.
- `test_matmul` passes on-device: the GPU matmul matches `atlas::linear()` within a
  **measured-then-pinned** tolerance across aligned, non-aligned, `m=1`, and the
  representative TinyLlama shapes (q/o/kv proj, mlp gate/up/down, lm_head).
- `test_device` still passes against the extracted shared harness (no behavior change).
- The CPU build + 10/10 CTest suite still pass with `ATLAS_USE_CUDA=OFF`.

## Design decisions

- **NT layout, no `Wᵀ` materialized.** `linear()`'s `[out, in]` weight contracts along the
  contiguous dim of both operands; the kernel consumes that directly, so tile loads are
  coalesced and there is no transpose pass or extra device buffer.
- **Same FP32 numeric type as the oracle.** The only GPU↔CPU difference is summation order,
  which makes the diff small and *explainable*; mixing in TF32/FP16 now would confound a
  correctness bug with a precision budget. Precision tricks belong to the perf follow-up.
- **Tolerance measured, then pinned** — first kernel where the diff is nonzero, so this is
  where the Phase 1 method earns its keep. Never guess the tolerance up front.
- **Reuse `linear()` as the oracle, not a re-derivation.** The CPU path is already validated
  against HuggingFace (Phase 1); checking the kernel against it inherits that trust.
- **Extract the harness now.** doc-07 promised "the pattern every later kernel reuses"; with
  a second kernel arriving, the harness moves to a shared header rather than being copied a
  third time.
- **Correctness before speed.** Step 2 is explicitly a *correct tiled baseline*; the
  A6000-tuned performance is a follow-up that must hold the same oracle + tolerance. This
  keeps the failure domains apart, consistent with the Step-1/Step-2 split.
