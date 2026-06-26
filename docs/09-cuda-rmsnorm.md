# Phase 2 · Step 3 — fused RMSNorm kernel (`out = x · rsqrt(mean(x²)+eps) · w`, validated against `rmsnorm()`)

> Status: **planned** — design only; no kernel code yet.
> Predecessor: Step 2 — tiled matmul kernel — **done** ([08-cuda-matmul.md](08-cuda-matmul.md))
> Successor: the remaining kernels (RoPE, SwiGLU activation, fused attention) ride the same infra + harness, then the full GPU forward pass validated against `reference/logits.npy`.

## Goal

Write the **second real CUDA kernel** and prove it correct. After the matmul (Step 2), RMSNorm
is the next op in TinyLlama's forward pass — it normalizes the residual stream before every
attention and MLP block, and once more before the LM head. It is the natural next kernel: small,
self-contained, and the place the **block reduction** pattern first appears — the same warp-level
reduction that attention's softmax will reuse later.

Step 3 ships a **fused single-pass reduction RMSNorm** that computes exactly what the proven CPU
`atlas::rmsnorm()` computes (`engine/src/model.cpp:67`), validated element-wise against it across
aligned, non-aligned, decode, and the real TinyLlama hidden width. It rides entirely on the
Step-1 infra (`DeviceTensor`, the copy helpers, `CUDA_CHECK`) and the shared diff harness
extracted in Step 2 (`engine/cuda/tests/test_harness.h`), so a failure here is unambiguously a
*kernel* bug — the build path, the copies, and the comparison mechanism are already proven green.

Like the matmul, this is **not bit-exact** to the CPU oracle: the parallel sum-of-squares reorders
the FP32 additions relative to the CPU's sequential accumulation, so the diff is small-but-nonzero.
It is the same Phase 1 **measure-the-diff-then-pin-it** discipline as Step 2 — but the diff's
*source* is now a **reduction reorder**, not a tiled GEMM accumulation. The reduction is a single
2048-term sum per row (vs the matmul's contraction up to 5632), so the expected diff is smaller
still.

### Scope boundary (what Step 3 is *not*)

- **Not** fused with attention or the residual add. Step 3 delivers RMSNorm as a standalone
  kernel matching `rmsnorm()`'s contract; fusing norm+matmul or norm+residual is a later perf
  concern, not this step.
- **Not** the in-place residual-add / embedding-lookup / SwiGLU / RoPE kernels — each is its own
  later step on this same infra.
- **Not** wired into a model forward pass yet. Step 3 delivers `launch_rmsnorm` + its test;
  swapping `rmsnorm()` call sites to the GPU kernel comes with the full GPU forward step.
- **Not** TF32/FP16. The kernel is FP32-in / FP32-out, matching `rmsnorm()`; precision tricks
  belong to the perf follow-up, same split as doc-08.

## The operation

`rmsnorm()` computes, per row, **with no mean subtraction** (the Llama/RMSNorm definition):

```
out[i][j] = x[i][j] · rsqrt( mean_j(x[i][j]²) + eps ) · w[j]
```

| tensor | shape | layout |
|--------|-------|--------|
| `x`   | `[m, n]` | row-major; each row contiguous in `n` |
| `w`   | `[n]`    | per-feature gain, shared across all rows |
| `out` | `[m, n]` | row-major, same shape as `x` |

`eps = c.rms_norm_eps`. Each output row depends only on its own input row plus the shared weight
`w`, so **rows are fully independent** — the natural parallel decomposition is one thread block
per row.

For TinyLlama, **`n = 2048` (the hidden width) in every call site**: `input_layernorm` and
`post_attention_layernorm` in each layer, and the final `model.norm` before the LM head
(`engine/src/model.cpp:367/379/388`). `m` is the sequence length — large during prefill, `m = 1`
during decode. `n = 2048` is not a clean multiple of an arbitrary block size, and the test also
covers a deliberately non-multiple `n`, so **tail masking in the reduction is mandatory**.

## Kernel design

A classic **block-per-row reduction**:

- **One thread block per row** (`grid.x = m`); the block reduces its row's `n` elements. A fixed
  block size (baseline `BLOCK = 256`; tuning is the perf follow-up).
- Each thread **strides** over the row (`j = threadIdx.x, threadIdx.x + BLOCK, …`), accumulating a
  partial sum-of-squares in a register. With `n = 2048, BLOCK = 256` that is 8 elements/thread;
  consecutive threads read consecutive `j`, so the loads are coalesced.
- **Two-level reduction:** reduce within each warp with `__shfl_down_sync`, write one partial per
  warp to `__shared__`, `__syncthreads()`, then the first warp reduces the per-warp partials to the
  row's `ss`.
- Thread 0 (or the whole block via a shared broadcast) computes `scale = rsqrtf(ss / n + eps)`
  once; every thread then writes `out[i][j] = x[i][j] · scale · w[j]` over its strided elements.
- **Tail masking:** a strided index `j >= n` contributes `0` to the partial sum and skips its
  store — this is what makes a non-`BLOCK`-multiple `n` correct (the reduction analogue of the
  matmul's boundary masking).
- **FP32 accumulation** in registers/shared, matching `rmsnorm()`'s `float ss`. No TF32, no FP16 —
  same numeric type as the oracle, so the only diff is reduction *order*. (`rsqrtf` vs the CPU's
  `1.0f / std::sqrt` is a named, measured numeric difference, not an algorithmic one.)
- Launched through a host wrapper that sizes the grid and calls `CUDA_CHECK_KERNEL()` after the
  `<<<>>>` (the Step-1 launch-error discipline).

### Host API (`engine/cuda/rmsnorm.h` / `rmsnorm.cu`)

A new translation unit added to the `atlas_cuda` library, mirroring `rmsnorm()`'s contract on
device tensors:

```cpp
// out = x · rsqrt(mean(x²)+eps) · w, per row, on the device. x:[m,n], w:[n],
// out:[m,n] — pre-allocated by the caller. Same shape asserts as atlas::rmsnorm().
void launch_rmsnorm(const DeviceTensor& x, const DeviceTensor& w,
                    float eps, DeviceTensor& out);
```

`rmsnorm.h` is host-includable (declaration only, like `matmul.h` / `device_tensor.h`); the kernel
and `<cuda_runtime.h>` use stay in `rmsnorm.cu`. Shape asserts match `atlas::rmsnorm()`:
`x.shape == out.shape`, 2-D, `w.numel() == n`.

## Validation — reuse the Step-2 harness, measure then pin

`atlas_cuda` already links `atlas_engine` (`engine/cuda/CMakeLists.txt`), so the CPU `rmsnorm()`
oracle — itself validated against HuggingFace in Phase 1 — is directly callable from the test, and
the check inherits that trust. The flow per shape is identical to `test_matmul.cu`:

1. `fill_prng` deterministic inputs for `x` and `w` (host `Tensor`s).
2. CPU oracle: `atlas::rmsnorm(x, w, eps, ref)` with TinyLlama's real `rms_norm_eps`.
3. GPU: `to_device` both, `launch_rmsnorm`, `to_host` the result.
4. `compare(got, ref)` (from `test_harness.h`) → `max-abs` / `mean-abs`; **measure on the first
   green run, then pin** the gate. A single 2048-term reduction reorders far less than the
   matmul's up-to-5632-term contraction, so the expected tolerance is at or below the matmul's —
   but the number is measured, never guessed.

No new harness work: Step 2 already lifted `Diff`/`compare`/`fill_prng`/`CHECK` into the shared
`test_harness.h`; `test_rmsnorm.cu` just includes it.

### Test cases (`engine/cuda/tests/test_rmsnorm.cu`)

- **Aligned small** (e.g. `m=4, n=64`) — the simplest path, `n` a clean block multiple, no tail.
- **Non-multiple `n`** (e.g. `m=3, n=100`) — exercises the reduction tail masking.
- **Decode** shape `m=1, n=2048` — single row, single block.
- **Prefill** shape `m=8, n=2048` — the real TinyLlama hidden width across a multi-row grid.
- Each reports `max-abs`/`mean-abs` and asserts against the pinned tolerance. Blob-free: random
  inputs, no 4.4 GB weight blob.

## Build & test workflow (unchanged loop)

- `engine/cuda/CMakeLists.txt`: add `rmsnorm.cu` to the `atlas_cuda` library sources; add a
  `test_rmsnorm` executable (`tests/test_rmsnorm.cu`, links `atlas_cuda`) and
  `add_test(NAME test_rmsnorm ...)`, exactly like the `test_matmul` block.
- `scripts/build_cuda.sh`: no change — it builds the whole `atlas_cuda` target.
- `scripts/test_cuda.sh`: widen the CTest filter from `-R 'test_device|test_matmul'` to
  `-R 'test_device|test_matmul|test_rmsnorm'`.
- The loop is the same: `edit engine/cuda/ → build_cuda.sh → test_cuda.sh → iterate`, pinned to
  one card on the shared box (`CUDA_VISIBLE_DEVICES=1`).
- Portable check unchanged: the CPU build stays green at 10/10 with `ATLAS_USE_CUDA=OFF`
  (`rmsnorm.cu` only compiles under the flag).

## Performance follow-up (named, deferred)

Once correctness is pinned, a perf pass tunes the *same* kernel against the *same* oracle and
tolerance — no correctness regression allowed. Levers, roughly in order: block-size sweep,
`float4` vectorized loads/stores over the row, fusing the norm with the following matmul (avoid a
round-trip to global memory), and — only if the accuracy budget allows and re-validated — TF32.
This is where the A6000-tuned numbers come from; Step 3 proper just earns the right to chase them
by being correct first.

## Files created / touched

| File | Change |
|------|--------|
| `engine/cuda/rmsnorm.h` | new — `launch_rmsnorm` declaration (host-includable) |
| `engine/cuda/rmsnorm.cu` | new — fused block-per-row reduction RMSNorm kernel + launcher |
| `engine/cuda/tests/test_rmsnorm.cu` | new — validates `launch_rmsnorm` vs `rmsnorm()` across shapes |
| `engine/cuda/CMakeLists.txt` | add `rmsnorm.cu` to `atlas_cuda`; add `test_rmsnorm` target + test |
| `scripts/test_cuda.sh` | widen CTest filter to `test_device|test_matmul|test_rmsnorm` |
| `docs/09-cuda-rmsnorm.md` | this spec |

## Done when

- `launch_rmsnorm` builds into `atlas_cuda` via `scripts/build_cuda.sh` on the lab A6000 box.
- `test_rmsnorm` passes on-device: the GPU RMSNorm matches `atlas::rmsnorm()` within a
  **measured-then-pinned** tolerance across aligned, non-multiple-`n`, `m=1` decode, and the
  TinyLlama `n=2048` prefill shapes.
- `test_device` and `test_matmul` still pass (no regression).
- The CPU build + 10/10 CTest suite still pass with `ATLAS_USE_CUDA=OFF`.

## Design decisions

- **Block-per-row reduction.** Rows are independent and `n` (2048) fits a single block's
  cooperative reduction, so one block per row is the simplest correct decomposition and keeps all
  of a row's traffic coalesced. Cross-row batching/vectorization is a perf follow-up.
- **Same FP32 numeric type as the oracle.** The only GPU↔CPU difference is reduction order (plus
  `rsqrtf` vs `1/sqrt`), which makes the diff small and *explainable*; mixing in TF32/FP16 now
  would confound a correctness bug with a precision budget.
- **Tolerance measured, then pinned** — consistent with Step 2; the diff source is a reduction
  reorder this time, so it is re-measured rather than assumed from the matmul's pin.
- **Reuse `rmsnorm()` as the oracle, not a re-derivation.** The CPU path is already validated
  against HuggingFace (Phase 1); checking the kernel against it inherits that trust.
- **Reuse the Step-2 harness as-is.** `test_harness.h` already holds `compare`/`fill_prng`/`CHECK`;
  the second kernel test consumes it unchanged — the payoff of extracting it in Step 2.
- **Correctness before speed.** Step 3 is explicitly a *correct fused baseline*; the A6000-tuned
  performance (and norm-matmul fusion) is a follow-up that must hold the same oracle + tolerance,
  keeping the failure domains apart.
