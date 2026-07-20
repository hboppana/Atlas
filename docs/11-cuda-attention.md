# Phase 2 · Step 5 — fused attention kernel (`ctx = softmax(q·kᵀ/√hd, causal) · v`, validated against `attention()`)

> Status: **done** — landed and validated on the A6000 (CUDA 12.6): 5/5 CTest, worst
> measured max-abs 8.94e-08, pinned at 1e-6.
> Predecessor: Step 4 — utility kernels (`embed`/`add`/`swiglu`/`rope`) — **done** ([10-cuda-kernels.md](10-cuda-kernels.md))
> Successor: Step 6 — full GPU forward pass (`GpuModel::forward()`), validated against `reference/logits.npy` ([12-cuda-forward.md](12-cuda-forward.md)).

## Goal

Write the **last missing GPU primitive**: fused causal GQA attention — Q·Kᵀ, scaled, causally
masked, softmaxed, and multiplied by V in a **single launch**. After this step every op in
TinyLlama's forward pass has a proven GPU counterpart (matmul, RMSNorm, embed, add, SwiGLU,
RoPE, attention), and Step 6 is pure wiring.

Attention gets its own translation unit and step — the doc-10 scope boundary — because its
complexity (two reductions per query row, causal masking, the 8:1 GQA asymmetry) justifies
isolated test discipline, unlike the four one-liner utility kernels.

This is the **correctness-first fused baseline**, not FlashAttention: the whole score row for
one query lives in shared memory, softmax runs over it, then the weighted-V accumulation —
one block does the entire pipeline for one (query position, head) pair. The online-softmax /
K-V-tiled rewrite is the named perf follow-up, held to the same oracle and pinned tolerance.

### Scope boundary

- **Not** online softmax / K-V tiling (flash-style). The baseline keeps the full score row in
  shared memory — 8 KB at TinyLlama's `max_seq = 2048`, well under the 48 KB default.
- **Not** a KV-cache decode path. Same contract as the CPU oracle: full `[seq, ·]` q/k/v in,
  `[seq, ·]` ctx out. Incremental decode is a Step-6+ concern.
- **Not** wired into a forward pass — that is Step 6.
- **Not** TF32/FP16. FP32-in / FP32-out, matching the oracle.

## The operation

The CPU oracle is `atlas::attention()` (`model.cpp:110–152`, declared `model.h:67`), itself
validated against HuggingFace in Phase 1 (`test_attention`). For each query head `hq` and
query position `i`:

```
hkv = hq / (n_heads / n_kv_heads)            # GQA: contiguous-block kv mapping
score[j] = (q[i,hq,:] · k[j,hkv,:]) / √hd    for j ≤ i   (causal: j > i masked out)
prob = softmax(score)                         # FP32, max-subtracted
ctx[i,hq,:] = Σ_j prob[j] · v[j,hkv,:]
```

| tensor | shape | TinyLlama dims |
|--------|-------|----------------|
| `q`, `ctx` | `[seq, n_heads·head_dim]` | `[seq, 32·64 = 2048]` |
| `k`, `v`   | `[seq, n_kv_heads·head_dim]` | `[seq, 4·64 = 256]` — the 8:1 GQA asymmetry |
| `n_heads`, `n_kv_heads`, `head_dim` | scalars | 32, 4, 64 |

## Kernel design

**Grid `(seq, n_heads)`** — one block owns one (query position `i` = `blockIdx.x`, query
head `hq` = `blockIdx.y`) pair; `BLOCK = 128` threads; dynamic shared memory `seq` floats
for the score row. Three phases, `__syncthreads()` between them:

1. **Scores.** Thread-per-key: thread `t` computes the full 64-term dot
   `q[i,hq,:]·k[j,hkv,:]/√hd` for its keys `j = t, t+BLOCK, … ≤ i` into `scores[j]`.
   Causality is by construction — `j > i` is never computed. The dot's `d`-loop is
   sequential per thread, same order as the oracle.
2. **Softmax.** `block_reduce_max` over each thread's partial max → row max `m` (exact —
   max is order-insensitive); `scores[j] = __expf(scores[j] − m)`; `block_reduce_sum` →
   `denom`. Both reductions come from the shared `reduce.cuh` (below).
3. **Weighted V.** Threads over `d` (head_dim = 64 active threads), each accumulating
   `Σ_j (scores[j]/denom) · v[j,hkv,d]` in a register — the **same j-order as the oracle's
   accumulation loop**, so no reduction reorder in this phase.

GPU↔CPU diff sources, by construction: FMA contraction of the dots, `__expf` vs `std::exp`,
and the `denom` sum reorder (one ≤seq-term reduction — the rmsnorm-class diff). Everything
else matches the oracle's arithmetic order. Expected ~1e-6 scale; measured then pinned.

**Shared reduction header — `engine/cuda/reduce.cuh` (new).** `block_reduce_sum` is lifted
from `rmsnorm.cu`'s anonymous namespace, plus the `block_reduce_max` variant (same
warp-shuffle two-level shape, `fmaxf`/`-INFINITY` instead of `+`/`0`); `rmsnorm.cu`
switches to including it. One departure from "verbatim": the functions are **templated on
the block size** — rmsnorm's copy hardcoded `BLOCK = 256`, and attention runs 128, so the
shared version takes it as a template parameter (`block_reduce_sum<BLOCK>`). Otherwise
behavior-identical — `test_rmsnorm`'s pinned tolerance is the guard, re-run green after
the switch.

## Host API (`engine/cuda/attention.h` / `attention.cu`)

Host-includable, declaration-only, mirroring the oracle signature:

```cpp
// Fused causal GQA attention: ctx = softmax(q·kᵀ/√head_dim, causal) · v, one launch.
// q/ctx [seq, n_heads·head_dim]; k/v [seq, n_kv_heads·head_dim]; query head hq reads
// kv head hq / (n_heads / n_kv_heads). Softmax in FP32 with max subtraction.
void launch_attention(const DeviceTensor& q, const DeviceTensor& k, const DeviceTensor& v,
                      int n_heads, int n_kv_heads, int head_dim, DeviceTensor& ctx);
```

The launcher asserts the same shape contract as the oracle (q/ctx width `n_heads·head_dim`,
k/v width `n_kv_heads·head_dim`, equal `seq` everywhere), plus the baseline's own bound —
`seq` floats of dynamic shared memory must fit (48 KB default → `seq ≤ 12288`; TinyLlama's
2048 is 6x under) — then launches with `seq·sizeof(float)` dynamic smem and calls
`CUDA_CHECK_KERNEL()`.

## Validation — same harness, measure then pin

Oracle flow per shape, identical to Steps 2–4 (`test_harness.h`): `fill_prng` q/k/v →
`attention()` on host → `ref`; `to_device` ×3, `launch_attention`, `to_host` → `got`;
`compare()` → max-abs/mean-abs; **measure on first green run, then pin**. `ctx` is an
out-param on both sides, so no upload-before-oracle ordering concern (unlike Step 4's
in-place kernels).

### Test cases (`engine/cuda/tests/test_attention_gpu.cu`)

Named `test_attention_gpu` — Phase 1 already registers CTest target `test_attention`
(`engine/CMakeLists.txt:40`), and `build-cuda/` builds both.

| case | seq | n_heads | n_kv_heads | head_dim | exercises |
|------|-----|---------|------------|----------|-----------|
| small-gqa | 4 | 4 | 2 | 16 | tiny GQA (q_per_kv=2), hand-checkable |
| mha-edge  | 4 | 2 | 2 | 16 | q_per_kv=1 — the no-GQA edge |
| prefill   | 8 | 32 | 4 | 64 | TinyLlama real dims, 8:1 |
| decode    | 1 | 32 | 4 | 64 | single query row, one key |
| long-seq  | 128 | 32 | 4 | 64 | mask variety + smem sizing beyond one BLOCK of keys |

Each case reports max-abs/mean-abs and asserts the pinned tolerance. Blob-free, random
inputs. `test_rmsnorm` re-run in the same suite guards the `reduce.cuh` extraction.

**Measured (A6000, CUDA 12.6) → pinned.** Worst cases prefill and long-seq, both
max-abs = 8.94e-08 (mean-abs ~6e-09); small-gqa/mha-edge 2.98e-08; decode **exactly 0** —
a single-key softmax is exactly 1.0 (`__expf(0)/1`), so ctx is a bit-exact copy of v's
row on both paths. Below rmsnorm's 5.36e-07 despite the extra `__expf`, because softmax
probs are ≤ 1 and inputs are ~[-0.5, 0.5). **Pinned at 1e-6** — ~11x headroom over the
measured worst, ~five orders of magnitude below a real-bug signal.

## Build & test workflow (unchanged loop)

- `engine/cuda/CMakeLists.txt`: add `attention.cu` to `atlas_cuda`; add `test_attention_gpu`
  executable + `add_test`, exactly like the `test_kernels` block.
- `scripts/test_cuda.sh`: widen the CTest filter to
  `-R 'test_device|test_matmul|test_rmsnorm|test_kernels|test_attention_gpu'`.
- Same loop: `edit engine/cuda/ → build_cuda.sh → test_cuda.sh → iterate`, pin a card with
  `CUDA_VISIBLE_DEVICES` on Suramar.

## Performance follow-up (named, deferred)

Once correctness is pinned, the perf pass holds the same oracle + tolerance:

- **Online softmax + K-V tiling** (flash-style): streaming max/denom rescale, K/V tiles
  through shared memory — removes the seq-sized score buffer and the seq bound entirely.
- **Warp-per-dot score phase** (shuffle-reduced dots, coalesced k reads) and **float4 loads**.
- Block-size sweep; benchmark via `scripts/benchmark.py` against the matmul-composed
  unfused path.

## Files created / touched

| File | Change |
|------|--------|
| `engine/cuda/attention.h` | new — `launch_attention` declaration |
| `engine/cuda/attention.cu` | new — fused kernel + launcher |
| `engine/cuda/reduce.cuh` | new — `block_reduce_sum` (lifted from rmsnorm.cu) + `block_reduce_max` |
| `engine/cuda/rmsnorm.cu` | include `reduce.cuh` instead of its local copy |
| `engine/cuda/tests/test_attention_gpu.cu` | new — validates vs `attention()` oracle |
| `engine/cuda/CMakeLists.txt` | add `attention.cu`; add `test_attention_gpu` |
| `scripts/test_cuda.sh` | widen CTest filter |
| `docs/11-cuda-attention.md` | this spec |

## Done when

- `launch_attention` builds into `atlas_cuda` via `scripts/build_cuda.sh`.
- `test_attention_gpu` passes on-device: GPU matches `attention()` within its
  measured-then-pinned tolerance across all five shapes.
- `test_device`, `test_matmul`, `test_rmsnorm`, `test_kernels` still pass — in particular
  `test_rmsnorm` proves the `reduce.cuh` extraction changed nothing.
- CTest reports 5/5; `ATLAS_USE_CUDA=OFF` CPU build stays 10/10.

## Design decisions

- **Fused baseline over flash-style online softmax.** Correctness before speed (the
  Steps 2–4 principle): the full-score-row design mirrors the oracle's loop structure
  almost exactly, so a diff is attributable — FMA, `__expf`, or the denom reorder, nothing
  else. Online softmax's incremental rescale is 2–3x the code with subtle failure modes,
  and TinyLlama's 2048 max_seq fits the shared-memory bound with 6x room. It is the perf
  follow-up, not the first version. (Settled with the user, 2026-07-11.)
- **`reduce.cuh` extraction over a second copy.** Attention needs both a max and a sum
  block reduction; rmsnorm already has the sum. Two divergent copies of warp-shuffle
  reduction code is the worse risk — `test_rmsnorm`'s pinned tolerance makes the lift
  behavior-provable in one suite run. (Settled with the user, 2026-07-11.)
- **One block per (query, head); no K-tiling.** `seq × n_heads` blocks saturate the A6000
  for prefill (8×32 = 256 blocks minimum real case) and the shape is trivially correct for
  decode (`grid = (1, 32)`). Uneven per-block work (query `i` has `i+1` keys) is a perf
  concern, not a correctness one.
- **Phase-3 accumulation preserves the oracle's j-order.** Each output element is one
  thread's sequential loop — deliberately *not* a parallel reduction — so the weighted-V
  sum is order-identical to the CPU and the only phase-3 diff is FMA contraction. The
  parallel-reduction version belongs to the perf follow-up.
- **`test_attention_gpu`, not `test_attention`.** Phase 1 owns that CTest name
  (`engine/CMakeLists.txt:40`) and both suites build in `build-cuda/`; a duplicate target
  is a configure error.
