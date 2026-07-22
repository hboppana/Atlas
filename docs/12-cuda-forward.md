# Phase 2 · Step 6 — full GPU forward pass (`GpuModel::forward()`, validated against `reference/logits.npy`)

> Status: **done** — validated on Suramar (A6000, sm_86), 2026-07-22. CTest 6/6 in
> `build-cuda/`, CPU 10/10.
> Predecessor: Step 5 — fused attention kernel — **done** ([11-cuda-attention.md](11-cuda-attention.md))
> Successor: Step 7 — token generation loop (greedy decode) en route to the pybind11 bridge
> + FastAPI serving.

## Goal

Wire the five proven kernels (matmul, rmsnorm, embed/add/swiglu/rope, attention) into the
**complete TinyLlama forward pass on the GPU**: token ids in, FP32 `[seq, vocab]` logits
out, every intermediate living on the device — one H2D upload of the ids at the top, one
D2H download of the logits at the bottom, nothing else crossing the bus per call.

Steps 2–5 proved each op against its CPU oracle in isolation; this step proves the
**composition** against the end-to-end oracle — the same `reference/logits.npy` (HF golden
logits for "The capital of France is") that gates the CPU engine in `test_forward`, with
the same three-part bar: pinned max-abs/mean-abs, per-row argmax agreement, and the last
row decoding to "Paris". Doc-11 called this step "pure wiring"; the work is the plumbing
that wiring needs — getting the 4.4 GB weight blob onto the device once, and holding the
launch sequence exactly to `Model::forward()`'s op order (`model.cpp:300–392`).

### Scope boundary

- **Not** a KV cache or incremental decode. Same contract as `Model::forward()`: full-prompt
  prefill, logits for every position. The generation loop is Step 7.
- **Not** INT8. FP32 weights only — `Model::qweights` is ignored on this path. GPU
  quantization (W8A32 kernels) is a named follow-up, not this step.
- **Not** performance work. Every launcher keeps its `CUDA_CHECK_KERNEL()` sync (~230
  syncs per call over 22 layers), scratch is allocated per call, no streams/graphs/overlap.
  Correctness first; the benchmark pass comes after, holding this step's pinned tolerance.
- **Not** the Python bridge or server — Step 7+ (`server/bridge.py`, `serve.py`).
- **Not** multi-GPU. One device, pinned with `CUDA_VISIBLE_DEVICES` as usual on Suramar.

## The operation

The oracle is `Model::forward()` (`model.cpp:300`), itself validated against HF within
pinned 1.5e-3 max-abs / 1e-4 mean-abs (measured 1.44e-4 / 8.27e-6, `test_forward`).
Structure to reproduce launch-for-call:

```
embed gather                                  → launch_embed
per layer × 22:
  rmsnorm(input_layernorm)                    → launch_rmsnorm
  q/k/v projections                           → launch_matmul ×3
  rope(q, 32 heads); rope(k, 4 heads)         → launch_rope ×2
  fused causal GQA attention                  → launch_attention
  o_proj; h += attn                           → launch_matmul, launch_add
  rmsnorm(post_attention_layernorm)           → launch_rmsnorm
  gate/up projections; swiglu; down_proj      → launch_matmul ×2, launch_swiglu, launch_matmul
  h += mlp                                    → launch_add
final rmsnorm; lm_head                        → launch_rmsnorm, launch_matmul
```

All launches go on the default stream in program order, so cross-kernel ordering is
guaranteed without explicit events. The RoPE cos/sin tables are computed **on the host**
(cheap: `[seq, 64]` ×2) and uploaded once per call.

## Host API (`engine/cuda/forward.h` / `forward.cu`)

The device blob upload is ~0.3 s of PCIe traffic and must not happen per call, so the API
is a struct that owns device state, not a free function — created once, `forward()` many
times. Move-only, same discipline as `WeightStore`/`DeviceTensor`; host-includable header.

```cpp
// The GPU mirror of Model: uploads the FP32 weight blob to the device once at create(),
// then runs the full TinyLlama forward pass on-device. FP32 only (Model::qweights is
// ignored). The Model must outlive the GpuModel — weight shapes/offsets are read from
// its WeightStore per call, mirroring the views-into-mapping lifetime rule.
struct GpuModel {
    static GpuModel create(const Model& model);  // one cudaMalloc + one H2D of the blob

    // Full-sequence prefill: token ids -> FP32 logits [seq, vocab] on the host.
    // Same contract and op order as Model::forward().
    Tensor forward(const std::vector<int>& token_ids) const;
};
```

### Weights on device — whole-blob upload + views

This is what `DeviceTensor::view` was built for (`device_tensor.h:42` names "the eventual
slice of the mmap'd weights' device copy" as its purpose): **one** `cudaMalloc` of
`WeightStore::size_bytes()`, **one** H2D memcpy of the whole mapped blob, then each weight
is a zero-copy `DeviceTensor::view` at the same offset as its host view —
`dev_base + (weights.get(name).data - weights.base())`, shape copied from the host tensor.
No per-tensor allocation, no name→device map to maintain; `WeightStore` stays the single
source of truth for shapes and offsets. 4.4 GB sits comfortably in the A6000's 48 GB
alongside worst-case activations (~400 MB at max_seq 2048, dominated by the
`[2048, 32000]` logits buffer).

`WeightStore` gains two trivial public accessors to make the offset math possible —
`const float* base() const` and `size_t size_bytes() const`. No behavior change; the CPU
suite is the guard.

### Shared RoPE tables — extract, don't duplicate

`Model::forward()` computes the cos/sin tables inline (`model.cpp:319–336`). Duplicating
that math in `forward.cu` is a silent-divergence risk of exactly the kind the `reduce.cuh`
extraction eliminated, so the loop is lifted into a shared host helper in `model.cpp`,
declared in `model.h` next to the other building blocks:

```cpp
// Precompute RoPE cos/sin tables for seq positions: [seq, head_dim] each, the head_dim/2
// angles concatenated with themselves (the half-split layout rope() expects).
void rope_tables(int64_t seq, const Config& cfg, Tensor& cos, Tensor& sin);
```

`Model::forward()` switches to calling it; `test_forward`'s pinned tolerance plus
`test_rope` prove the lift changed nothing, exactly the rmsnorm-extraction pattern.

### Token ids

`launch_embed` takes a raw `const int* ids_d` (`kernels.h` flagged "Step 6 wraps this"
because `DeviceTensor` is float-typed). `forward()` does the `cudaMalloc`/`cudaMemcpy`/
`cudaFree` for the id buffer directly, every call `CUDA_CHECK`-wrapped. A typed
`DeviceBuffer<int>` abstraction is not warranted for one 24-byte buffer.

### Scratch

Per-call `DeviceTensor::alloc` of the same nine buffers `Model::forward()` keeps
(`h, x, q, k, v, ctx, attn, gate, up, mlp` — sized by `seq`), reused across the 22 layers.
Persistent scratch preallocated at max_seq is a serving-era follow-up.

## Validation — the end-to-end oracle, measure then pin

### Test (`engine/cuda/tests/test_forward_gpu.cu`)

Mirrors `test_forward` structure exactly, including its SKIP discipline: the npy reader
self-test runs first; missing `ATLAS_WEIGHTS_DIR` blob → **SKIP green** (the blob is a
local artifact, never committed). With the blob:

1. Load tokenizer + `Model` (paths via the same `ATLAS_REFERENCE_DIR`/`ATLAS_WEIGHTS_DIR`
   compile definitions, added to this target in `engine/cuda/CMakeLists.txt`).
2. `GpuModel::create`, `forward(ids)` on the reference prompt ids `{1, 450, 7483, 310, 3444, 338}`.
3. **Shape**: logits `[6, 32000]`.
4. **Smoke**: argmax of the last row decodes to "Paris" via the Step 2 tokenizer.
5. **Rigorous**: per-logit max-abs/mean-abs vs `reference/logits.npy` under pinned
   ceilings; per-row argmax matches the reference at all 6 positions.
6. **Attribution diagnostic**: also run `model.forward(ids)` on the CPU and report GPU-vs-CPU
   max-abs/mean-abs (printed, not pinned). If the HF check ever reddens, this number says
   whether the drift is kernel-side (GPU≠CPU) or shared engine-side (both ≠ HF).

### Expected numerics

Every kernel is within ~1e-6 of its CPU oracle in isolation, but 22 layers compound and
the dominant diff source is the tiled matmul's accumulation reorder across 155 projections
(including the 4.4-GFLOP lm_head row-dot). The CPU engine — same FP32, different summation
order than PyTorch BLAS — measured 1.44e-4 against HF; the GPU pass should land at the
same order of magnitude, ~1e-4 to 1e-3. **Measure on first green run, then pin** at ~10x,
per the standing rule. The CPU pins (1.5e-3 / 1e-4) are the sanity reference: a GPU
measurement far above 1e-3 max-abs is a bug signal, not a tolerance to accommodate.
Argmax agreement at all 6 positions is binary and non-negotiable.

**Measured** (first green run, 2026-07-22): GPU vs HF `max_abs=2.04e-4`,
`mean_abs=8.10e-6` — the predicted order of magnitude, right alongside the CPU engine's
1.44e-4 / 8.27e-6. Pinned at `2e-3` / `8e-5`. Per-row argmax matched the reference at all
6 positions and the last row decoded to "Paris" (id 3681, logit 13.39). The attribution
diagnostic read GPU vs CPU `max_abs=8.01e-5`, `mean_abs=6.18e-6` — smaller than either
side's gap to HF, i.e. the two engines agree with each other more closely than either
agrees with PyTorch, which is what a correct composition should look like.

Informational wall-clock from the same run: `GpuModel::create` 1.08 s (the one-time 4.4 GB
upload), then GPU forward 0.053 s vs CPU forward 7.15 s — ~134x, with every launcher still
paying its `CUDA_CHECK_KERNEL()` sync. Not a benchmark; see the perf follow-ups.

Also print wall-clock for `GpuModel::forward` vs `Model::forward` — informational only
(shared box, noisy), not asserted; the real benchmark pass comes later.

## Build & test workflow (unchanged loop)

- `engine/cuda/CMakeLists.txt`: add `forward.cu` to `atlas_cuda`; add `test_forward_gpu`
  executable + `add_test`, with the `ATLAS_REFERENCE_DIR`/`ATLAS_WEIGHTS_DIR` compile
  definitions copied from `engine/CMakeLists.txt`'s pattern.
- `scripts/test_cuda.sh`: widen the CTest filter to
  `-R 'test_device|test_matmul|test_rmsnorm|test_kernels|test_attention_gpu|test_forward_gpu'`.
- The weight blob must exist on Suramar (`scripts/convert_weights.py` if absent).
- Same loop: `edit engine/cuda/ → build_cuda.sh → test_cuda.sh → iterate`, pin a card with
  `CUDA_VISIBLE_DEVICES`.

## Performance follow-up (named, deferred)

Held to this step's oracle + pinned tolerance once it lands:

- **Drop per-launch syncs** — `CUDA_CHECK_KERNEL()`'s device sync after every launch is
  pure launch-latency overhead in a 230-launch pass; a debug-only sync mode + one sync
  before D2H is the shape of the fix.
- **Persistent scratch** at max_seq; **last-row-only lm_head** for decode (Step 7 wants
  `[1, vocab]`, not `[seq, vocab]`).
- The already-named kernel follow-ups (flash-style attention, matmul tiling sweep),
  benchmarked end-to-end via `scripts/benchmark.py` against the CPU engine, logged to
  `docs/results/`.

## Files created / touched

| File | Change |
|------|--------|
| `engine/cuda/forward.h` | new — `GpuModel` declaration |
| `engine/cuda/forward.cu` | new — blob upload, weight views, launch orchestration |
| `engine/include/model.h` | `WeightStore::base()`/`size_bytes()` accessors; `rope_tables()` declaration |
| `engine/src/model.cpp` | lift RoPE table loop into `rope_tables()`; `forward()` calls it |
| `engine/cuda/tests/test_forward_gpu.cu` | new — end-to-end vs `reference/logits.npy` |
| `engine/cuda/CMakeLists.txt` | add `forward.cu`; add `test_forward_gpu` + path defines |
| `scripts/test_cuda.sh` | widen CTest filter |
| `docs/12-cuda-forward.md` | this spec |

## Done when

- `GpuModel` builds into `atlas_cuda` via `scripts/build_cuda.sh`.
- `test_forward_gpu` passes on-device with the blob present: shape, "Paris", per-row
  argmax vs the reference, and measured-then-pinned max-abs/mean-abs vs `logits.npy`;
  GPU-vs-CPU diff reported for attribution.
- `test_forward_gpu` SKIPs green without the blob.
- The five existing CUDA tests still pass; the CPU build (`ATLAS_USE_CUDA=OFF`) stays
  10/10 — in particular `test_forward` and `test_rope` prove the `rope_tables()` lift
  changed nothing.
- CTest in `build-cuda/` reports 6/6.

## Design decisions

- **`GpuModel` in `engine/cuda/`, not a `Model::forward_gpu()` member.** Doc-11's
  successor line sketched a member; refined here because Phase 2's guard discipline keeps
  `engine/include/model.h` CUDA-free, and a member declared in the CPU tree but defined
  only inside `atlas_cuda` is a link trap for any CPU-only caller. A struct also gives the
  upload-once state a home — a free function would re-upload 4.4 GB per call or hide a
  global. Matches the free-standing `launch_*` precedent. (Proposed — flag if `Model::`
  spelling matters.)
- **Whole-blob upload + offset views over 155 per-tensor copies.** One alloc, one memcpy,
  zero device-side bookkeeping, and `WeightStore` remains the single source of truth for
  shapes/offsets — the design `DeviceTensor::view` was explicitly built for in Step 1.
  Cost: two public accessors on `WeightStore`.
- **`rope_tables()` extraction over duplicating the table math in `forward.cu`.** Two
  copies of the inv-freq/half-split loop is the same divergence risk the `reduce.cuh`
  lift retired; the CPU suite makes the extraction behavior-provable before any GPU code
  runs.
- **Composition validated against HF, with GPU-vs-CPU as diagnostic only.** `logits.npy`
  is the project's end-to-end oracle and keeps this test's bar identical to `test_forward`'s;
  the GPU-vs-CPU number is printed for attribution, not pinned, so one tolerance — the
  one that means "matches the real model" — gates the step.
- **FP32 only; `qweights` ignored.** The INT8 path's value on GPU is bandwidth, which is a
  performance feature; landing it here would mix a numerics change into the wiring step
  and make any logit diff ambiguous between composition bugs and quantization error.
- **Keep per-launch syncs and per-call scratch.** Slower, but every failure surfaces at
  the offending launch with a clean error, which is what a first end-to-end bring-up
  needs. Both removals are named perf follow-ups with the tolerance already pinned.
