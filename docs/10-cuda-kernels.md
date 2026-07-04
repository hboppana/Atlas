# Phase 2 · Step 4 — Utility kernels (`launch_embed`, `launch_add`, `launch_swiglu`, `launch_rope`)

> Status: **planned** — design only; no kernel code yet.
> Predecessor: Step 3 — fused RMSNorm kernel — **done** ([09-cuda-rmsnorm.md](09-cuda-rmsnorm.md))
> Successor: Step 5 — fused attention kernel (`attention.cu`), then Step 6 — full GPU forward pass.

## Goal

Write the four remaining **small utility kernels** that the TinyLlama forward pass needs
beyond matmul and RMSNorm — then the only missing GPU primitive is attention. After this step
every per-element and gather operation in `model.cpp` has a GPU counterpart, proven correct
against its CPU oracle, and the infra is ready for the fused attention kernel that completes
Phase 2.

The four operations are:

| Kernel | CPU oracle | Where used in forward |
|--------|-----------|----------------------|
| `launch_embed` | inline row-gather in `Model::forward()` | token → hidden state |
| `launch_add`   | `atlas::add()` (`tensor.cpp:161`) | two residual adds per layer |
| `launch_swiglu`| `atlas::swiglu()` (`model.cpp`) | MLP gate in each layer |
| `launch_rope`  | `atlas::rope()` (`model.cpp`) | q and k, each layer |

All four are element-wise or gather kernels: no shared memory, no reduction — simpler than
RMSNorm and matmul. A single new translation unit (`engine/cuda/kernels.cu`) groups them so
the number of link targets stays manageable; all four declarations live in `kernels.h`.

Like the prior steps, this is about **correctness first**. Each kernel is validated against
its CPU oracle within a measured-then-pinned tolerance. The same Step-1 infra
(`DeviceTensor`, copy helpers, `CUDA_CHECK`) and Step-2 harness (`test_harness.h`) are
reused without modification.

### Scope boundary

- **Not** fused with any neighbor. `launch_embed` is a standalone gather; `launch_add` is
  standalone elementwise; `launch_swiglu` is standalone gating; `launch_rope` is standalone
  in-place rotation. Fusion (e.g. norm + residual + add) is a perf follow-up.
- **Not** the attention kernel — that is its own translation unit and step (Step 5,
  `attention.cu`), because its complexity and test discipline justify isolation.
- **Not** wired into a model forward pass yet. Step 4 delivers the four launchers + their
  tests; wiring them into `Model::forward_gpu()` comes with Step 6.
- **Not** TF32/FP16. All kernels are FP32-in / FP32-out, matching their CPU oracles.

## The operations

### `launch_embed` — token embedding gather

The CPU path (inline, `model.cpp:310–317`):

```
for i in [0, seq):
    h[i, :] = embed[token_ids[i], :]
```

`embed` is `[vocab_size=32000, H=2048]` row-major; `token_ids` is a 1-D index array of
length `seq`. The output `h` is `[seq, H]`. This is a **row gather**: each output row is an
independent copy of one embedding row.

**GPU design:** one thread block per token (`grid.x = seq`), `BLOCK = 256` threads striding
over `H = 2048` elements/row (8 elements/thread, coalesced). Token ids are passed as a
device-side `int32` array (separate small allocation via `DeviceTensor`'s existing
`alloc({n})` with appropriate dtype, or the caller casts — see Host API below). No
shared memory, no synchronization.

| tensor | shape | layout |
|--------|-------|--------|
| `embed` | `[vocab, H]` | `DeviceTensor` — pre-uploaded weight |
| `ids`   | `[seq]`      | device `int32` array |
| `out`   | `[seq, H]`   | row-major, pre-allocated |

### `launch_add` — element-wise in-place residual add

The CPU oracle (`tensor.cpp:161`, via `elementwise`):

```
out[i] += b[i]   (same shape, applied element-wise)
```

The forward pass always calls this as `add(h, attn_or_mlp, h)` — the accumulator `h` is
both input and output. The GPU kernel is **in-place**: `a[i] += b[i]` with `a` and `b` both
being `DeviceTensor` of equal shape.

**GPU design:** 1-D grid of thread blocks, each thread handles one element. `grid =
ceil(numel / BLOCK)`, `BLOCK = 256`. For `h` shape `[seq, H=2048]`, `numel = seq * 2048`;
consecutive threads read consecutive elements — coalesced.

| tensor | shape | layout |
|--------|-------|--------|
| `a` (in-place) | `[m, n]` | flat, pre-allocated |
| `b`             | `[m, n]` | same shape as `a` |

### `launch_swiglu` — SwiGLU gating in place

The CPU oracle (`model.cpp`):

```
gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i]   (SiLU(gate) * up)
```

In-place in `gate`; `up` is read-only. Both `[seq, I=5632]` in the forward pass.

**GPU design:** 1-D grid, `BLOCK = 256`, one element per thread. `__expf` for the
exponential (FP32 device intrinsic, matches precision of `std::exp` on CPU up to rounding).
`numel = seq * 5632`.

| tensor | shape | layout |
|--------|-------|--------|
| `gate` (in-place) | `[m, I]` | flat |
| `up`               | `[m, I]` | same shape |

### `launch_rope` — in-place RoPE (NeoX half-split)

The CPU oracle (`model.cpp:87–105`): HF Llama's NeoX rotation — **not** interleaved
adjacent pairs. For each `(pos, head)`:

```
x_rot = x * cos + rotate_half(x) * sin
rotate_half(x): first half → -x[half:], second half → x[:half]
cos/sin: [seq, head_dim], where head_dim angles = [theta_0..theta_{hd/2-1}, theta_0..theta_{hd/2-1}]
```

`x` is `[seq, n_heads * head_dim]` for queries or `[seq, n_kv_heads * head_dim]` for keys.
Applied in-place.

**GPU design:** 2-D grid — `(grid.x = seq * n_heads, grid.y = 1)`, `BLOCK = head_dim / 2`
(= 32 for TinyLlama's `head_dim = 64`). Each block owns one `(pos, head)` pair and rotates
its `head_dim / 2` pairs in parallel:

```
thread i (in [0, hd/2)):
    x0 = x[pos, head*hd + i]
    x1 = x[pos, head*hd + i + hd/2]
    x[pos, head*hd + i]       = x0 * cos[pos, i]       - x1 * sin[pos, i]
    x[pos, head*hd + i + hd/2] = x1 * cos[pos, i+hd/2] + x0 * sin[pos, i+hd/2]
```

The `cos`/`sin` tables are already computed by the CPU `forward()` preamble (size
`[max_seq, head_dim]`); they are uploaded once and passed as `DeviceTensor` (read-only).

| tensor | shape | note |
|--------|-------|------|
| `x` (in-place) | `[seq, n_heads * hd]` | q or k |
| `cos` | `[seq, hd]` | precomputed, read-only |
| `sin` | `[seq, hd]` | precomputed, read-only |
| `n_heads`, `head_dim` | scalars | passed to launcher |

## Host API (`engine/cuda/kernels.h` / `kernels.cu`)

A single host-includable header, declaration-only:

```cpp
// Embed: gather seq rows of embed[vocab, H] by integer token ids into out[seq, H].
// ids_d: device int32 array of length seq (pre-uploaded by caller).
void launch_embed(const DeviceTensor& embed, const int* ids_d,
                  int64_t seq, DeviceTensor& out);

// Element-wise in-place add: a[i] += b[i], same shape.
void launch_add(DeviceTensor& a, const DeviceTensor& b);

// SwiGLU in-place: gate[i] = SiLU(gate[i]) * up[i].
void launch_swiglu(DeviceTensor& gate, const DeviceTensor& up);

// RoPE in-place (NeoX half-split): rotates x [seq, n_heads*head_dim] using
// cos/sin tables [seq, head_dim].
void launch_rope(DeviceTensor& x, const DeviceTensor& cos, const DeviceTensor& sin,
                 int n_heads, int head_dim);
```

`kernels.h` stays host-includable; `<cuda_runtime.h>` and the `__global__` kernels live in
`kernels.cu`. Each launcher asserts shape contracts matching the CPU oracle, then calls
`CUDA_CHECK_KERNEL()` after the launch.

**Note on `launch_embed` and integer ids:** `DeviceTensor` currently stores `float*`. For
the embedding gather the caller allocates a small `int32` device buffer separately
(`cudaMalloc`/`cudaMemcpy` directly) and passes a raw `const int*` — no change to
`DeviceTensor` needed. This is a minor annoyance that is resolved when Step 6 wires the
full forward pass and can own the allocation.

## Validation — same harness, measure then pin

The CPU oracles are all already in `atlas_engine` (`atlas_cuda` links it). Flow per kernel,
per shape:

1. `fill_prng` deterministic inputs.
2. CPU oracle on host tensors → `ref`.
3. `to_device`, call launcher, `to_host` → `got`.
4. `compare(got, ref)` → `max-abs` / `mean-abs`; **measure on first green run, then pin**.

All four kernels are element-wise or gather — no reduction reordering, so the GPU↔CPU diff
is dominated by `__expf` vs `std::exp` in SwiGLU and `__cosf`/`__sinf` vs table lookup in
RoPE. Expected: all within `1e-5` or better; measured number is the record, not the guess.

`add` and `embed` are bit-exact (integer gather and FP32 addition without reordering) —
expected `max-abs = 0`; pinned at `0`.

### Test cases (`engine/cuda/tests/test_kernels.cu`)

**`launch_embed`**
- `vocab=100, H=64, seq=4` — small, aligned.
- `vocab=32000, H=2048, seq=8` — TinyLlama real dims.

**`launch_add`**
- `[4, 64]` small.
- `[8, 2048]` TinyLlama hidden.

**`launch_swiglu`**
- `[4, 64]` small.
- `[8, 5632]` TinyLlama intermediate (prefill).
- `[1, 5632]` decode.

**`launch_rope`**
- `[4, 32*64]` — q-shaped, small seq.
- `[1, 32*64]` decode q.
- `[8, 4*64]` — k-shaped (GQA kv heads).

Each case reports `max-abs`/`mean-abs` and asserts the pinned tolerance.

## Build & test workflow

- `engine/cuda/CMakeLists.txt`: add `kernels.cu` to `atlas_cuda` sources; add
  `test_kernels` executable + `add_test`, exactly like the `test_rmsnorm` block.
- `scripts/build_cuda.sh`: no change — builds the whole `atlas_cuda` target.
- `scripts/test_cuda.sh`: widen CTest filter to
  `-R 'test_device|test_matmul|test_rmsnorm|test_kernels'`.
- Same loop: `edit engine/cuda/ → build_cuda.sh → test_cuda.sh → iterate`,
  `CUDA_VISIBLE_DEVICES=0` on Suramar.

## Files created / touched

| File | Change |
|------|--------|
| `engine/cuda/kernels.h` | new — four `launch_*` declarations |
| `engine/cuda/kernels.cu` | new — four kernels + launchers |
| `engine/cuda/tests/test_kernels.cu` | new — validates all four against CPU oracles |
| `engine/cuda/CMakeLists.txt` | add `kernels.cu` to `atlas_cuda`; add `test_kernels` |
| `scripts/test_cuda.sh` | widen CTest filter to include `test_kernels` |
| `docs/10-cuda-kernels.md` | this spec |

## Done when

- All four launchers build into `atlas_cuda` via `scripts/build_cuda.sh`.
- `test_kernels` passes on-device: each GPU kernel matches its CPU oracle within its
  measured-then-pinned tolerance across all test shapes.
- `test_device`, `test_matmul`, and `test_rmsnorm` still pass (no regression).
- CTest reports 4/4 passing; `ATLAS_USE_CUDA=OFF` CPU build stays 10/10.

## Design decisions

- **One translation unit for all four.** These kernels are each 10–30 lines of device code;
  splitting them into four `.cu` files adds link targets without benefit. One `kernels.cu`
  keeps the build lean. If any kernel grows substantially (it won't — attention lives
  separately), it gets its own file then.
- **`launch_embed` takes raw `const int*` not `DeviceTensor`.** `DeviceTensor` is typed
  `float*`; adding a parallel int-tensor type for one use-site is premature. The caller
  manages the small `cudaMalloc`/`cudaFree` for the id buffer directly — the same pattern
  used for other scalar/index arguments in the CUDA ecosystem. Step 6 wraps this cleanly.
- **`add` and `embed` expected bit-exact.** The add is a straight FP32 element-wise sum
  with no reordering; the embed is an integer-indexed memcpy. Both should pin at `max-abs = 0`.
  If they don't on the first run, that is a bug, not a tolerance question.
- **`launch_rope` grid is `(seq * n_heads, 1)` with `BLOCK = head_dim/2 = 32`.** Each block
  handles exactly one (pos, head) pair; `head_dim/2 = 32` threads map onto one warp, so
  the rotation pairs are computed in a single warp with no shared memory needed. Clean, no
  partial warps for TinyLlama's fixed `head_dim = 64`.
- **Correctness before speed.** Same principle as Steps 2 and 3: earn correctness first, then
  the perf pass (vectorized loads, fused kernels) is a named follow-up that holds the same
  oracle + tolerance.
