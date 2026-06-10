# Phase 1 · Step 4 — Forward-Pass Validation

> Status: **done** — `test_forward` green with the tightened pins (see Measured results)
> Predecessor: Step 3 — model + weight loading + forward pass — **done** ([03-model-weights.md](03-model-weights.md))
> Successor: Step 5 — INT8 quantization (FP32 → INT8, measure the accuracy delta)

## Goal

Replace Step 3's "top-1 is `▁Paris`" smoke check with a rigorous, per-logit,
tolerance-checked comparison of the C++ forward pass against the HuggingFace golden
oracle `reference/logits.npy`. That requires the one piece of infrastructure Step 3
deferred: a **zero-dependency C++ `.npy` reader**.

This is the step that earns the right to say "the engine is correct" — every later step
(quantization's accuracy delta in Step 5, the CUDA kernels in Phase 2) measures itself
against an FP32 baseline that this step proves matches the reference. A top-1 match can
survive a subtly wrong RoPE or a transposed projection; a 192,000-element tolerance check
cannot.

**Why a tolerance and not exact equality:** both sides compute in FP32, but floating-point
addition is not associative — our naive triple-loop `matmul` accumulates in a different
order than PyTorch's BLAS kernels, and `expf`/`std::exp` in softmax need not be
bit-identical to torch's. Bitwise equality is unachievable and not the goal; agreement to
~1e-3 on logits of magnitude ~15 is.

## Definition of done

- `cmake --build` still green; `npy.cpp` joins `atlas_engine`.
- `test_forward` loads `reference/logits.npy` via the new reader and, in addition to the
  Step 3 shape/top-1 checks:
  - **max absolute difference** over all `6 × 32000` logits is `< 1e-2` (hard gate);
  - **mean absolute difference** is `< 1e-3` (catches a broad systematic drift that a
    max-only check on one lucky outlier could miss);
  - **per-row argmax** matches the reference for all 6 positions (the semantic check —
    greedy decoding agrees everywhere, not just at the last position).
  - The test **prints** the measured max/mean abs diff. After the first green run, tighten
    the pinned thresholds to ~10× the observed values and record them here — the gates
    above are ceilings, not targets.
- The reader is **self-tested against the second oracle**: load
  `reference/token_ids.npy` (int32) and check it equals the hard-coded
  `{1, 450, 7483, 310, 3444, 338}` — this exercises the integer dtype path and proves the
  header parsing on a file the test already knows the contents of.
- SKIP behavior unchanged: no weight blob → `test_forward` SKIPs green
  (`logits.npy` itself is committed, so it is never the missing piece).

## Measured results (first green run, 2026-06-09)

Over all 6 × 32000 logits:

| Metric | Measured | Pinned (≈10× measured) | Original ceiling |
|--------|----------|------------------------|------------------|
| max abs diff | 1.44e-4 | **< 1.5e-3** | 1e-2 |
| mean abs diff | 8.27e-6 | **< 1e-4** | 1e-3 |

Per-row argmax matched the reference at all 6 positions
(`529, 29871, 310, 278, 29892, 3681` — row 5 is `▁Paris`, logit 13.3885).

**Found along the way:** Step 3's `test_forward` checked the blob's existence with
`stat()`, which on MinGW is 32-bit and fails on the 4.4 GB blob — the test had been
silently SKIPping. Replaced with the `std::ifstream(path).good()` idiom (the same
gotcha Step 3 already fixed inside `model.cpp`).

## The `.npy` files on disk (what the reader must parse)

Both oracles are **NPY format version 1.0**, written by `np.save` in
`scripts/download_weights.py`:

| File | descr | shape | size |
|------|-------|-------|------|
| `reference/token_ids.npy` | `<i4` (int32 LE) | `(6,)` | 152 B |
| `reference/logits.npy` | `<f4` (float32 LE) | `(6, 32000)` | 768,128 B |

The v1.0 layout, in order:

1. **Magic**: 6 bytes `\x93NUMPY`.
2. **Version**: 2 bytes, major `\x01` minor `\x00`.
3. **Header length**: 2 bytes, little-endian `uint16` (`HEADER_LEN`).
4. **Header**: `HEADER_LEN` bytes of ASCII — a Python dict literal like
   `{'descr': '<f4', 'fortran_order': False, 'shape': (6, 32000), }`, space-padded and
   `\n`-terminated so the total preamble is 64-byte aligned. Don't rely on the padding
   width; just read exactly `HEADER_LEN` bytes.
5. **Data**: the raw array bytes, C-order (row-major), immediately after the header.

The reader does **not** need a Python-literal parser — extract the three keys with plain
string search (`find("'descr':")` etc.). It supports exactly what the oracles need and
**fails loudly** on anything else: version ≠ 1.0, `fortran_order: True`, or a `descr`
other than `<f4` / `<i4`. (Both are fine here: `np.save` writes C-order, and this machine
is little-endian — assert, don't handle.)

## The reader (`engine/include/npy.h` + `engine/src/npy.cpp`, new)

Two free functions in `namespace atlas`, following the tensor.h out-param-free style
(these allocate — loading is not a hot path):

```cpp
// reference/logits.npy -> owning FP32 Tensor (asserts descr == '<f4').
Tensor load_npy_f32(const std::string& path);

// reference/token_ids.npy -> ids (asserts descr == '<i4').
std::vector<int> load_npy_i32(const std::string& path);
```

Design points:

- **Lives in the engine lib, not the test tree.** Step 5 (quantize accuracy delta) and
  Phase 2 (CUDA kernel validation) will compare against `.npy` oracles too; one reader,
  reused. It is read-only — no `.npy` writer until something needs it.
- **Returns an owning `Tensor`** (the `owns=true` path) — these files are small (≤ 768 KB);
  no mmap, plain `std::ifstream` in binary mode. The MinGW 32-bit `stat()` gotcha from
  Step 3 doesn't apply at these sizes, but use the same `std::ifstream(path).good()`
  existence idiom for consistency.
- **Zero dependencies**, hand-rolled header parse — same ethos as the tokenizer and the
  safetensors converter: parse the format by hand, support only what the contract needs,
  assert the rest.

## The comparison (`engine/tests/test_forward.cpp`, extended)

Step 3's test already loads the model, runs the prompt, and checks shape + top-1. Step 4
appends, in the same executable (the 6-token forward is the expensive part — ~11 s — so
one run serves both the smoke checks and the tolerance check):

1. `ref = load_npy_f32(ATLAS_REFERENCE_DIR "/logits.npy")` — check shape `[6, 32000]`.
2. Single pass over all 192,000 elements accumulating `max|Δ|`, `Σ|Δ|` (accumulate the
   sum in `double` — 192k FP32 adds in FP32 would themselves add noise to the metric).
3. Per-row argmax of `ref` vs per-row argmax of the C++ logits, all 6 rows.
4. Print the measured `max_abs` / `mean_abs` and the per-row argmax ids, then `CHECK`
   the three gates from the definition of done.
5. Reader self-test: `load_npy_i32(.../token_ids.npy) == expected_ids`. This one runs
   **before** the weight-blob SKIP check — the oracle is committed, so the reader is
   exercised even on machines without the 4.4 GB blob.

## Files created / touched in this step

| File | Contents |
|------|----------|
| `engine/include/npy.h` (new) | `load_npy_f32`, `load_npy_i32` declarations. |
| `engine/src/npy.cpp` (new) | NPY v1.0 magic/version/header parse, dtype asserts, data read. |
| `engine/tests/test_forward.cpp` (edit) | reader self-test vs `token_ids.npy`; tolerance + per-row-argmax comparison vs `logits.npy`. |
| `engine/CMakeLists.txt` (edit) | add `src/npy.cpp` to `atlas_engine`. |

`tensor.*`, `tokenizer.*`, `model.*` are untouched — this step adds a measuring stick,
not model code. If the comparison fails, the fix happens in `model.cpp` and this step's
test is the proof it worked.

## Design decisions

- **Reader in the lib, not the tests** — it's shared infrastructure for Steps 4/5 and
  Phase 2 validation, and it keeps `test_forward.cpp` about the contract, not the format.
- **Extend `test_forward` rather than add a new test target** — the tolerance check needs
  the same ~11 s forward pass the smoke test already runs; two targets would double the
  slowest test for no isolation benefit.
- **Measure-then-pin tolerances.** `1e-2` max-abs / `1e-3` mean-abs are deliberately
  loose ceilings chosen to catch real bugs (a wrong RoPE split or GQA mapping moves
  logits by whole units, not milli-units) without flaking on FP32 accumulation-order
  noise. After the first green run the observed numbers get recorded here and the pins
  tightened, so a future regression that "only" degrades agreement 100× still fails.
- **`double` accumulation for the diff statistics** — the metric must be more precise
  than the thing it measures.
- **Support only NPY v1.0, little-endian, C-order, `<f4`/`<i4`** — assert and abort on
  anything else. Generality nobody uses is a from-scratch anti-goal.
- **Tokenizer-oracle self-test piggybacks here** — `test_tokenizer` keeps its committed
  plain-text fixtures (the Step 2 deliberate deviation stands); the `.npy` ids check in
  `test_forward` is about proving the *reader*, not re-proving the tokenizer.

## Explicitly deferred (not this step)

- INT8 quantization and its accuracy-delta measurement → **Step 5** (it will reuse this
  step's reader and comparison pattern).
- KV cache, multi-token generation loop, sampling → after Phase 1 validation.
- An `.npy` **writer**, NPY v2.0/3.0, big-endian or Fortran-order support → never, unless
  an oracle actually needs it.
- SIMD / multithreading / CUDA → Phase 2.

## Reference oracle

- `reference/token_ids.npy` — `(6,)` int32, `[1, 450, 7483, 310, 3444, 338]` — the
  reader's self-test fixture.
- `reference/logits.npy` — `(6, 32000)` float32, prompt "The capital of France is",
  produced by HF on CPU with `dtype=torch.float32` (`scripts/download_weights.py`) — the
  target of the tolerance check. Committed (768 KB), so this step's checks run anywhere;
  only the weight blob remains a local-only artifact.
