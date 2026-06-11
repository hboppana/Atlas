# Phase 1 · Addendum — Test-Suite Hardening (4 → 10 targets)

> Status: **done** — 10/10 CTest targets green; the 8 blob-free targets run in < 2 s
> Predecessor: Step 5 — INT8 quantization — **done** ([05-quantization.md](05-quantization.md))
> Successor: Phase 2 — CUDA kernels + serving layer (these component tests become the
> per-kernel CPU references)

## Goal

Before Phase 2's heavy refactoring (KV cache, CUDA kernels, the POSIX mmap path), close
the structural gap in the Phase 1 suite: the e2e oracle tests (`test_forward`,
`test_quantize`) are strong but **blob-gated** — on any machine without the 4.4 GB local
blob (CI included), the model math had zero coverage. And when an e2e diff appears, it
says *something* broke, not *what*.

The expansion adds 6 blob-free targets so every architectural risk named in docs/03 —
the RoPE half-split, the GQA head mapping, the mmap/manifest contract — has an isolated,
hand-checkable test that fails with a per-element report.

## The suite (10 targets)

| Target | Covers | Oracle | Blob? |
|--------|--------|--------|-------|
| `test_tensor` | Tensor foundation + **new**: `view` aliasing, move-assign, 3-D strides | hand-computed | no |
| `test_tokenizer` | exact ids vs committed fixtures | `token_ids.npy` path | no |
| `test_tokenizer_edges` (new) | BOS contract, decode∘encode == identity on awkward inputs (space runs, unicode/byte-fallback, `\n`/`\t`, empty), **idempotence sweep over all 32000 vocab entries** | properties | no |
| `test_model_math` (new) | `linear`, `rmsnorm` (incl. the no-mean-subtraction discriminator vs LayerNorm), `swiglu` | hand-computed | no |
| `test_rope` (new) | half-split rotation vs the closed form in double at multiple positions; pos-0 identity; norm preservation | closed form | no |
| `test_attention` (new) | causal mask, GQA mapping (heads {0,1}→kv0 / {2,3}→kv1 on a 4:2 config), uniform-softmax running mean, peaked softmax, seq=1 | hand-computed | no |
| `test_npy` (new) | NPY v1.0 parse on **synthetic files written byte-by-byte by the test** (incl. a non-64-aligned header); 7 death cases (bad magic/version, fortran, descr, truncation, dtype mismatch, missing file) | by construction | no |
| `test_weightstore` (new) | mmap + manifest views on a synthetic 56-byte blob: shapes, values, pointer contiguity, `owns == false`; 5 death cases (missing name, misaligned offset, past-end, empty manifest, missing blob) | by construction | no |
| `test_forward` | FP32 e2e vs HF oracle (Step 4 pins: max-abs < 1.5e-3, mean-abs < 1e-4) | `logits.npy` | **yes** |
| `test_quantize` | quant units + INT8 e2e (Step 5 pins: max-abs < 3.0, mean-abs < 0.17, argmax ×6) | `logits.npy` | **yes** |

Blob-free coverage went from 2 targets (tensor, tokenizer) to **8**, which is what a
future CI job runs. The 8 complete in under 2 seconds.

## Enabling refactor (model.cpp, behavior-neutral)

`linear`, `rmsnorm`, `rope` left model.cpp's anonymous namespace, and `attention` +
`swiglu` were **extracted from `forward`** — all five now declared in `model.h` under
"model math building blocks". Two consumers beyond `forward()`: these component tests,
and Phase 2 CUDA validation, which compares each kernel against these CPU references one
component at a time. The loop bodies moved verbatim; `test_forward`/`test_quantize`
passing at unchanged pins after the move is the behavior-neutrality proof.

`forward`'s attention sublayer is now four calls (rmsnorm → q/k/v proj → rope →
attention → o_proj) and the SwiGLU block is one (`swiglu(gate, up)`).

## Design decisions

- **Discriminating tests, verified by breaking the code.** `test_rope` was checked
  against the classic bug: temporarily switching model.cpp to interleaved-pair rotation
  made it fail (and `test_model_math` stayed green — correct isolation), then the change
  was reverted. A test that can't fail on the bug it targets is decoration.
- **Death tests via self-subprocess.** `die()` calls `exit(1)`, so failure modes can't
  be tested in-process. Each format test re-invokes its own exe (`argv[0]`) with a case
  name; the child runs the death scenario and the parent CHECKs a non-zero exit. The
  child *returns 0* if the loader wrongly accepts bad input, which the parent flags —
  the polarity that catches a silently-permissive parser.
- **Synthetic fixtures written by the test, not committed.** `test_npy` builds NPY
  byte streams in code and `test_weightstore` writes a 56-byte blob + manifest — the
  expected parse is known by construction, nothing new is committed, and the
  mmap/manifest path finally runs without the 4.4 GB blob.
- **Properties for the tokenizer, not pinned ids.** decode∘encode == identity compares
  *text*, so encode is free to re-segment; the exact-id pins stay in `test_tokenizer`
  against the committed fixtures. The vocab sweep skips ids that decode to a lone
  non-ASCII/control byte (raw `<0xNN>` tokens aren't valid UTF-8 on their own; the
  byte-fallback path is covered by the multi-byte unicode round trips instead).
- **`atlas_add_test()` CMake helper** replaced the per-target boilerplate: one line per
  target, uniform `ATLAS_*_DIR` defines and MinGW static-link flags.
- **Deliberate non-tests** (assessed and rejected): the Python converters (validated
  transitively — a wrong BF16 upcast cannot pass `test_forward`'s 1.44e-4 gate), a CLI
  smoke target (thin compose of tested parts, costs a ~22 s forward), debug-only
  asserts (compiled out in Release), performance gates (flaky; Phase 2 benchmarks own
  perf), NaN-weight handling (assert-don't-handle ethos).

## Files created / touched

| File | Change |
|------|--------|
| `engine/include/model.h` / `src/model.cpp` | building blocks exposed; `attention`/`swiglu` extracted (behavior-neutral) |
| `engine/tests/test_model_math.cpp`, `test_rope.cpp`, `test_attention.cpp`, `test_npy.cpp`, `test_weightstore.cpp`, `test_tokenizer_edges.cpp` | new targets |
| `engine/tests/test_tensor.cpp` | + view aliasing, move-assign, 3-D strides |
| `engine/CMakeLists.txt` | `atlas_add_test()` helper; 10 registrations |
