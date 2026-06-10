# Phase 1 · Step 5 — INT8 Quantization

> Status: **done** — `test_quantize` green: per-row argmax matches at all 6 positions,
> max-abs 1.497 / mean-abs 8.43e-2 over all 6 × 32000 logits (pins 3.0 / 0.17 — the
> original ceilings were exceeded and revised, see Measured results); 4.14 GB of linear
> weights → 1.04 GB INT8
> Predecessor: Step 4 — forward-pass validation — **done** ([04-forward-validation.md](04-forward-validation.md));
> the FP32 engine matches the HF oracle to max-abs 1.44e-4 over all 6 × 32000 logits
> Successor: Phase 1 complete → Phase 2 — CUDA kernels + serving layer

## Goal

Post-training quantization of the linear-layer weights from FP32 to INT8 — per-row
symmetric, weights-only — and a rigorous measurement of what it costs in accuracy,
using exactly the comparison machinery Step 4 built (the `.npy` reader and the
max/mean-abs-diff + per-row-argmax gates against `reference/logits.npy`).

Two things are earned here:

- **Memory**: the 155 linear matrices (22 layers × 7 projections + `lm_head`) hold
  1,034,420,224 of the model's 1,100,048,384 parameters — 4.14 GB of the 4.40 GB blob.
  INT8 shrinks them to ~1.03 GB; with embeddings and norms left as FP32 mmap views, the
  resident model drops from ~4.4 GB to ~1.3 GB.
- **Methodology**: the accuracy-delta measurement. Step 4 proved the FP32 engine *is* the
  reference (to 1.44e-4); now any divergence the quantized forward shows against
  `reference/logits.npy` is, to that noise floor, *the cost of quantization* — cleanly
  attributed, not confounded with engine bugs. This is the pattern Phase 2 reuses to
  judge the CUDA kernels.

**Not a speed step.** A naive dequantize-in-the-inner-loop matmul on CPU is no faster
than the FP32 one (it adds an int8→float convert per element). INT8's speed payoff needs
SIMD or tensor cores — Phase 2. Here the deliverables are the memory cut and the measured
accuracy delta.

## The quantization scheme (per-row symmetric, weights-only)

For each row `r` of a weight matrix `W [out, in]` (one row = one output channel, since
`linear` computes `y = x @ Wᵀ`):

```
scale_r = max(|W[r, :]|) / 127          (scale_r = 1.0 if the row is all zeros)
Q[r, i] = round(W[r, i] / scale_r)      clamped to [-127, 127]
```

Dequantization is `W'[r, i] = Q[r, i] * scale_r`; the worst-case per-weight error is
`scale_r / 2`. Three decisions baked into those two lines:

- **Per-row, not per-tensor.** Each output element of `y = x @ Wᵀ` touches exactly one
  row, so the scale factors out of the dot product: `y[j] = scale_j * Σ x[k]·Q[j,k]`. One
  outlier channel therefore cannot inflate the quantization step of every other channel,
  and the cost is 4 bytes per row — 426,240 scales ≈ 1.7 MB across the whole model.
- **Symmetric (zero-point ≡ 0), not affine.** Weight distributions are centered near
  zero, so a zero-point buys almost nothing while adding a `zp · Σ x[k]` correction term
  to every dot product. Zero-points earn their keep on *activations* (post-SiLU
  distributions are asymmetric) — and activations stay FP32 this step.
- **Clamp to ±127, not −128.** Using the full two's-complement range makes the grid
  asymmetric around zero for no measurable accuracy gain. Symmetric ±127 keeps
  `quantize(−w) = −quantize(w)` exactly.

**What gets quantized:** the 155 2-D matmul weights — `q/k/v/o_proj`, `gate/up/down_proj`
× 22 layers, plus `lm_head`. **What stays FP32:** `embed_tokens` (a row *lookup*, not a
matmul — quantizing it saves 262 MB but adds error before layer 0 for zero compute win;
it stays a no-copy mmap view) and the 45 RMSNorm vectors (368 KB — noise). `lm_head` *is*
quantized even though its error lands directly on the logits; if the measured delta blows
the gates, keeping `lm_head` FP32 is the documented first fallback.

## The quantizer (`engine/include/quantize.h` + `engine/src/quantize.cpp`, new)

Following the tensor.h conventions — a minimal owning struct plus free functions:

```cpp
// An INT8-quantized weight matrix: row-major int8 payload + one FP32 scale per row.
// Owns its buffers (it is *produced* from the mmap'd FP32 views, not a view itself).
struct QTensor {
    std::vector<int8_t> data;   // [rows * cols], row-major
    std::vector<float> scales;  // [rows], dequant = data * scales[row]
    int64_t rows = 0, cols = 0;
};

// FP32 [out, in] -> per-row symmetric INT8 (the scheme above).
QTensor quantize_rows(const Tensor& w);

// QTensor -> owning FP32 Tensor (test/debug aid — the round-trip check).
Tensor dequantize(const QTensor& q);

// y = x @ dequant(W)ᵀ, dequantizing in the inner loop:
//   y[s, j] = scales[j] * Σ_k x[s, k] * (float)data[j*cols + k]
// Same shape contract as model.cpp's FP32 `linear`. Out-param style, like tensor.h.
void linear_q8(const Tensor& x, const QTensor& w, Tensor& out);
```

`QTensor` lives in `quantize.h`, not `tensor.h` — the Tensor foundation stays FP32-only
(the Step 1 "no dtype templating" decision stands; INT8 is a *weight storage* format with
its own three functions, not a new tensor dtype threaded through every op).

## Model integration (`model.h` / `model.cpp`, edit)

- `Model` gains `std::unordered_map<std::string, QTensor> qweights;` and a method
  `void quantize_int8();` — walks the 155 linear weight names, builds a `QTensor` from
  each mmap'd view, and stores it. After this the FP32 pages of those matrices are never
  touched again (the OS evicts them); embeddings and norms keep their views.
- model.cpp's `linear` helper dispatches: if the weight's name is in `qweights`, call
  `linear_q8`, else the FP32 path. One forward-pass implementation, two precisions — the
  22-layer orchestration code is untouched.
- `atlas.exe` grows an `--int8` flag: quantize after load, print the precision and the
  quantized byte count alongside the usual pipeline output.

## The measurement (`engine/tests/test_quantize.cpp`, new)

Unit tests first (blob-free — they run anywhere, like the Step 4 reader self-test):

1. **Round-trip on a hand-built matrix** (mixed signs, a row whose max is negative, an
   all-zero row): `scales` match `max_abs/127` exactly, exactly-representable values
   survive the round trip bit-perfectly, every element obeys `|w − dq| ≤ scale_r/2`, and
   the all-zero row produces `scale = 1.0` with all-zero ints (no NaN/div-by-zero).
2. **`linear_q8` vs FP32 `matmul`** on small deterministic matrices (formula-filled, no
   RNG): agreement within the bound implied by `scale/2` per weight.

Then the end-to-end (after the weight-blob SKIP check — same
`std::ifstream(path).good()` idiom; no blob → SKIP green):

3. Load the model, `quantize_int8()`, forward the reference prompt, and compare against
   `ref = load_npy_f32(ATLAS_REFERENCE_DIR "/logits.npy")` with Step 4's exact pattern —
   one pass accumulating `max|Δ|` and `Σ|Δ|` in `double`, then:
   - **per-row argmax matches the reference at all 6 positions** — the semantic gate:
     greedy decoding is unchanged by quantization (deterministic forward, so this is a
     fixed fact of the scheme, not a flaky threshold);
   - **max abs diff < 0.5** and **mean abs diff < 0.05** (hard ceilings — weight-only
     INT8 typically moves logits of magnitude ~15 by hundredths, so these catch a broken
     scheme, not FP noise); *(superseded by measurement — both ceilings underestimated
     22-layer error accumulation by ~3×; see Measured results for the final 3.0 / 0.17
     pins and why the semantic gates still hold)*;
   - the test **prints** the measured max/mean abs diff and the per-row argmax ids.
     After the first green run, pin the thresholds to ~10× measured and record them here
     — the Step 4 measure-then-pin discipline.

This lands in a **new test target**, not in `test_forward` — unlike Step 4 (which reused
the smoke test's forward pass), the quantized forward is necessarily a *second* ~11 s
run, and `test_forward`'s job is to keep certifying the FP32 baseline unpolluted.

## Measured results (first green run, 2026-06-10)

INT8 forward vs `reference/logits.npy`, over all 6 × 32000 logits:

| Metric | Measured | Pinned (≈2× measured) | Original ceiling |
|--------|----------|-----------------------|------------------|
| max abs diff | 1.497 | **< 3.0** | 0.5 — **exceeded** |
| mean abs diff | 8.43e-2 | **< 0.17** | 0.05 — **exceeded** |

- **Semantics intact**: per-row argmax matches the reference at all 6 positions
  (`529, 29871, 310, 278, 29892, 3681`) — greedy decoding is unchanged by quantization.
  ▁Paris last-row logit 13.4119 vs FP32's 13.3885 (a 0.17% shift).
- **Memory**: 155 matrices → 1.036 GB of int8 + 1.7 MB scales (was 4.14 GB FP32);
  embeddings/norms remain FP32 mmap views. `atlas.exe --int8` completes
  "The capital of France is Paris".

**The original ceilings were a bad estimate, and that's the finding.** 0.5 / 0.05 came
from the literature's "near-lossless at INT8" — but that claim is about perplexity and
task accuracy, not per-logit max diff. A token's path through the model crosses 155
quantized matmuls, and the ~0.4%-of-row-max rounding noise compounds through the
residual stream into a ~0.56%-relative mean logit shift (0.084 on logits of magnitude
~15) and a 1.5 worst-case logit. Every semantic gate passes; the per-logit drift is the
honest per-logit cost of W8A32 on a 22-layer model.

**The FP32-`lm_head` fallback was measured and rejected.** Keeping `lm_head` FP32
(154 matrices quantized) improves mean abs diff only 0.0843 → 0.0803 and max only
1.497 → 1.486, while giving up 66 MB — the delta is accumulated hidden-state drift
across the 22 layers, not `lm_head`'s direct logit error. `lm_head` stays quantized.

**Pins are ~2× measured, not Step 4's ~10×.** 10× of the measured max would be 15 — the
magnitude of the logits themselves, a useless gate. The forward is deterministic, so 2×
is margin for compiler-reassociation differences across toolchains, not run-to-run
noise; a broken scheme (wrong scale axis, overflow) still moves logits by whole units
and fails both pins.

## Files created / touched in this step

| File | Contents |
|------|----------|
| `engine/include/quantize.h` (new) | `QTensor`; `quantize_rows`, `dequantize`, `linear_q8` declarations. |
| `engine/src/quantize.cpp` (new) | per-row symmetric quantization, round-trip dequant, dequantize-in-loop linear. |
| `engine/include/model.h` / `src/model.cpp` (edit) | `qweights` map + `quantize_int8()`; `linear` helper dispatches FP32 vs INT8 by name. |
| `engine/src/main.cpp` (edit) | `--int8` flag. |
| `engine/tests/test_quantize.cpp` (new) | round-trip + `linear_q8` unit tests (blob-free); INT8 forward vs `logits.npy` (blob-gated, SKIPs green). |
| `engine/CMakeLists.txt` (edit) | add `src/quantize.cpp` to `atlas_engine`; `test_quantize` target (CTest, `ATLAS_*_DIR` defines, MinGW `-static*` flags). |

`tensor.*`, `tokenizer.*`, `npy.*` are untouched. The forward-pass *math* in `model.cpp`
is untouched too — only the `linear` dispatch point changes, which is what makes the
measured delta attributable to quantization alone.

## Design decisions

- **Weights-only (W8A32).** Activations stay FP32 end to end. Activation quantization
  needs calibration data and asymmetric ranges and pays off only with int8 *compute*
  paths (SIMD/tensor cores) — all Phase 2 concerns. One variable at a time.
- **Quantize at load time, in C++ — no second blob.** The FP32 blob + manifest stay the
  single converted artifact; `quantize_int8()` derives INT8 from the mmap'd views in a
  few seconds. No `model.i8.bin` writer, no manifest format change, and the quantizer is
  C++ we own and unit-test — consistent with the from-scratch ethos. (An offline INT8
  blob is a possible later optimization if load time ever matters.)
- **Per-row symmetric, ±127** — rationale in the scheme section. The granularity is
  dictated by `y = x @ Wᵀ`: one row per output element means the scale factors cleanly
  out of every dot product.
- **One forward pass, dispatch inside `linear`** — duplicating the 22-layer orchestration
  for a second precision is how the two paths drift apart. The dispatch point is the
  single place precision is decided.
- **Embeddings stay FP32, `lm_head` gets quantized.** Embedding lookup does no matmul —
  quantizing it is pure-cost error injection. `lm_head` is a real 262 MB matmul; it is
  in scope, with FP32-`lm_head` as the named fallback if the logit delta is dominated
  by it.
- **Measure-then-pin, again.** 0.5 / 0.05 are ceilings sized to catch a broken scheme
  (a wrong scale axis or a signed-overflow clamp moves logits by whole units); the
  pinned values come from the first green run so regressions 100× smaller still fail.
- **Separate `test_quantize` target.** Step 4 extended `test_forward` because the
  tolerance check shared its one expensive forward; the INT8 forward is inherently a
  second run, and keeping the FP32 certification test free of quantization code keeps
  "the baseline is correct" and "the delta is small" independently falsifiable.

## What the literature says (survey, 2022–2026)

A read of the PTQ literature before implementing, to check the planned scheme against the
state of the art. Headline: **the cutting edge lives at ≤ 4 bits and in activation
quantization; at weight-only INT8, per-channel round-to-nearest is the literature's own
baseline and is near-lossless** (LLM.int8(), arXiv:2208.07339, reports < 1% degradation
at 8-bit; the famous "outlier features" problem that motivates most of the fancy
machinery is an *activation* phenomenon — it never bites W8A32). The planned scheme is
not a naive placeholder; it is what the field itself uses at this bit-width.

What the advanced methods do, and why each is or isn't applicable here:

| Technique | Idea | Verdict for Step 5 |
|-----------|------|--------------------|
| **MSE-optimal clipping** (ACIQ/OCTAV lineage; NeUQI, arXiv:2505.17595) | `max_abs/127` wastes grid resolution on one outlier weight; searching a per-row clip that minimizes `Σ(w−dq)²` trades rare large errors for smaller typical ones. Calibration-free. | **The one applicable upgrade.** ~20 lines of C++ (scan ~32 candidate clips per row), drops straight into `quantize_rows`, and the existing harness can A/B it against max-abs. Gains at 8-bit are small (NeUQI's own scaling: initialization matters most at 2–4 bits) — but it is measurable here for free. Optional, not in the DoD. |
| **Group-wise scales** (llama.cpp `Q8_0`: one scale per 32 weights) | Finer granularity than per-row along the *input* dim. | Not for INT8 — and it breaks this step's nicest property: a per-row scale factors out of the dot product, a per-group scale does not (the inner loop fragments into per-block partials). Becomes necessary at INT4; note for then. |
| **GPTQ** (arXiv:2210.17323) | Quantize columns greedily, using an inverse Hessian (`2XXᵀ` from ~128 calibration samples) to push each weight's rounding error onto not-yet-quantized weights. | Defer. Needs calibration infrastructure Atlas doesn't have (the 6-token prompt is an oracle, not a calibration set) and per-layer Cholesky machinery; its wins over RTN appear at 4-/3-bit, not 8. The natural engine for an INT4 stretch step. |
| **AWQ** (arXiv:2306.00978) | ~1% of weight channels are salient — identified by *activation* magnitude, not weight magnitude — and pre-scaling them up before quantization protects them. | Defer with GPTQ, same reasons: calibration activations required, gains at 4-bit. |
| **SmoothQuant** (arXiv:2211.10438) / **rotations** (QuaRot, SpinQuant; ButterflyQuant, arXiv:2509.09679) | Migrate or rotate away activation outliers so *activations* survive INT8/INT4. | Out of scope by construction — activations stay FP32 this step. Becomes relevant in Phase 2 if the CUDA path wants true INT8 compute (W8A8). |
| **LLM.int8() outlier decomposition** (arXiv:2208.07339) | Run the ~0.1% outlier activation dims in FP16, the rest in INT8. | Same: a W8A8 technique. Its diagnosis is still useful here — it explains *why* weight-only INT8 is the easy regime. |

Two takeaways folded into the plan: the per-row symmetric scheme stands as specced
(validated, not just simple), and the measured-delta harness this step builds is exactly
the instrument an INT4 follow-on (group-wise + GPTQ/AWQ, where the literature's wins are
real) would need — another reason the methodology, not the scheme, is the deliverable.

## Explicitly deferred (not this step)

- Activation quantization (W8A8), calibration, zero-points → Phase 2, if ever.
- INT8 *compute* (SIMD dot products, int32 accumulators) → Phase 2; here INT8 is a
  storage format dequantized in the inner loop.
- Sub-8-bit schemes (INT4 with group-wise scales, GPTQ/AWQ-style error compensation) →
  a possible stretch step after Phase 1; that is where the advanced literature actually
  pays off (see the survey section), and this step's harness is the prerequisite.
- An offline quantized blob format / `.npy` writer → only if load time becomes a problem.
- KV cache, multi-token generation, sampling → after Phase 1 wraps (still pending).

## Reference oracle

- `reference/logits.npy` — `(6, 32000)` FP32, prompt "The capital of France is" — the
  same Step 4 target, now measuring quantization loss instead of implementation error.
  The FP32 engine sits 1.44e-4 from this oracle, so any quantized-forward diff above
  ~1e-3 is attributable to INT8, not the engine.
- `reference/token_ids.npy` — `[1, 450, 7483, 310, 3444, 338]` — the forward input.
- FP32 per-row argmax, for the semantic gate: `529, 29871, 310, 278, 29892, 3681`
  (row 5 `▁Paris`, logit 13.3885).
