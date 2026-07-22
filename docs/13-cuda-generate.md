# Phase 2 · Step 7 — greedy decode loop (`GpuModel::generate()`, validated against the CPU loop)

> Status: **spec** — not yet implemented.
> Predecessor: Step 6 — full GPU forward pass — **done** ([12-cuda-forward.md](12-cuda-forward.md))
> Successor: Step 8 — pybind11 bridge + FastAPI streaming `/generate`, which closes Phase 2.

## Goal

Turn the proven one-shot prefill into a **completion**: prompt ids in, N generated ids out,
argmax-greedy, EOS-terminated. Step 6 produces a `[seq, vocab]` logit tensor and stops;
everything Atlas has printed so far is a single next-token prediction (`main.cpp:73`). This
step is the first time the project emits text.

The work is deliberately small. `GpuModel::forward()` is already validated end-to-end
against `reference/logits.npy` (2.04e-4 max-abs, per-row argmax matching HF at all six
positions), so the loop on top of it is ~10 lines: forward → argmax the last row → append →
repeat. The value of the step is not the loop but the **contract** around it — where the
loop lives, what stops it, how a token escapes to a caller mid-generation, and what oracle
proves it right.

### Scope boundary

- **Not a KV cache.** Each new token re-runs the *full* forward over the grown sequence.
  This is O(n²) in generated length and the honest cost is visible in the test's timing
  print. It buys the thing that matters here: **zero new kernels and zero new numerics**,
  so the loop's correctness reduces to an exact comparison against a CPU loop that shares
  its structure line-for-line. The KV cache (and the decode-mode attention kernel it needs)
  is the headline perf follow-up, taken up after the serving layer exists.
- **Not sampling.** Greedy argmax only, exactly as `main.cpp:65–71` already does. Greedy is
  deterministic, which is what makes token-for-token equality against the CPU a legitimate
  bar. Temperature / top-p belongs to the serving layer (`server/config.py` already plans
  the knob) and lands in Step 8 or later.
- **Not INT8.** FP32 weights only; `Model::qweights` stays ignored on the GPU path, same as
  Step 6. A quantization change here would make any token divergence ambiguous.
- **Not the bridge or the server.** No Python, no pybind11, no FastAPI — Step 8. The one
  concession to it is the `on_token` callback (below), declared now so streaming does not
  force an API change later.
- **Not chat templating.** Raw prompt ids in, raw ids out. TinyLlama-*Chat*'s
  `<|system|>/<|user|>/<|assistant|>` framing is a serving-layer concern.

## The operation

```
ids = prompt_ids
for step in 0 .. max_new_tokens-1:
    logits = forward(ids)            # [len(ids), vocab] — the Step 6 pass, unchanged
    next   = argmax_last_row(logits) # greedy
    if next == EOS: break            # EOS is not emitted
    out.push(next); ids.push(next)
    if on_token and !on_token(next): break
return out
```

That is the whole algorithm on both CPU and GPU. Cost per step grows with `len(ids)`: the
attention kernel is O(n²) in sequence length and the `lm_head` matmul is O(n·vocab·dim), so
a 64-token completion from a 6-token prompt does roughly 20x the work of the first forward
in its last step alone. Expect the test's per-token timing to climb monotonically; that
number *is* the argument for the KV cache and should be recorded, not hidden.

## Host API

### `engine/cuda/forward.h` — `GpuModel::generate()`

```cpp
// Greedy decode on top of forward(): argmax the last logit row, append, repeat.
//
// Stops at max_new_tokens, or when the model produces EOS (Tokenizer::kEosId) — the EOS
// id is NOT included in the result. Returns the generated ids only, without the prompt.
//
// on_token, when set, is called with each id the moment it is produced and returns false
// to stop early. This is the hook Step 8's FastAPI endpoint streams from; it is declared
// now so that adding streaming later needs no signature change. It is called before the
// next forward begins, so a caller that abandons the request stops paying for it
// immediately.
//
// No KV cache (docs/13): every step re-runs the full-prompt forward over the grown
// sequence, so per-token cost grows with length. Correctness first.
std::vector<int> generate(const std::vector<int>& prompt_ids,
                          int max_new_tokens,
                          const std::function<bool(int)>& on_token = {}) const;
```

`<functional>` joins the header's includes. The method is `const`, like `forward()` — all
per-call state is scratch inside `forward()`.

### `engine/include/model.h` / `engine/src/model.cpp` — two shared lifts

Same discipline as the `rope_tables()` extraction in Step 6 and the `reduce.cuh` extraction
in Step 5: anything both paths need lives in **one** place, so the two engines cannot
silently diverge.

```cpp
// Greedy next-token pick: index of the maximum logit in the LAST row of a [seq, vocab]
// logits tensor. Ties resolve to the lowest index (first-wins scan), which must match on
// both engines for GPU/CPU decode comparisons to be exact.
int argmax_last_row(const Tensor& logits);

// Greedy decode loop — the CPU twin of GpuModel::generate(), identical contract and
// identical structure. It is the oracle test_generate_gpu compares against, and what the
// CLI uses in the CPU-only build.
std::vector<int> generate(const std::vector<int>& prompt_ids,
                          int max_new_tokens,
                          const std::function<bool(int)>& on_token = {}) const;
```

`argmax_last_row` is the argmax currently open-coded in `main.cpp:66–71`; the tests carry
their own copies too. Lifting it also fixes the tie-break as a *stated* rule (first index
wins) rather than an accident of three separate loops — which matters, because the whole
validation bar below rests on both engines breaking ties the same way.

`Model::generate` is deliberately a **duplicate structure**, not a shared template
parameterized over the forward function. The loop is six lines; a `template <class Fwd>`
helper in a CUDA-free header that both `model.cpp` and `forward.cu` instantiate would save
those six lines and cost a header-level coupling between the CPU tree and the CUDA tree
that the Phase 2 guard discipline exists to prevent. The test asserts the two loops agree,
which is a stronger guarantee than sharing code would give (it also catches divergence in
`forward()` itself).

## Validation

### Test (`engine/cuda/tests/test_generate_gpu.cu`)

Structure and SKIP discipline copied from `test_forward_gpu`: npy self-test first, missing
`ATLAS_WEIGHTS_DIR` blob → **SKIP green**, paths from the same `ATLAS_REFERENCE_DIR` /
`ATLAS_WEIGHTS_DIR` compile definitions.

Load tokenizer + `Model`, `GpuModel::create(model)`, then, on the reference prompt ids
`{1, 450, 7483, 310, 3444, 338}` ("The capital of France is"):

1. **Oracle equality** — `gpu.generate(ids, 8)` vs `model.generate(ids, 8)`: the two id
   sequences must be **exactly equal**. A discrete bar, not a tolerance — there is nothing
   to pin here.
2. **Anchor** — the first generated id is `3681` ("Paris"), the value Step 6 measured and
   `docs/12` recorded. This makes the test fail loudly if the forward pass regresses,
   independent of whether CPU and GPU regress together.
3. **Decoded text** — the completion round-trips through `Tokenizer::decode`; printed for
   eyeballing, asserted only as non-empty (the continuation past "Paris" is not a
   contract — TinyLlama-1.1B base behaviour, not a fixed string).
4. **Determinism** — two consecutive `gpu.generate(ids, 4)` calls return identical ids.
   Cheap, and it is the assertion that would catch per-call scratch being read before it is
   fully written.
5. **Stopping conditions** —
   - `generate(ids, 4)` returns exactly 4 ids (the budget is honoured, no off-by-one);
   - an `on_token` that returns `false` on its 2nd call returns exactly 2 ids;
   - `generate(ids, 0)` returns empty and runs no forward.
   EOS termination is asserted **structurally**, not by finding a prompt that ends: 8 greedy
   tokens from this prompt will not hit EOS, so the test drives the EOS path directly by
   checking that `Tokenizer::kEosId` never appears in any returned sequence and leaves the
   break itself covered by inspection. (If a natural EOS turns up during bring-up, promote
   it to a real assertion and record the prompt here.)
6. **Tie-break margin diagnostic** — exact argmax equality is only a sound bar if the top-2
   logit gap at each step comfortably exceeds the GPU-vs-CPU drift, which `docs/12`
   measured at `max_abs=8.01e-5`. So the test **prints the top-1 minus top-2 margin at
   every step**. Margins in the 1e-1..1e+1 range mean the bar is safe by four-plus orders of
   magnitude. If a mismatch ever appears, the rule is: it is a **bug signal, not a
   tolerance to relax** — check the margin at the diverging step first, and only if it is
   genuinely near the drift floor is "the models disagree on a coin flip" the right reading.
7. **Wall-clock, informational** — total generate time and per-token time for GPU and CPU,
   printed, never asserted (shared box, noisy). Print per-step GPU times so the growth with
   sequence length is visible. From Step 6's 0.053 s prefill at seq 6, expect roughly
   0.05–0.1 s per GPU token at these lengths and ~7–15 s per CPU token — the CPU oracle run
   dominates the test's wall time, so keep `max_new_tokens` small (8) in the equality check.

### Expected result

No new numerics are introduced, so there is nothing to measure-then-pin in this step — the
only new failure modes are logic (off-by-one, EOS handling, callback semantics) and they
are all binary. Record the observed completion text and the per-step margins here on the
first green run for future reference.

## Build & test workflow (unchanged loop)

- `engine/cuda/CMakeLists.txt`: add a `test_generate_gpu` executable + `add_test`, with the
  `ATLAS_REFERENCE_DIR` / `ATLAS_WEIGHTS_DIR` compile definitions copied from
  `test_forward_gpu`'s block. No new source joins `atlas_cuda` — `generate()` goes in the
  existing `forward.cu`.
- `scripts/test_cuda.sh`: widen the CTest filter to include `test_generate_gpu` (→ 7/7).
- Same loop: `edit engine/cuda/ → build_cuda.sh → test_cuda.sh → iterate`, with a card
  pinned via `CUDA_VISIBLE_DEVICES` (check `nvidia-smi` for the freer one).
- The CPU build (`ATLAS_USE_CUDA=OFF`) must stay 10/10; `test_forward` is the guard proving
  the `argmax_last_row` lift changed nothing.

## CLI demo

`engine/src/main.cpp` gains `-n N` (default 1, so today's output is preserved verbatim) to
print a real completion via `Model::generate`, and drops its open-coded argmax in favour of
`argmax_last_row`. CPU-only and slow at ~7 s per token, but it makes Phase 2's progress
demonstrable without a server, and it exercises `Model::generate` in the CPU build where
the CUDA tests cannot reach.

## Performance follow-ups (named, deferred)

Inherited from `docs/12` and amplified by a loop that calls `forward()` N times:

- **KV cache + a decode-mode attention kernel** (query length 1 over cached K/V, plus
  persistent per-layer K/V buffers) — the real fix for the O(n²) loop and the largest
  remaining Phase 2 perf item. Held to this step's oracle: the cached loop must produce the
  identical id sequence.
- **Last-row-only `lm_head`** — decode needs `[1, vocab]`, not `[seq, vocab]`; already named
  in `docs/12` and now has a concrete caller.
- **Persistent scratch** at max_seq instead of per-call `DeviceTensor::alloc` — N
  allocations per completion today.
- **Drop per-launch syncs** — ~230 device syncs per forward × N forwards per completion.

## Files created / touched

| File | Change |
|------|--------|
| `engine/cuda/forward.h` | `GpuModel::generate()` declaration; `<functional>` |
| `engine/cuda/forward.cu` | `GpuModel::generate()` definition |
| `engine/include/model.h` | `argmax_last_row()` + `Model::generate()` declarations |
| `engine/src/model.cpp` | their definitions |
| `engine/src/main.cpp` | `-n N`; use `argmax_last_row` |
| `engine/cuda/tests/test_generate_gpu.cu` | new — vs the CPU loop |
| `engine/cuda/CMakeLists.txt` | `test_generate_gpu` target + path defines |
| `scripts/test_cuda.sh` | widen the CTest filter |
| `docs/13-cuda-generate.md` | this spec |

## Done when

- `GpuModel::generate()` builds into `atlas_cuda` via `scripts/build_cuda.sh`.
- `test_generate_gpu` passes on-device with the blob present: exact id equality with the
  CPU loop, "Paris" as the first token, determinism, and all three stopping conditions;
  per-step margins and timings reported.
- `test_generate_gpu` SKIPs green without the blob.
- The six existing CUDA tests still pass — CTest 7/7 in `build-cuda/`.
- The CPU build (`ATLAS_USE_CUDA=OFF`) stays 10/10, `test_forward` proving the
  `argmax_last_row` lift changed nothing.
- `atlas` CLI with `-n 12` prints a multi-token completion.

## Design decisions

- **Step 7 is the decode loop alone; the bridge and server become Step 8.** `docs/12`'s
  successor line bundled "generation loop → pybind11 → FastAPI" into one step. Split here
  for the same reason Step 1 split infra from the first kernel: three failure domains in
  one step means a red test does not tell you which layer broke. The decode loop is also
  the piece with a *rigorous* oracle available (the CPU engine); the server's correctness
  bar is a different kind of thing entirely.
- **No KV cache in this step.** The cache is the single largest remaining perf win and
  genuinely deserves its own step — it needs a new attention kernel variant, persistent
  per-layer device buffers, and its own numerics validation. Landing it *inside* the loop's
  bring-up step would mean a first failure could be the loop or the cache, with no way to
  bisect. The cacheless loop is correct, provably so, and gives the cache a token-exact
  oracle to be held against.
- **Greedy only.** Determinism is what makes "the two engines produce identical ids" a
  usable bar. Sampling would replace it with a distributional comparison — far weaker for
  far more machinery — and the temperature knob's natural home is the serving config
  anyway.
- **`on_token` declared now, unused by everything except the test.** Streaming is a stated
  Phase 2 requirement (`/generate` streams tokens as produced). Adding the callback later
  would change the signature of the only public generation entry point after the bridge is
  already binding it. One `std::function` parameter with a default costs nothing today and
  removes a breaking change from Step 8.
- **EOS excluded from the returned ids.** Callers want text; `Tokenizer::decode` skips
  specials anyway, so including it would be invisible in the output but would corrupt the
  count semantics of `max_new_tokens` and confuse a streaming consumer.
- **`argmax_last_row` lifted, tie-break fixed as first-index-wins.** Three open-coded
  copies were already drifting toward a tie-break disagreement that would surface as a
  mysterious one-token divergence between engines. One definition, one stated rule.
- **`Model::generate` duplicates the loop rather than sharing a templated helper.** Six
  lines of duplication against a header-level CPU↔CUDA coupling; the guard discipline wins,
  and the equality assertion covers the duplication better than sharing would.
