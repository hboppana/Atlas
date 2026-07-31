# Phase 2 · Step 8 — pybind11 bridge + FastAPI streaming `/generate`

> Status: **spec** — not yet implemented.
> Predecessor: Step 7 — greedy decode loop — **done** ([13-cuda-generate.md](13-cuda-generate.md))
> Successor: **Phase 3** — the RAG pipeline (`rag/`), which is the first consumer of this
> surface. Step 8 closes Phase 2.

## Goal

Make the engine **callable from Python**, and make a completion **streamable over HTTP**.

Everything Atlas can do today it does from a C++ `main()`. `GpuModel::generate()` is proven
token-exact against the CPU oracle (docs/13), but the only way to reach it is the `atlas`
CLI. Phases 3–5 — RAG, the LangGraph agent, the MCP server — are all Python, and all of
them assume an inference surface that can be imported or called over HTTP. There is no
`server/` directory yet. This step builds it.

Two layers, one step:

1. **`_atlas_engine`** — a pybind11 extension module over the existing C++ classes.
   Python never reimplements inference; it calls `Tokenizer::encode`, `Model::generate` /
   `GpuModel::generate`, `Tokenizer::decode`, and nothing else.
2. **`server/`** — a thin `bridge.py` façade over that module, and a FastAPI `serve.py`
   whose `POST /generate` streams tokens as they are produced.

The step is bundled but **not blurred**: the bridge is validated by a pytest that never
imports FastAPI, so a red test names one layer. That is the same failure-domain discipline
that split Step 1 (infra before kernels) and Step 7 (decode loop before serving) — applied
here as an internal boundary rather than a step boundary, because the bridge alone is not a
shippable artifact and the two layers share exactly one contract (`Engine.stream()`).

The `on_token` callback that Step 7 declared "for Step 8's FastAPI endpoint" is finally
used. No engine signature changes in this step — that was the point of declaring it early.

### Scope boundary

- **Not sampling.** Greedy argmax only, inherited from docs/13. `config.py` gets **no
  temperature knob**: a parameter that silently does nothing is worse than an absent one.
  Temperature / top-p arrives with the sampling work, and adds a field then.
- **Not a KV cache.** Every generated token still re-runs the full forward over the grown
  sequence. The server therefore inherits ~0.054 s/token on the A6000 and O(n²) growth.
  This is the honest number and the endpoint should report it, not hide it. The cache
  remains the headline Phase 2 perf follow-up.
- **Not batching.** One model instance, one in-flight generation, serialized by a lock
  (below). Continuous batching is a named follow-up, not a gap.
- **Not chat templating.** `POST /generate` takes a raw prompt string and produces a raw
  continuation — TinyLlama-1.1B-**base** behaviour, e.g. `"Paris.\n\n2. B."` from
  docs/13. TinyLlama-Chat's `<|system|>/<|user|>/<|assistant|>` framing is a named
  follow-up: it is a *string formatting* concern with its own correctness question (does
  our tokenizer round-trip the special tokens?) and does not belong in the step that
  brings up the transport.
- **Not INT8 on the GPU path.** Same as Steps 6–7: `Model::qweights` stays ignored by
  `GpuModel`. The CPU path's `--int8` is not exposed either.
- **Not auth, rate limiting, or OpenAI-compatible routes.** A local single-user server.
- **Not multi-GPU.** One card, selected by `CUDA_VISIBLE_DEVICES` as everywhere else in
  Phase 2.

## The operation

```
POST /generate {"prompt": "The capital of France is", "max_new_tokens": 8}

  ids   = tokenizer.encode(prompt)              # C++ Tokenizer, BOS prepended
  engine.generate(ids, max_new_tokens, on_token = push-to-queue)
      -> per token: decode(ids[:i+1]) minus what was already sent  ->  SSE event
  terminal event: {"done": true, "generated": n, "finish_reason": ...}
```

Nothing here is new math. The whole step is plumbing around a proven `generate()`, and its
risk lives entirely in three places: the GIL, object lifetimes across the language
boundary, and incremental detokenization. Each is called out explicitly below.

## Layer 1 — the native module (`engine/bindings/`)

### Build guard

A new top-level `option(ATLAS_BUILD_PYTHON "Build the pybind11 extension module" OFF)`,
mirroring the existing `ATLAS_USE_CUDA` guard exactly: when OFF nothing pybind11 is
referenced, so the Phase 1 CPU build needs no Python dev headers and stays 10/10. When ON,
`add_subdirectory(engine/bindings)`.

pybind11 is located from the active environment rather than vendored:

```cmake
execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_atlas_engine atlas_engine.cpp)
```

The module links `atlas_engine`, and additionally `atlas_cuda` when `ATLAS_USE_CUDA=ON`.
Output goes to `${CMAKE_BINARY_DIR}/python/`, so:

| Build tree | Module contains | `_atlas_engine.has_cuda` |
|---|---|---|
| `build/python/` | `Tokenizer`, `Model` | `False` |
| `build-cuda/python/` | `Tokenizer`, `Model`, `GpuModel` | `True` |

**That split is the device-selection mechanism.** There is no runtime CUDA probe in the
module and no `#ifdef` soup in Python: which `.so` is on the path decides what the server
can do, and `has_cuda` is how Python asks. It also means the bridge and the server are
developable and testable on any box with a C++ toolchain, not only on Suramar.

Module name is `_atlas_engine` (leading underscore) so it cannot be confused with the
CMake static-library target `atlas_engine`, and to signal that `server/bridge.py` is the
public face.

### Bound surface

Deliberately minimal — every binding is a maintenance contract, so only what a caller
needs today:

```python
_atlas_engine.has_cuda: bool
_atlas_engine.BOS_ID: int          # Tokenizer::kBosId == 1
_atlas_engine.EOS_ID: int          # Tokenizer::kEosId == 2

Tokenizer.load(vocab_path, merges_path) -> Tokenizer
Tokenizer.encode(text, add_bos=True) -> list[int]
Tokenizer.decode(ids) -> str
Tokenizer.vocab_size -> int

Model.load(bin_path, manifest_path) -> Model
Model.generate(prompt_ids, max_new_tokens, on_token=None) -> list[int]

GpuModel.create(model) -> GpuModel        # only when has_cuda
GpuModel.generate(prompt_ids, max_new_tokens, on_token=None) -> list[int]
```

`forward()` is **not** bound. No caller in Phase 2–5 needs raw logits, and binding it would
commit us to a `Tensor` ↔ NumPy conversion contract (ownership, strides, lifetime of a view
into a 4.4 GB mapping) for zero present benefit. If Phase 5's `run_inference` tool ever
wants logits, that is when the contract gets designed. `quantize_int8()` is likewise
unbound — out of scope per above.

### Three traps this binding must get right

**1. The GIL.** `generate()` is a multi-second, pure-C++ loop. Bound naively it holds the
GIL for its whole duration and the FastAPI event loop freezes — nothing else in the process
runs, including the code trying to *send* the tokens it produces. So:

- Bind `generate` with `py::call_guard<py::gil_scoped_release>()`.
- The `on_token` trampoline is a C++ lambda that re-acquires with `py::gil_scoped_acquire`
  before calling the Python callable, and converts its return to `bool` (a callback
  returning `None` must be read as "keep going", not "stop").
- The callback fires on the *calling* thread (docs/13: it is invoked inline, before the
  next forward begins), so there is no cross-thread state to reason about — the thread that
  released the GIL is the one re-acquiring it.
- Exceptions raised inside the Python callback must not unwind through CUDA code. Catch in
  the trampoline, record, return `false` to stop the loop cleanly, then re-raise on the
  C++ side after `generate()` returns.

**2. Lifetime across the boundary.** `GpuModel` holds a **non-owning** `const Model*`, and
that `Model` owns the mmap the device views were built from (`forward.h:69`, and the same
rule `WeightStore` imposes). In Python, `GpuModel.create(Model.load(...))` would drop the
last reference to the `Model` immediately and leave the `GpuModel` pointing at an unmapped
blob — a use-after-free that would show up as garbage logits or a segfault long after the
call. Fix: `py::keep_alive<0, 1>()` on `GpuModel.create` (keep argument 1 alive as long as
the return value lives). The test suite asserts this by constructing exactly that
drop-the-model pattern and generating afterwards.

**3. Move-only types.** `Model`, `GpuModel`, `WeightStore`, and `DeviceTensor` are all
move-only by design. Bind construction only through the existing static factories
(`def_static("load", ...)`, `def_static("create", ...)`), which return by value and let
pybind11 move into the `unique_ptr` holder. Never expose a copy constructor or a
default-constructed instance — a copied `GpuModel` would double-free the device blob.

## Layer 2 — `server/bridge.py`

The only Python file that knows `_atlas_engine` exists. Everything above it (`serve.py`,
and later `rag/`, `agent/`, `mcp/`) talks to `Engine`.

### Module discovery

The extension is a build artifact, not an installed package. `bridge.py` resolves it in
order: `$ATLAS_ENGINE_MODULE` (explicit path to the `.so`, wins) → `build-cuda/python/` →
`build/python/`, preferring CUDA when both exist. It reports which one it loaded at
startup. If none is found the error names `scripts/build_cuda.sh` and the
`ATLAS_BUILD_PYTHON=ON` flag rather than surfacing a bare `ModuleNotFoundError`.

### `Engine`

```python
class Engine:
    @classmethod
    def load(cls, cfg: Config) -> "Engine": ...   # tokenizer + Model + GpuModel, ONCE

    @property
    def device(self) -> str: ...                  # "cuda" | "cpu"

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...
    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[TokenEvent]: ...
```

- Loading is once per process, at server startup: the tokenizer fixtures, the mmap'd
  weights, and — on CUDA — the ~0.3 s / 4.4 GB H2D upload in `GpuModel::create`. Doing this
  per request would dominate every response.
- `device` is `"cuda"` when `has_cuda` and `cfg.device != "cpu"`, else `"cpu"`. Requesting
  `device="cuda"` against a CPU-only module is a startup error, not a silent downgrade.
- `generate()` is `stream()` drained — one implementation, no second decode path that can
  disagree with the streamed one.

### `stream()` — worker thread + queue

`Engine.generate` blocks in C++ for the whole completion, but the caller wants tokens *as
they arrive*. So `stream()` runs the C++ call on a worker thread and yields from a
`queue.Queue`:

```
worker thread:  engine.generate(ids, n, on_token=lambda id: q.put(...) or not cancelled)
                finally: q.put(SENTINEL)
generator:      while (item := q.get()) is not SENTINEL: yield item
                on GeneratorExit / consumer gone: set cancelled, drain, join
```

This is what Step 7's `on_token` early-stop contract was for: when the HTTP client
disconnects, the generator is closed, `cancelled` is set, and the very next `on_token`
returns `False`, so the loop stops *before* paying for another forward. On a cacheless
engine at ~0.054 s/token that matters; on a long completion it matters a lot.

Exceptions raised on the worker thread are put on the queue and re-raised in the
generator's thread, so a C++ abort path does not become a silent hang.

### Serialization

A module-level `threading.Lock` held for the duration of a generation. One `GpuModel`, one
4.4 GB device blob, per-call scratch allocations, and all launches on the default stream —
two concurrent generations would interleave kernels and scratch on the same stream with no
correctness argument behind them. Requests therefore **queue**. This is a deliberate
property of a single-model, no-batching server, stated here so it is not later mistaken for
a bug. Concurrency arrives with batching (follow-ups, below).

### The incremental-detokenization trap

The subtle correctness issue of this step. `Tokenizer::decode` is not a per-token map — per
`tokenizer.h` it (a) reassembles `<0xNN>` byte-fallback runs into raw bytes, and (b) strips
*the single leading space* introduced by normalization. So `decode([id])` on each token in
turn is wrong twice over: the first token loses its leading space, and a multi-byte UTF-8
character split across byte-fallback tokens decodes to mojibake or throws.

The rule: **keep the full generated-id list, decode the prefix each step, emit the delta.**

```python
text = tok.decode(ids[:i + 1])
delta, sent = text[len(sent):], text
```

The prefix decode is O(i) per token, negligible against a ~54 ms forward. A delta that
would end mid-UTF-8-sequence is held back and merged into the next one, so every SSE event
carries valid UTF-8.

## Layer 3 — `server/config.py` + `server/serve.py`

### `config.py`

A frozen dataclass, every field `ATLAS_*` env-overridable, defaults resolved relative to
the repo root:

| Field | Default | Note |
|---|---|---|
| `weights_dir` | `weights/tinyllama-1.1b-chat` | `model.f32.bin` + `model.manifest.txt` |
| `tokenizer_dir` | `reference/tokenizer` | `vocab.txt` + `merges.txt` |
| `module_path` | auto-discovered | `$ATLAS_ENGINE_MODULE` override |
| `device` | `auto` | `auto` \| `cuda` \| `cpu` |
| `max_new_tokens` | `64` | hard cap; requests above it are rejected |
| `host` / `port` | `127.0.0.1` / `8000` | local by default, not `0.0.0.0` |

No `temperature` (see scope boundary). `repo-structure.md` lists it as this file's concern
and it will land there — with sampling.

### `serve.py`

- **Startup.** A FastAPI `lifespan` handler builds the `Engine` once and stashes it on
  `app.state`. Startup cost (mmap + 4.4 GB upload) is paid before the first request, and a
  bad weights path fails at boot rather than on request one.

- **`POST /generate`** — pydantic body `{prompt: str, max_new_tokens: int = 32,
  stream: bool = true}`. `max_new_tokens` is bounded `1..cfg.max_new_tokens` by the model,
  so an over-cap request is a 422 from validation, not a 40-second surprise. Empty prompt
  is a 422.

  Streaming responses are `StreamingResponse(..., media_type="text/event-stream")`:

  ```
  data: {"index": 0, "id": 3681, "token": "Paris"}
  data: {"index": 1, "id": 29889, "token": "."}
  ...
  data: {"done": true, "generated": 8, "finish_reason": "length", "elapsed_s": 0.43}
  data: [DONE]
  ```

  `finish_reason` is `"eos"` | `"length"` | `"disconnect"`. Both the token id and its text
  delta are carried: the text is what a human wants, the id is what makes a response
  checkable against the C++ oracle without re-tokenizing.

  With `stream: false` the same generator is drained server-side and returned as
  `{"text": ..., "ids": [...], "generated": n, "finish_reason": ..., "elapsed_s": ...}`.

- **`GET /health`** — `{"status": "ok", "device": "cuda", "has_cuda": true,
  "model_loaded": true}`. Cheap, no generation; this is what Phase 3+ clients poll.

**Why SSE.** It is the convention for token streaming (every LLM API a Phase 4/5 client
might also speak uses it), it has a browser-native consumer in `EventSource`, `curl -N`
renders it readably for manual checking, and its framing gives a natural place for the
terminal metadata event that raw chunked text has nowhere to put.

## Validation

The bar for this step is **not numerical** — no new math is introduced, so nothing needs
measure-then-pinning. The bar is that the Python path returns *exactly* what the C++ path
returns, which is a discrete comparison against values docs/13 already recorded.

### `server/tests/test_bridge.py` — imports no FastAPI

The boundary that keeps the two layers separable. SKIP-green (not fail) when the extension
module or the gitignored weight blob is absent, matching `test_forward_gpu` /
`test_generate_gpu` discipline.

1. **Tokenizer round-trip** — `encode("The capital of France is")` equals the reference ids
   `[1, 450, 7483, 310, 3444, 338]`; `decode(encode(s))` round-trips.
2. **Oracle equality** — `generate(ids, 8)` returns exactly
   `[3681, 29889, 13, 13, 29906, 29889, 350, 29889]`, the sequence docs/13 recorded on the
   A6000. Exact list equality, no tolerance. First token `3681` ("Paris") is asserted
   separately so a forward-pass regression is distinguishable from a loop regression.
3. **Text** — the completion decodes to `"Paris.\n\n2. B."` (docs/13's recorded
   continuation; base-model behaviour, asserted here because the *bridge* must not alter
   it, while the assertion is understood to be pinned to this engine, not to TinyLlama).
4. **Streaming equals batch** — `"".join(e.token for e in engine.stream(p, 8))` equals
   `engine.generate(p, 8)`. This is the assertion that catches the incremental-decode trap.
5. **Early stop** — an `on_token` returning `False` on its 2nd call yields exactly 2 ids;
   closing the `stream()` generator after 2 events stops the worker (assert the thread
   joins promptly and fewer than 8 forwards ran).
6. **Budget** — `max_new_tokens=0` returns empty and runs no forward; `=4` returns 4.
7. **Lifetime** — build a `GpuModel` from a `Model` whose only Python reference is then
   dropped, force a `gc.collect()`, and generate: must still produce the oracle sequence.
   This is the `keep_alive` regression test.
8. **GIL release** — while a generation runs on one thread, a second thread must be able to
   make progress (e.g. increment a counter / acquire the interpreter). Fails loudly if the
   `call_guard` is dropped.

### `server/tests/test_serve.py` — FastAPI `TestClient`

1. `GET /health` returns 200 with the expected device and `model_loaded: true`.
2. SSE stream over the reference prompt: at least one token event, ids in order with
   contiguous `index`, concatenated `token` fields equal the docs/13 text, terminal event
   present with `finish_reason: "length"`, `[DONE]` last.
3. `stream: false` returns the identical `text` and `ids`.
4. Disconnecting mid-stream stops generation (worker joins, `finish_reason` recorded as
   `"disconnect"` in the server log).
5. Validation: empty prompt → 422; `max_new_tokens` above `cfg.max_new_tokens` → 422;
   missing `prompt` → 422.

### Expected result

Nothing to pin. Record on the first green run: which module path loaded, the reported
device, the streamed text, and **time-to-first-token vs. total** for an 8-token completion
— on the A6000 that should be roughly 0.054 s and 0.43 s respectively (docs/13). TTFT
being ≈ one token's time is the observable proof that the endpoint streams rather than
buffers, and the gap between it and total is the argument for the KV cache stated in the
units a server cares about.

## Build & test workflow

```
scripts/build_cuda.sh          # gains -DATLAS_BUILD_PYTHON=ON
scripts/run_server.sh          # new: uvicorn server.serve:app, honours CUDA_VISIBLE_DEVICES
pytest server/tests            # new: the two suites above
curl -N -X POST localhost:8000/generate -d '{"prompt":"The capital of France is"}'
```

- Same lab-box loop as Steps 2–7, with a card pinned via `CUDA_VISIBLE_DEVICES` (check
  `nvidia-smi` for the freer one).
- `scripts/test_cuda.sh` stays a **CTest** runner (7/7); the Python suites are a separate
  `pytest` invocation, because they need an interpreter with the deps installed and their
  failure modes are unrelated to the kernels'.
- `requirements.txt`: uncomment the Phase 2 block (`fastapi`, `uvicorn`, `pybind11`) and
  add `pytest` + `httpx` (the `TestClient` transport). Installed into the project env per
  the standing convention.
- The CPU build (`ATLAS_USE_CUDA=OFF`) must stay 10/10, and must additionally build the
  module with `ATLAS_BUILD_PYTHON=ON` — the CPU module is what makes this step reviewable
  off the lab box.
- `.gitignore` gains `*.so` (build artifacts already land under the ignored `build*/`, but
  the extension is the first thing a stray `pip`/manual build could drop elsewhere).

## Performance follow-ups (named, deferred)

Inherited from docs/12–13, plus this step's own:

- **KV cache + decode-mode attention** — still the headline item; it is what turns
  ~0.054 s/token growing with length into a flat, much smaller number. Held to Step 7's
  token-exact oracle, which the bridge tests now also enforce end-to-end.
- **Last-row-only `lm_head`** — decode needs `[1, vocab]`, not `[seq, vocab]`.
- **Persistent scratch / dropped per-launch syncs** — ~230 device syncs per forward, ×N
  forwards per completion, ×N completions per server lifetime.
- **Continuous batching** — the lock serializes requests today. Batching needs a
  scheduler, batched kernels, and per-sequence KV cache; it comes after the cache.
- **Sampling (temperature / top-p)** — a feature follow-up, and the moment `config.py`
  grows the knob `repo-structure.md` anticipates.
- **Chat templating** — makes completions assistant-shaped instead of base continuations.

## Files created / touched

| File | Change |
|------|--------|
| `engine/bindings/atlas_engine.cpp` | new — the pybind11 module |
| `engine/bindings/CMakeLists.txt` | new — `pybind11_add_module(_atlas_engine ...)` |
| `CMakeLists.txt` | `option(ATLAS_BUILD_PYTHON)` + guarded `add_subdirectory` |
| `server/__init__.py` | new — package marker |
| `server/config.py` | new — paths, device, caps, host/port |
| `server/bridge.py` | new — module discovery + `Engine` (`generate`/`stream`) |
| `server/serve.py` | new — FastAPI app, `POST /generate` (SSE), `GET /health` |
| `server/tests/test_bridge.py` | new — bridge vs the C++ oracle, no FastAPI import |
| `server/tests/test_serve.py` | new — endpoint + streaming behaviour |
| `scripts/build_cuda.sh` | add `-DATLAS_BUILD_PYTHON=ON` |
| `scripts/run_server.sh` | new — uvicorn runner |
| `requirements.txt` | uncomment Phase 2 deps; add `pytest`, `httpx` |
| `.gitignore` | `*.so` |
| `docs/14-bridge-serving.md` | this spec |
| `.claude/skills/atlas-cuda-serving/SKILL.md`, `repo-structure.md` | mark Phase 2 complete |

## Done when

- `_atlas_engine` builds in **both** trees: `build/python/` (CPU, `has_cuda == False`) and
  `build-cuda/python/` (CUDA, `has_cuda == True`).
- `pytest server/tests` is green on the lab box with the blob present, and SKIPs green
  without it.
- The bridge returns the docs/13 id sequence exactly, and the streamed text equals the
  batch text.
- `curl -N` against `/generate` visibly emits tokens one at a time, with TTFT ≈ one token's
  latency rather than the full completion time.
- Client disconnect stops generation without waiting out `max_new_tokens`.
- CTest stays **7/7** in `build-cuda/` and **10/10** in the CPU build.

## Design decisions

- **Bridge and server in one step, split by the test boundary.** Steps 1 and 7 both split
  failure domains into separate steps; here the split is internal instead. The bridge alone
  is not a shippable artifact (nothing consumes it), the two layers share exactly one
  contract (`Engine.stream()`), and a bridge test that never imports FastAPI already
  guarantees a red test names one layer. A separate step would buy isolation we get for
  free and cost a commit that closes nothing.
- **SSE over NDJSON or raw chunks.** It is what LLM clients expect, `EventSource` consumes
  it without a client library, `curl -N` renders it, and its event framing gives the
  terminal `finish_reason` metadata a home. Raw text would also throw away the token ids
  that make the response checkable against the C++ oracle.
- **The module ships in both build trees, and that is device selection.** No runtime CUDA
  probe, no conditional imports: `has_cuda` is a compile-time fact of the `.so` that got
  loaded. It keeps the bridge and the whole serving layer developable and reviewable on any
  box, which matters because Suramar is shared and both cards are usually busy.
- **`forward()` is not bound.** Binding it means designing a `Tensor` ↔ NumPy ownership
  contract over views into a 4.4 GB mapping. No caller through Phase 5 needs it today, and
  an unused binding is a contract we would be maintaining for nothing.
- **Serialized single request, deliberately.** One model, one default stream, per-call
  scratch. Concurrency without batched kernels would be interleaving, not parallelism —
  slower and unvalidated. Stated as a property so it is not later read as a defect.
- **Streaming built on a worker thread rather than an async C++ callback.** The engine is
  synchronous and blocking by design; a thread plus a queue is the smallest correct adapter
  to an async server, and it keeps `GpuModel::generate()` unchanged. It also makes
  cancellation natural — the docs/13 `on_token`-returns-`False` path is exactly a
  disconnect.
- **Prefix-decode-and-delta rather than per-token decode.** Forced by
  `Tokenizer::decode`'s contract (leading-space strip + `<0xNN>` byte-run reassembly).
  Per-token decoding would corrupt the first token's spacing and break any multi-byte
  character — quietly, and only for some prompts. The cost is O(n²) character work against
  an O(n²) *forward pass*, i.e. free.
- **No temperature field until sampling exists.** A config knob that is read and ignored is
  a lie in the API surface; every caller that sets it would carry a false belief.
- **Raw prompts, no chat template.** Templating is a string concern with its own
  correctness question (special-token round-tripping through our tokenizer) and would blur
  the transport bring-up. Named as a follow-up, with `"Paris.\n\n2. B."` documented as the
  expected *base-model* output so nobody reads it as a bug.
