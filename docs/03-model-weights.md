# Phase 1 · Step 3 — Model + Weight Loading + Forward Pass

> Status: **done** — forward pass runs end-to-end; `test_forward` green
> (top-1 of the last position: id 3681 `▁Paris`, logit 13.3885; ~11 s single-threaded -O3)
> Predecessor: Step 2 — BPE tokenizer — **done** ([02-tokenizer.md](02-tokenizer.md))
> Successor: Step 4 — forward-pass validation (C++ `.npy` reader, compare to `reference/logits.npy`)

## Goal

Implement the full TinyLlama-1.1B forward pass in C++ — embedding → 22 transformer
layers → final norm → LM head — and the weight loading that feeds it. Given the prompt
token ids (`[1, 450, 7483, 310, 3444, 338]` from Step 2) the engine produces a
`[seq, 32000]` FP32 logits tensor, and greedy-decoding the last position predicts
`▁Paris`.

This is the heart of the engine: the first step with real model math (RoPE, RMSNorm, GQA
attention, SwiGLU) and the first that touches the 2 GB of weights. Still CPU-only, FP32.

**Validation split:** this step builds the forward pass and proves it *runs* end-to-end
with a sane top-1 prediction. The rigorous, tolerance-checked comparison against
`reference/logits.npy` is **Step 4** (it needs a C++ `.npy` reader, deferred to there).

## Definition of done

- `cmake --build` still green. `model.cpp` joins `atlas_engine`; a `src/main.cpp` CLI loads
  the model and runs inference.
- `convert_weights.py` produces the FP32 weight blob + manifest from `model.safetensors`.
- A `test_forward` (or `main`) loads the weights, runs the prompt, and:
  - logits shape is `[6, 32000]`;
  - `argmax` of the last row decodes (via the Step 2 tokenizer) to `▁Paris` —
    the same top-1 the Step 0 reference reported.
- No tolerance check yet — that is Step 4's contract.

## The weights on disk (read off `weights/tinyllama-1.1b-chat/model.safetensors`)

One 2.2 GB safetensors file, **201 tensors, all `BF16`**. The engine is FP32-only in
Phase 1, so conversion must **upcast BF16 → FP32** (lossless — BF16 is the top 16 bits of
an FP32, so this is a zero-extend of the mantissa; it matches what HF did when it built
`logits.npy` with `dtype=torch.float32`).

Tensor inventory (PyTorch `nn.Linear` stores weights as `[out_features, in_features]`, no
biases anywhere, no learned RoPE params):

| Tensor | Shape | Role |
|--------|-------|------|
| `model.embed_tokens.weight` | `[32000, 2048]` | token embedding table |
| `model.layers.{0..21}.input_layernorm.weight` | `[2048]` | RMSNorm before attention |
| `model.layers.{i}.self_attn.q_proj.weight` | `[2048, 2048]` | Q projection (32 heads × 64) |
| `model.layers.{i}.self_attn.k_proj.weight` | `[256, 2048]` | K projection (**4** kv heads × 64) |
| `model.layers.{i}.self_attn.v_proj.weight` | `[256, 2048]` | V projection (4 kv heads × 64) |
| `model.layers.{i}.self_attn.o_proj.weight` | `[2048, 2048]` | attention output projection |
| `model.layers.{i}.post_attention_layernorm.weight` | `[2048]` | RMSNorm before MLP |
| `model.layers.{i}.mlp.gate_proj.weight` | `[5632, 2048]` | SwiGLU gate |
| `model.layers.{i}.mlp.up_proj.weight` | `[5632, 2048]` | SwiGLU up |
| `model.layers.{i}.mlp.down_proj.weight` | `[2048, 5632]` | SwiGLU down |
| `model.norm.weight` | `[2048]` | final RMSNorm |
| `lm_head.weight` | `[32000, 2048]` | output projection to vocab (**not** tied — present as its own tensor) |

`22 layers × 9 + embed + norm + lm_head = 201`. ✓ The K/V projections are `256 = 4×64`
(not `2048`) — that **8:1 GQA** asymmetry is the thing most likely to trip up the
attention reshape.

## Where the weights come from (`scripts/convert_weights.py`, new)

Same division of labor as the tokenizer: **Python parses the complex format once; C++ reads
a flat blob.** The engine never parses safetensors JSON or decodes BF16.

`convert_weights.py` reads `model.safetensors`, upcasts every tensor to FP32, and writes:

- `weights/tinyllama-1.1b-chat/model.f32.bin` — all tensor bytes concatenated, FP32,
  little-endian, in a fixed order. ~4.4 GB. **Gitignored** (unlike the 1 MB tokenizer
  fixtures — too big to commit; regenerated locally from the downloaded weights).
- `weights/tinyllama-1.1b-chat/model.manifest.txt` — plain text, one tensor per line:
  `name byte_offset ndim d0 d1 ...`. Line-oriented and diffable, mirroring the tokenizer
  export. The engine reads this to know where each tensor lives in the blob.

Keep the PyTorch `[out, in]` layout as-is (no transpose at convert time); the forward pass
computes `y = x @ Wᵀ` directly (see `linear` below). This keeps the converter trivial and
matches the reference's own computation.

**Loading in C++ (`model.cpp`):** `mmap` the `.bin` once and build a **non-owning
`Tensor::view`** per manifest entry — this is the first real use of the `owns=false` path
the `Tensor` foundation was designed for (Step 1), so the 4.4 GB is never copied. On
Windows/MinGW use the Win32 file-mapping API (`CreateFileMappingA` / `MapViewOfFile`); wrap
it so the mapping outlives every view (views point into it). A `WeightStore` struct owns
the mapping handle + base pointer and hands out views by name.

## The forward pass (`model.cpp`) — what it must reproduce

Modern Llama, **not GPT-2**. Hyperparameters pinned in `reference/config.json`: `hidden
2048`, `layers 22`, `heads 32`, `kv_heads 4`, `head_dim 64`, `intermediate 5632`, `vocab
32000`, `rms_norm_eps 1e-5`, `rope_theta 10000`.

1. **Embed**: `h = embed_tokens[token_ids]` → `[seq, 2048]`.
2. **For each of 22 layers** (pre-norm residual blocks):
   - `r = h`
   - **RMSNorm** with `input_layernorm.weight`:
     `x_i = x_i / sqrt(mean(x²) + 1e-5) * w_i` (per row; **no** mean-subtraction — this is
     RMSNorm, not LayerNorm). Compute the variance in FP32.
   - **Attention**:
     - `q = x @ q_projᵀ` → `[seq, 2048]` → view as `[seq, 32, 64]`
     - `k = x @ k_projᵀ` → `[seq, 256]`  → view as `[seq, 4, 64]`
     - `v = x @ v_projᵀ` → `[seq, 256]`  → view as `[seq, 4, 64]`
     - **RoPE** on `q` and `k` (details below).
     - **GQA**: query head `hq` uses kv head `hq / 8` (8 query heads per kv head; HF's
       `repeat_kv` expands each kv head 8× in contiguous blocks).
     - **Scaled dot-product, causal**: `scores = q·kᵀ / sqrt(64)`; mask `j > i` to `-inf`;
       `softmax` over keys (in FP32); `ctx = softmax @ v` → `[seq, 32, 64]` → `[seq, 2048]`.
     - `attn = ctx @ o_projᵀ` → `[seq, 2048]`.
   - `h = r + attn` (residual).
   - `r = h`; **RMSNorm** with `post_attention_layernorm.weight`.
   - **SwiGLU MLP**:
     - `g = x @ gate_projᵀ` → `[seq, 5632]`; `u = x @ up_projᵀ` → `[seq, 5632]`
     - `m = SiLU(g) * u`, where `SiLU(z) = z * sigmoid(z)` (a.k.a. swish)
     - `d = m @ down_projᵀ` → `[seq, 2048]`
   - `h = r + d` (residual).
3. **Final RMSNorm** with `model.norm.weight`.
4. **LM head**: `logits = h @ lm_headᵀ` → `[seq, 32000]`.

### RoPE — the detail most likely to be wrong

HF Llama uses the **half-split ("NeoX") rotation**, not interleaved adjacent pairs:

- `inv_freq[i] = 1 / 10000^(2i/64)` for `i in 0..31` (32 frequencies).
- For position `m`: `angle[i] = m * inv_freq[i]`. Build `cos`/`sin` of length 64 by
  **concatenating the 32 angles with themselves**: `emb = cat(angle, angle)`.
- `rotate_half(x) = cat(-x[32:64], x[0:32])`.
- `x_rot = x * cos + rotate_half(x) * sin`, applied per head over the 64 dims.

Get the split-in-half (not even/odd interleave) right or the logits diverge silently.

## Files created in this step

| File | Contents |
|------|----------|
| `scripts/convert_weights.py` (new) | safetensors (BF16) → FP32 `model.f32.bin` + `model.manifest.txt`. Asserts the 201 expected tensors/shapes against `reference/config.json`. |
| `engine/include/model.h` (new) | `Config`, `WeightStore` (mmap + views), `Model` with `forward(ids) -> logits`. |
| `engine/src/model.cpp` (new) | weight mmap/manifest loading; RMSNorm, RoPE, GQA attention, SwiGLU built from `tensor.h` primitives + a `linear` (matmul-with-transposed-B) helper. |
| `engine/src/main.cpp` (new) | CLI: load tokenizer + model, encode a prompt, run forward, greedy-decode, print tokens. |
| `engine/tests/test_forward.cpp` (new) | smoke test: shape `[6, 32000]`, `argmax` last row decodes to `▁Paris`. (Tolerance check vs `logits.npy` is Step 4.) |
| `engine/CMakeLists.txt` (edit) | add `src/model.cpp` to `atlas_engine`; add `main` + `test_forward` targets. |

`tensor.*` and `tokenizer.*` are untouched (reused as-is). The model math lives in
`model.cpp`, built **from** the `tensor.h` primitives — keeping the tensor=memory/ops vs
model=network split from Step 1.

## Design decisions

- **Correctness-first, FP32, single-threaded.** Match `reference/logits.npy`; no SIMD/
  threading/KV-cache yet. A naive `[seq,2048]` forward over a 6-token prompt is instant.
- **Convert once in Python, mmap a flat FP32 blob in C++.** No safetensors/BF16/JSON parsing
  in the engine — consistent with the tokenizer's Python-exports-flat-files boundary.
- **Keep `[out, in]` weight layout; compute `y = x @ Wᵀ`** via a `linear` helper, rather than
  transposing at convert time. Simpler converter, matches the reference.
- **Non-owning `Tensor` views over the mmap** — the payoff of the Step 1 "owns *or* views"
  design; the 4.4 GB is mapped, never copied.
- **No KV cache / no incremental decode this step.** Full-sequence forward only; the cache
  is a generation optimization that arrives once the math is proven correct.
- **Prefill only, raw prompt** (no chat template) — same prompt as Steps 0/2.

## Explicitly deferred (not this step)

- C++ `.npy` reader + tolerance-checked logits comparison → **Step 4**.
- KV cache, multi-token generation loop, sampling (top-p/temperature) → after validation.
- INT8 quantization → **Step 5**.
- SIMD / multithreading / any CUDA → Phase 2.
- Chat template / special-token-aware prompting.

## Reference oracle

- `reference/config.json` — the hyperparameters above (the converter validates against it).
- `reference/token_ids.npy` — `[1, 450, 7483, 310, 3444, 338]`, the forward input.
- `reference/logits.npy` — `[6, 32000]` FP32, prompt "The capital of France is". The Step 0
  run reported a `▁Paris`-topped top-5; reproducing that top-1 is this step's bar, and the
  full per-logit match is Step 4's.

## Implementation notes (what the build actually hit)

- **MinGW's 32-bit `stat()` overflows on the 4.4 GB blob** and reports it missing —
  `test_forward`'s skip-check uses `std::ifstream(path).good()` instead. (The mmap path
  itself uses `GetFileSizeEx`, which is 64-bit.)
- **`CMAKE_BUILD_TYPE` now defaults to Release** in the top-level CMakeLists: the naive
  single-threaded forward is ~6.6 GMAC and needs `-O3` to finish in seconds (an
  unoptimized Debug build takes minutes). Pass `-DCMAKE_BUILD_TYPE=Debug` explicitly when
  the Tensor asserts (compiled out under `NDEBUG`) are wanted.
- The result on the reference prompt: top-1 id **3681** (`▁Paris`), logit **13.3885**.
  `atlas.exe` (the new CLI) prints the full pipeline and the completion
  `"The capital of France is Paris"`.
