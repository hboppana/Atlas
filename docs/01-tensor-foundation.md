# Phase 1 · Step 1 — Tensor Foundation + C++ Build

> Status: **planned** (not yet started)
> Predecessor: Step 0 — ground truth (`scripts/download_weights.py`, golden oracles in `reference/`) — **done**
> Successors: tokenizer → model + weight loading → forward-pass validation → quantize

## Goal

Stand up the `engine/` C++ project with a `Tensor` class solid enough to build the
entire TinyLlama forward pass on top of. Nothing model-specific in this step — just
the data structure, memory model, and the handful of core ops everything else needs.

Build everything CPU-only on Windows (MSVC), FP32, validated against the HuggingFace
reference captured in Step 0.

## Definition of done

- `cmake --build` succeeds on MSVC (C++17), CUDA path guarded off.
- `test_tensor.exe` passes: shape/stride/indexing, reshape, and `matmul` checked
  against hand-computed values.

## Files created in this step

| File | Contents |
|------|----------|
| `CMakeLists.txt` (root) | Top-level project, C++17, CPU-only path (CUDA guarded off for now), adds `engine/` |
| `engine/CMakeLists.txt` | `atlas_engine` static lib + test targets |
| `engine/include/tensor.h` | `Tensor` class declaration + free-function ops |
| `engine/src/tensor.cpp` | Memory management, reshape, core ops |
| `engine/tests/test_tensor.cpp` | Unit tests with a tiny zero-dependency assert harness |

## Design decisions

These are the deliberate choices for this layer. Defaults chosen for correctness-first,
phase-by-phase development; revisit only with reason.

- **FP32 only.** No templating on dtype in this step. INT8 gets its own path in the
  `quantize` step once the FP32 engine is proven correct against `reference/logits.npy`.
- **Owns *or* views.** `Tensor` holds a `float*` + shape/strides plus a flag for whether
  it owns the buffer. Owned = allocated activations; viewing = a non-owning window over
  mmap'd weights (added later). This is what lets `tensor.cpp` do "mmap weight loading"
  without copying ~2 GB.
- **Row-major, contiguous, explicit strides.** Reshape/slice become free (no copy).
- **Core ops as free functions in `tensor.h`:** `matmul`, elementwise `add` / `mul`.
  Model-specific math (RMSNorm, RoPE, attention, SwiGLU) lives in `model.cpp` later and
  is *built from* these primitives. Matches the structure's split: tensor = memory/ops,
  model = the network.
- **Zero-dependency test harness.** A small `CHECK(...)` macro — no GoogleTest/Catch2.
  Less build friction on Windows; fits the from-scratch ethos.

## Tensor API sketch (indicative, not final)

```cpp
// engine/include/tensor.h
struct Tensor {
    float* data = nullptr;          // pointer to first element (owned or viewed)
    std::vector<int64_t> shape;     // logical dimensions, row-major
    std::vector<int64_t> strides;   // element strides per dim
    bool owns = false;              // true => free data in destructor

    static Tensor zeros(std::vector<int64_t> shape);   // owned, allocated
    static Tensor view(float* ptr, std::vector<int64_t> shape); // non-owning

    int64_t numel() const;
    Tensor reshape(std::vector<int64_t> shape) const;  // view, no copy
    float& at(std::initializer_list<int64_t> idx);     // bounds-checked in debug
    // ... move-only semantics; no accidental deep copies
};

// Core ops (free functions). Out-param style to make allocations explicit.
void matmul(const Tensor& a, const Tensor& b, Tensor& out); // [m,k] x [k,n] -> [m,n]
void add(const Tensor& a, const Tensor& b, Tensor& out);    // elementwise (+ broadcast row)
void mul(const Tensor& a, const Tensor& b, Tensor& out);    // elementwise
```

## Explicitly deferred (not this step)

- `convert_weights.py` + real mmap weight loading → arrives with the **model loading**
  step; `Tensor` can be fully tested without real weights.
- A `.npy` reader in C++ (to compare against `reference/logits.npy` / `token_ids.npy`)
  → arrives with **forward-pass validation**.
- INT8 / quantization → the **quantize** step.
- Any CUDA → Phase 2.

## Reference oracles produced in Step 0

Consumed by later steps, recorded here for traceability:

- `reference/config.json` — pinned TinyLlama-1.1B-Chat hyperparameters
  (hidden 2048, layers 22, heads 32, kv_heads 4 (GQA 8:1), head_dim 64,
  intermediate 5632, vocab 32000, rms_eps 1e-5, rope_theta 10000).
- `reference/token_ids.npy` — oracle for `test_tokenizer.cpp` (note: BOS id 1 prepended).
- `reference/logits.npy` — oracle for `test_forward.cpp` (CPU FP32, prompt
  "The capital of France is").
