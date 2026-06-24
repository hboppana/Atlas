# Results — Phase 2 · Step 2 tiled matmul validation

Recorded run of `test_matmul` (the tiled CUDA matmul `y = x @ Wᵀ`) against the CPU
`atlas::linear()` oracle. Spec: [../08-cuda-matmul.md](../08-cuda-matmul.md).

## Environment

| | |
|---|---|
| Box | Suramar (lab) |
| GPU | NVIDIA RTX A6000 (Ampere, sm_86), card 1 (`CUDA_VISIBLE_DEVICES=1`) |
| Driver | 560.35.05 |
| CUDA toolkit | 12.6 (V12.6.85) |
| Build | `scripts/build_cuda.sh` → `build-cuda/`, Release |
| Code | branch `main`, on top of `103ab26` (this run's tree adds `matmul.{h,cu}`, `test_matmul.cu`, `test_harness.h`) |
| Date | 2026-06-24 |

## CTest

`CUDA_VISIBLE_DEVICES=1 scripts/test_cuda.sh` → `-R 'test_device|test_matmul'`:

```
1/2 Test #11: test_device ......................   Passed    0.55 sec
2/2 Test #12: test_matmul ......................   Passed    1.01 sec
100% tests passed, 0 tests failed out of 2
```

## Per-shape diff vs `linear()` (`test_matmul`)

Inputs are deterministic `fill_prng` values ~[-0.5, 0.5). Diff is element-wise GPU↔CPU; the
nonzero diff is FP32 summation-order reordering (tiled partial sums vs the oracle's sequential
dot), **not** a precision change — the kernel accumulates in FP32 like `linear()`.

| case | m | in | out | max-abs | mean-abs |
|------|----|----|-----|---------|----------|
| aligned | 32 | 32 | 32 | 2.38e-07 | 3.04e-08 |
| non-aligned | 3 | 50 | 17 | 1.79e-07 | 3.94e-08 |
| decode-m1 | 1 | 2048 | 2048 | 5.72e-06 | 7.70e-07 |
| q_proj | 8 | 2048 | 2048 | 4.77e-06 | 7.78e-07 |
| kv_proj | 8 | 2048 | 256 | 4.77e-06 | 7.85e-07 |
| mlp_gate | 8 | 2048 | 5632 | 5.72e-06 | 7.78e-07 |
| **mlp_down** | 8 | **5632** | 2048 | **1.14e-05** | 1.68e-06 |
| lm_head | 1 | 2048 | 32000 | 5.72e-06 | 7.64e-07 |

## Pinned tolerance

The diff grows with the contraction length `in`, as the design predicted. The worst case is
**mlp_down** (in=5632, the widest contraction in TinyLlama) at **max-abs 1.14e-05** — so that
shape is effectively the ceiling. `test_matmul.cu` pins:

```cpp
constexpr double kMaxAbsTol = 1e-4;  // ~9x over measured worst (1.14e-5); a real bug is O(1)
```

~9× headroom over the measured worst, yet ~four orders of magnitude under a real-bug signal (a
wrong sum is off by O(output magnitude) ~ O(1), not 1e-4). Per the standing
measure-then-pin discipline, re-measure and revisit this number if the kernel changes (e.g. the
named perf follow-up: register-blocking, float4 loads, TILE=32, TF32).

## Portable path unchanged

CPU build with `ATLAS_USE_CUDA=OFF` still green: `100% tests passed, 10/10` — the Step 2
changes are CUDA-only.
