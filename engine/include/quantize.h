#pragma once

#include <cstdint>
#include <vector>

#include "tensor.h"

namespace atlas {

// Post-training INT8 quantization of the linear-layer weights — per-row symmetric,
// weights-only (W8A32). For each row r of a weight matrix W [out, in]:
//
//   scale_r = max(|W[r, :]|) / 127     (scale_r = 1.0 if the row is all zeros)
//   Q[r, i] = round(W[r, i] / scale_r) clamped to [-127, 127]
//
// Per-row because `linear` computes y = x @ Wᵀ — each output element touches exactly
// one row, so the scale factors out of the dot product: y[j] = scale_j * Σ x[k]·Q[j,k].
// Symmetric (zero-point ≡ 0) because weight distributions are centered near zero;
// zero-points earn their keep on activations, which stay FP32 this step. Clamped to
// ±127, not -128, so quantize(-w) == -quantize(w) exactly. See docs/05-quantization.md.
//
// This lives here, not in tensor.h — the Tensor foundation stays FP32-only (the Step 1
// "no dtype templating" decision stands). INT8 is a weight *storage* format with its
// own three functions, not a new dtype threaded through every op.

// An INT8-quantized weight matrix: row-major int8 payload + one FP32 scale per row.
// Owns its buffers — it is *produced* from the mmap'd FP32 views, not a view itself
// (the only place quantized weights exist; there is no INT8 blob on disk). Move-only,
// like Tensor: a ~GB payload must never be silently duplicated.
struct QTensor {
    std::vector<int8_t> data;   // [rows * cols], row-major
    std::vector<float> scales;  // [rows]; dequant of element (r, i) is data[r*cols + i] * scales[r]
    int64_t rows = 0;
    int64_t cols = 0;

    QTensor() = default;
    QTensor(const QTensor&) = delete;
    QTensor& operator=(const QTensor&) = delete;
    QTensor(QTensor&&) noexcept = default;
    QTensor& operator=(QTensor&&) noexcept = default;
};

// FP32 [out, in] -> per-row symmetric INT8 (the scheme above). w must be 2-D and
// contiguous (every weight is — they are manifest views or owned buffers).
QTensor quantize_rows(const Tensor& w);

// QTensor -> owning FP32 Tensor [rows, cols] — the round-trip aid for tests and
// debugging. The forward path never materializes this; linear_q8 dequantizes in
// its inner loop instead.
Tensor dequantize(const QTensor& q);

// y = x @ dequant(W)ᵀ, dequantizing in the inner loop:
//   out[s, j] = scales[j] * Σ_k x[s, k] * (float)data[j*cols + k]
// Same shape contract as model.cpp's FP32 `linear`: x [seq, in], w [out, in],
// out [seq, out] pre-sized by the caller — tensor.h out-param style.
void linear_q8(const Tensor& x, const QTensor& w, Tensor& out);

}  // namespace atlas
