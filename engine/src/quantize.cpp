#include "../include/quantize.h"

#include <cassert>
#include <cmath>

// Per-row symmetric INT8 weight quantization and the dequantize-in-the-inner-loop
// linear it feeds — the W8A32 scheme from docs/05-quantization.md. Storage format
// only: activations stay FP32, and there is no INT8 compute path (SIMD int8 dot
// products are a Phase 2 concern). The accuracy cost is measured end-to-end against
// reference/logits.npy in test_quantize.

namespace atlas {

QTensor quantize_rows(const Tensor& w) {
    assert(w.shape.size() == 2 && "quantize_rows: weight must be 2-D");
    assert(w.strides[1] == 1 && w.strides[0] == w.shape[1] &&
           "quantize_rows: weight must be contiguous row-major");
    QTensor q;
    q.rows = w.shape[0];
    q.cols = w.shape[1];
    q.data.resize(static_cast<size_t>(q.rows * q.cols));
    q.scales.resize(static_cast<size_t>(q.rows));
    for (int64_t r = 0; r < q.rows; ++r) {
        const float* row = w.data + r * q.cols;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < q.cols; ++i) {
            const float a = std::fabs(row[i]);
            if (a > max_abs) max_abs = a;
        }
        // All-zero row: scale 1.0 quantizes to all zeros (no div-by-zero, no NaN).
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv = 1.0f / scale;
        int8_t* out = q.data.data() + r * q.cols;
        for (int64_t i = 0; i < q.cols; ++i) {
            // By construction |row[i] * inv| <= 127, but inv = 1/scale is itself
            // rounded, so the product can land a hair outside — clamp.
            long v = std::lroundf(row[i] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            out[i] = static_cast<int8_t>(v);
        }
        q.scales[r] = scale;
    }
    return q;
}

Tensor dequantize(const QTensor& q) {
    Tensor t = Tensor::zeros({q.rows, q.cols});
    for (int64_t r = 0; r < q.rows; ++r) {
        const float scale = q.scales[static_cast<size_t>(r)];
        const int8_t* src = q.data.data() + r * q.cols;
        float* dst = t.data + r * q.cols;
        for (int64_t i = 0; i < q.cols; ++i) {
            dst[i] = static_cast<float>(src[i]) * scale;
        }
    }
    return t;
}

// Mirrors model.cpp's FP32 `linear` (y = x @ Wᵀ, both row walks sequential), with the
// per-row scale applied once per dot product — it factors out of the sum, which is the
// whole point of quantizing per row.
void linear_q8(const Tensor& x, const QTensor& w, Tensor& out) {
    assert(x.shape.size() == 2 && out.shape.size() == 2);
    const int64_t m = x.shape[0];
    const int64_t in = x.shape[1];
    assert(w.cols == in && "linear_q8: weight in_features mismatch");
    assert(out.shape[0] == m && out.shape[1] == w.rows && "linear_q8: out shape mismatch");
    for (int64_t i = 0; i < m; ++i) {
        const float* xrow = x.data + i * in;
        for (int64_t o = 0; o < w.rows; ++o) {
            const int8_t* qrow = w.data.data() + o * in;
            float acc = 0.0f;
            for (int64_t k = 0; k < in; ++k) {
                acc += xrow[k] * static_cast<float>(qrow[k]);
            }
            out.data[i * w.rows + o] = acc * w.scales[static_cast<size_t>(o)];
        }
    }
}

}  // namespace atlas
