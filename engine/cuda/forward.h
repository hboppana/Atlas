#pragma once

#include <vector>

#include "../include/model.h"
#include "device_tensor.h"

namespace atlas {

// The full TinyLlama forward pass on the GPU (Phase 2 · Step 6, docs/12-cuda-forward.md):
// the composition of the five proven kernels (matmul, rmsnorm, embed/add/swiglu/rope,
// attention) into exactly the op order of Model::forward() (model.cpp). Token ids in,
// FP32 [seq, vocab] logits out, with every intermediate living on the device — one H2D
// upload of the ids at the top, one D2H download of the logits at the bottom, nothing
// else crossing the bus per call.
//
// A struct rather than a free function because the 4.4 GB weight blob is uploaded ONCE
// at create() (~0.3 s of PCIe traffic); a free function would either re-upload per call
// or hide a global. It lives here in engine/cuda/ rather than as a Model member so that
// engine/include/model.h stays CUDA-free — a member declared in the CPU tree but defined
// only inside atlas_cuda is a link trap for CPU-only callers.
//
// Scope, same as Model::forward(): full-prompt prefill, logits for every position, no KV
// cache and no incremental decode (Step 7). FP32 only — Model::qweights is IGNORED on
// this path, so any logit diff is attributable to composition, never to quantization.
struct GpuModel {
    // One cudaMalloc of WeightStore::size_bytes() + one H2D memcpy of the whole mapped
    // blob. Each weight is then a zero-copy DeviceTensor::view at the same offset as its
    // host view, so WeightStore stays the single source of truth for shapes and offsets.
    // The Model must outlive the GpuModel — the same views-into-a-mapping lifetime rule
    // WeightStore itself imposes.
    static GpuModel create(const Model& model);

    GpuModel() = default;
    ~GpuModel();

    // Move-only: the device blob pointer is unique (DeviceTensor / WeightStore discipline).
    GpuModel(const GpuModel&) = delete;
    GpuModel& operator=(const GpuModel&) = delete;
    GpuModel(GpuModel&& other) noexcept;
    GpuModel& operator=(GpuModel&& other) noexcept;

    // Full-sequence prefill: token ids -> FP32 logits [seq, vocab] on the host. Same
    // contract, op order, and launch order as Model::forward(); all launches go on the
    // default stream in program order, so cross-kernel ordering needs no explicit events.
    Tensor forward(const std::vector<int>& token_ids) const;

private:
    // Non-owning; supplies config, shapes and offsets on every call.
    const Model* model_ = nullptr;
    float* blob_ = nullptr;      // device copy of the whole mapped weight blob (owned)
    size_t blob_bytes_ = 0;

    // The device twin of model_->weights.get(name): same shape, same offset from base.
    DeviceTensor weight(const std::string& name) const;
};

}  // namespace atlas
