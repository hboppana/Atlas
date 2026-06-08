#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace atlas {

// The TinyLlama-1.1B forward pass in C++ — embedding -> 22 transformer layers ->
// final RMSNorm -> LM head — plus the weight loading that feeds it. CPU-only, FP32,
// single-threaded: correctness first, matched against reference/logits.npy. The model
// math (RMSNorm, RoPE, GQA attention, SwiGLU) is built from the tensor.h primitives in
// model.cpp; this header only declares the three types that compose it. See
// docs/03-model-weights.md.
//
// Division of labor mirrors the tokenizer: scripts/convert_weights.py parses the BF16
// safetensors ONCE and writes a flat FP32 blob (model.f32.bin) + a line-oriented manifest
// (model.manifest.txt). The engine never parses safetensors JSON or decodes BF16 — it
// mmaps the blob and reads each tensor's location from the manifest.

// Pinned TinyLlama-1.1B-Chat hyperparameters (reference/config.json). The engine doesn't
// parse JSON — these defaults ARE the contract, and convert_weights.py validates the
// downloaded weights against the same config. Modern Llama, not GPT-2: no biases, RMSNorm
// (not LayerNorm), RoPE, GQA, SwiGLU.
struct Config {
    int hidden_size = 2048;
    int intermediate_size = 5632;   // SwiGLU inner width
    int num_layers = 22;
    int num_heads = 32;             // query heads
    int num_kv_heads = 4;           // GQA: 8 query heads share each kv head
    int head_dim = 64;
    int vocab_size = 32000;
    int max_position = 2048;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;

    // Derived. kv_dim is 256 = 4*64 (not 2048) — the 8:1 GQA asymmetry. q_per_kv is how
    // many query heads each kv head serves (HF repeat_kv expands kv 8x in contiguous blocks).
    int kv_dim() const { return num_kv_heads * head_dim; }
    int q_per_kv() const { return num_heads / num_kv_heads; }
};

// Owns the mmap of model.f32.bin and hands out non-owning Tensor views into it, by name.
// This is the first real use of tensor.h's owns=false path: the 4.4 GB blob is mapped
// once and never copied — every weight is a Tensor::view window onto the mapping. The
// mapping outlives every view it produces (views point into it), so a Model that holds
// views must keep its WeightStore alive. Move-only: the OS mapping handle is never
// silently duplicated.
struct WeightStore {
    // mmap the blob and build one non-owning view per manifest line. The manifest format
    // is one tensor per line: "name byte_offset ndim d0 d1 ...". Exits on a missing or
    // malformed file (the blob + manifest are a generated contract, not user input).
    static WeightStore load(const std::string& bin_path, const std::string& manifest_path);

    WeightStore() = default;
    ~WeightStore();

    // Move-only (the mapping handle is unique).
    WeightStore(const WeightStore&) = delete;
    WeightStore& operator=(const WeightStore&) = delete;
    WeightStore(WeightStore&& other) noexcept;
    WeightStore& operator=(WeightStore&& other) noexcept;

    // Non-owning view of the named weight (e.g. "model.layers.0.self_attn.q_proj.weight").
    // Aborts if the name is absent — names come from the model code, not user input.
    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const;

private:
    // Platform-opaque mmap state (Win32 CreateFileMapping/MapViewOfFile under the hood),
    // kept as void* so the header stays free of <windows.h>. Details live in model.cpp.
    void* file_handle_ = nullptr;     // OS file handle
    void* mapping_handle_ = nullptr;  // OS file-mapping object
    float* base_ = nullptr;           // mapped base pointer (start of the FP32 blob)
    size_t size_bytes_ = 0;           // mapped length

    // name -> non-owning Tensor::view into base_. Owns the views (move-only Tensors live
    // here); get() hands out references that stay valid until this WeightStore is destroyed.
    std::unordered_map<std::string, Tensor> tensors_;
};

// The assembled TinyLlama model: a Config + the mmap'd weights, with the forward pass.
// Owns its WeightStore, so the weight views stay valid for the model's lifetime. Move-only.
struct Model {
    Config config;
    WeightStore weights;

    // Load the converted weights (model.f32.bin + model.manifest.txt) and pair them with
    // the pinned config. The blob is mmap'd, not read into memory.
    static Model load(const std::string& bin_path,
                      const std::string& manifest_path,
                      const Config& cfg = Config{});

    // Full-sequence forward pass over the prompt token ids -> FP32 logits [seq, vocab].
    // Prefill only: no KV cache, no incremental decode (those arrive after the math is
    // proven). Greedy-decoding argmax of the last row gives the next-token prediction.
    Tensor forward(const std::vector<int>& token_ids) const;
};

}  // namespace atlas
