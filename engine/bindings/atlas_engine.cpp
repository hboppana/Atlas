// Atlas Phase 2 · Step 8 — the pybind11 extension module (docs/14-bridge-serving.md).
//
// A thin, deliberately minimal binding over the proven C++ engine: Python never
// reimplements inference, it calls Tokenizer::encode -> {Model,GpuModel}::generate ->
// Tokenizer::decode and nothing else. server/bridge.py is the public face; this module is
// named with a leading underscore to say so (and to avoid colliding with the CMake static
// library target `atlas_engine`).
//
// What is NOT bound, on purpose:
//   - forward(): binding it means designing a Tensor <-> NumPy ownership contract over
//     views into a 4.4 GB mapping. No caller through Phase 5 needs raw logits today.
//   - quantize_int8(): out of scope for the serving path (GpuModel ignores qweights).
//
// GpuModel is compiled in only when ATLAS_WITH_CUDA is defined (i.e. the module was built
// in the ATLAS_USE_CUDA=ON tree). Which .so gets imported is therefore what selects the
// device; `has_cuda` is how Python asks. There is no runtime CUDA probe here.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "../include/model.h"
#include "../include/tokenizer.h"

#ifdef ATLAS_WITH_CUDA
#include "../cuda/forward.h"
#endif

namespace py = pybind11;

namespace {

// The generate() binding, shared by Model and GpuModel because docs/13 gave them the
// identical signature on purpose. Three things happen here that a naive `def("generate",
// &Model::generate)` would get wrong:
//
//  1. GIL. generate() is a multi-second pure-C++ loop; held under the GIL it would freeze
//     the FastAPI event loop, including the code trying to send the tokens it produces.
//     The GIL is released around the call and re-acquired per token inside the callback
//     (which docs/13 guarantees fires inline on this same thread, so there is no
//     cross-thread state to reason about).
//  2. A callback returning None means "keep going". Python functions that fall off the
//     end return None, and reading that as False would silently stop after one token.
//  3. An exception raised inside the Python callback must not unwind through CUDA code.
//     It is captured, the loop is stopped cleanly by returning false, and it is re-raised
//     on the C++ side once generate() has returned and the GIL is back.
template <class ModelT>
std::vector<int> generate_bound(const ModelT& model,
                                const std::vector<int>& prompt_ids,
                                int max_new_tokens,
                                py::object on_token) {
    std::exception_ptr callback_error;
    std::function<bool(int)> callback;

    if (!on_token.is_none()) {
        callback = [&](int token_id) -> bool {
            py::gil_scoped_acquire gil;
            if (callback_error) return false;  // already failing; stop without calling again
            try {
                py::object keep_going = on_token(token_id);
                return keep_going.is_none() ? true : keep_going.cast<bool>();
            } catch (...) {
                callback_error = std::current_exception();
                return false;
            }
        };
    }

    std::vector<int> generated;
    {
        py::gil_scoped_release release;
        generated = model.generate(prompt_ids, max_new_tokens, callback);
    }
    if (callback_error) std::rethrow_exception(callback_error);
    return generated;
}

constexpr const char* kGenerateDoc =
    "Greedy decode: argmax the last logit row, append, repeat.\n\n"
    "Returns the generated ids only (without the prompt). Stops at max_new_tokens or at\n"
    "EOS, which is not included. on_token, when given, is called with each id as it is\n"
    "produced and may return False to stop early -- it is called before the next forward\n"
    "begins, so abandoning a request stops paying for it immediately. Returning None\n"
    "means keep going.\n\n"
    "No KV cache: every step re-runs the full forward over the grown sequence.";

}  // namespace

PYBIND11_MODULE(_atlas_engine, m) {
    m.doc() = "Atlas C++/CUDA inference engine (Phase 2 Step 8 bridge).";

    // Compile-time facts of THIS .so. bridge.py branches on has_cuda rather than probing.
    m.attr("has_cuda") =
#ifdef ATLAS_WITH_CUDA
        true;
#else
        false;
#endif
    m.attr("BOS_ID") = atlas::Tokenizer::kBosId;
    m.attr("EOS_ID") = atlas::Tokenizer::kEosId;
    m.attr("UNK_ID") = atlas::Tokenizer::kUnkId;

    py::class_<atlas::Tokenizer>(m, "Tokenizer",
                                 "The TinyLlama SentencePiece-BPE tokenizer (engine/src/tokenizer.cpp).")
        .def_static("load", &atlas::Tokenizer::load, py::arg("vocab_path"), py::arg("merges_path"),
                    "Load the vocab.txt / merges.txt exported by scripts/export_tokenizer.py.")
        .def("encode", &atlas::Tokenizer::encode, py::arg("text"), py::arg("add_bos") = true,
             "Text -> token ids, BOS (id 1) prepended by default.")
        .def("decode", &atlas::Tokenizer::decode, py::arg("ids"),
             "Token ids -> text. Skips specials, reassembles <0xNN> byte runs, and strips\n"
             "the single leading space -- so this is NOT a per-token map. Decode a growing\n"
             "prefix and emit the delta when streaming (docs/14-bridge-serving.md).")
        .def_property_readonly("vocab_size", &atlas::Tokenizer::vocab_size);

    // Move-only (WeightStore owns an mmap): construction goes through the static factory,
    // which returns by value and lets pybind11 move into the holder. No copy is exposed --
    // a copied Model would double-unmap.
    py::class_<atlas::Model>(m, "Model",
                             "TinyLlama on the CPU: mmap'd FP32 weights + the forward pass.")
        // Wrapped rather than bound directly: Model::load's third parameter is the pinned
        // Config, defaulted in C++ but invisible to pybind11. Config is not exposed --
        // reference/config.json is the single source of truth for hyperparameters.
        .def_static("load",
                    [](const std::string& bin_path, const std::string& manifest_path) {
                        return atlas::Model::load(bin_path, manifest_path);
                    },
                    py::arg("bin_path"), py::arg("manifest_path"),
                    "Load model.f32.bin + model.manifest.txt (the blob is mmap'd, not read).")
        .def("generate", &generate_bound<atlas::Model>, py::arg("prompt_ids"),
             py::arg("max_new_tokens"), py::arg("on_token") = py::none(), kGenerateDoc);

#ifdef ATLAS_WITH_CUDA
    // keep_alive<0, 1>: GpuModel holds a NON-OWNING const Model* whose weight views point
    // into the Model's mmap (engine/cuda/forward.h). Without this, the idiomatic
    // GpuModel.create(Model.load(...)) would drop the last reference to the Model on the
    // same line and leave the GpuModel reading an unmapped blob -- a use-after-free that
    // surfaces as garbage logits or a segfault far from the cause.
    py::class_<atlas::GpuModel>(m, "GpuModel",
                                "TinyLlama on the GPU: the Step 6 forward pass + Step 7 decode loop.")
        .def_static("create", &atlas::GpuModel::create, py::arg("model"), py::keep_alive<0, 1>(),
                    "Upload the weight blob to the device once (~4.4 GB). The Model is kept\n"
                    "alive for the GpuModel's lifetime.")
        .def("generate", &generate_bound<atlas::GpuModel>, py::arg("prompt_ids"),
             py::arg("max_new_tokens"), py::arg("on_token") = py::none(), kGenerateDoc);
#endif
}
