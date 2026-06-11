# The inference engine. Phase 1: the Tensor foundation, BPE tokenizer, the
# TinyLlama forward pass, and INT8 quantization. The CUDA path gets added in Phase 2.
add_library(atlas_engine STATIC
    src/tensor.cpp
    src/tokenizer.cpp
    src/model.cpp
    src/npy.cpp
    src/quantize.cpp
)
target_include_directories(atlas_engine PUBLIC include)
target_compile_features(atlas_engine PUBLIC cxx_std_17)

# --- tests (zero-dependency harness, registered with CTest) ---
# One helper per target: tests/<name>.cpp -> exe + CTest registration. Every test gets
# the reference/weights path defines (harmless where unused; the blob-gated tests SKIP
# green when the gitignored weight blob is absent so CI without weights stays green).
# Under MinGW the GCC/libstdc++ runtimes are statically linked so the exes run from any
# shell without MSYS2's ucrt64\bin on PATH.
function(atlas_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE atlas_engine)
    target_compile_definitions(${name} PRIVATE
        ATLAS_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/reference"
        ATLAS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights/tinyllama-1.1b-chat")
    add_test(NAME ${name} COMMAND ${name})
    if (MINGW)
        target_link_options(${name} PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
endfunction()

# Foundations + fixtures-based (blob-free).
atlas_add_test(test_tensor)
atlas_add_test(test_tokenizer)
atlas_add_test(test_tokenizer_edges)

# Component tests for the model math building blocks declared in model.h (blob-free —
# real model-math coverage on machines without the blob).
atlas_add_test(test_model_math)
atlas_add_test(test_rope)
atlas_add_test(test_attention)

# Format/loader contracts on synthetic files the tests write themselves; failure modes
# via self-subprocess death tests (blob-free).
atlas_add_test(test_npy)
atlas_add_test(test_weightstore)

# Full-pipeline oracle checks vs reference/logits.npy (blob-gated, ~40 s each).
atlas_add_test(test_forward)
atlas_add_test(test_quantize)

# --- CLI: load tokenizer + model, encode a prompt, forward, greedy-decode, print ---
add_executable(atlas_cli src/main.cpp)
set_target_properties(atlas_cli PROPERTIES OUTPUT_NAME atlas)
target_link_libraries(atlas_cli PRIVATE atlas_engine)
target_compile_definitions(atlas_cli PRIVATE
    ATLAS_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/reference"
    ATLAS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights/tinyllama-1.1b-chat")
if (MINGW)
    target_link_options(atlas_cli PRIVATE -static -static-libgcc -static-libstdc++)
endif()
