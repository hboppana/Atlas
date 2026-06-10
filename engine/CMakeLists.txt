# The inference engine. Phase 1: the Tensor foundation, BPE tokenizer, and the
# TinyLlama forward pass. Quantize sources and the CUDA path get added in later steps.
add_library(atlas_engine STATIC
    src/tensor.cpp
    src/tokenizer.cpp
    src/model.cpp
)
target_include_directories(atlas_engine PUBLIC include)
target_compile_features(atlas_engine PUBLIC cxx_std_17)

# --- tests (zero-dependency harness, registered with CTest) ---
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE atlas_engine)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE atlas_engine)
# The tokenizer test loads the committed reference/tokenizer/*.txt fixtures; inject the
# absolute path so it resolves regardless of the CTest working directory.
target_compile_definitions(test_tokenizer PRIVATE
    ATLAS_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/reference")
add_test(NAME test_tokenizer COMMAND test_tokenizer)

# Forward-pass smoke test. Needs the tokenizer fixtures (committed) AND the converted
# weight blob (gitignored, regenerated locally by scripts/convert_weights.py) — it
# SKIPs green when the blob is absent so CI without weights stays green.
add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward PRIVATE atlas_engine)
target_compile_definitions(test_forward PRIVATE
    ATLAS_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/reference"
    ATLAS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights/tinyllama-1.1b-chat")
add_test(NAME test_forward COMMAND test_forward)

# --- CLI: load tokenizer + model, encode a prompt, forward, greedy-decode, print ---
add_executable(atlas_cli src/main.cpp)
set_target_properties(atlas_cli PROPERTIES OUTPUT_NAME atlas)
target_link_libraries(atlas_cli PRIVATE atlas_engine)
target_compile_definitions(atlas_cli PRIVATE
    ATLAS_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/reference"
    ATLAS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights/tinyllama-1.1b-chat")

# When built with MinGW-w64 (MSYS2 GCC), statically link the GCC/libstdc++ runtimes
# so the test exes run from any shell without MSYS2's ucrt64\bin on PATH.
if (MINGW)
    target_link_options(test_tensor PRIVATE -static -static-libgcc -static-libstdc++)
    target_link_options(test_tokenizer PRIVATE -static -static-libgcc -static-libstdc++)
    target_link_options(test_forward PRIVATE -static -static-libgcc -static-libstdc++)
    target_link_options(atlas_cli PRIVATE -static -static-libgcc -static-libstdc++)
endif()
