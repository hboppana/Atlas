# The inference engine. Phase 1: the Tensor foundation + BPE tokenizer.
# Model/quantize sources and the CUDA path get added in later steps.
add_library(atlas_engine STATIC
    src/tensor.cpp
    src/tokenizer.cpp
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

# When built with MinGW-w64 (MSYS2 GCC), statically link the GCC/libstdc++ runtimes
# so the test exes run from any shell without MSYS2's ucrt64\bin on PATH.
if (MINGW)
    target_link_options(test_tensor PRIVATE -static -static-libgcc -static-libstdc++)
    target_link_options(test_tokenizer PRIVATE -static -static-libgcc -static-libstdc++)
endif()
