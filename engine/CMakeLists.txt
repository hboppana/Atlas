# The inference engine. Phase 1: a static lib with the Tensor foundation only.
# Model/tokenizer/quantize sources and the CUDA path get added in later steps.
add_library(atlas_engine STATIC
    src/tensor.cpp
)
target_include_directories(atlas_engine PUBLIC include)
target_compile_features(atlas_engine PUBLIC cxx_std_17)

# --- tests (zero-dependency harness, registered with CTest) ---
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE atlas_engine)
add_test(NAME test_tensor COMMAND test_tensor)

# When built with MinGW-w64 (MSYS2 GCC), statically link the GCC/libstdc++ runtimes
# so the test exe runs from any shell without MSYS2's ucrt64\bin on PATH.
if (MINGW)
    target_link_options(test_tensor PRIVATE -static -static-libgcc -static-libstdc++)
endif()
