# Phase 2 · Step 8 — the pybind11 extension module (docs/14-bridge-serving.md).
# Built only when ATLAS_BUILD_PYTHON=ON. Produces build*/python/_atlas_engine*.so.

# Prefer the interpreter of the active environment (VIRTUAL_ENV / conda), because the
# module must be ABI-compatible with the interpreter that will import it, and because
# pybind11 is found through that interpreter below. Override with -DPython3_EXECUTABLE=...
if (NOT DEFINED Python3_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 comes from the environment (requirements.txt), not vendored: ask the very
# interpreter we just resolved where its CMake package lives.
if (NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_probe)
    if (NOT _pybind11_probe EQUAL 0)
        message(FATAL_ERROR
            "ATLAS_BUILD_PYTHON=ON but pybind11 is not installed for ${Python3_EXECUTABLE}.\n"
            "Install the Phase 2 deps into the active environment:\n"
            "  pip install -r requirements.txt")
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_atlas_engine atlas_engine.cpp)
target_link_libraries(_atlas_engine PRIVATE atlas_engine)

# When the CUDA path is on, the module additionally binds GpuModel. ATLAS_WITH_CUDA is the
# compile-time fact the module reports back to Python as `has_cuda`.
if (ATLAS_USE_CUDA)
    target_link_libraries(_atlas_engine PRIVATE atlas_cuda)
    target_compile_definitions(_atlas_engine PRIVATE ATLAS_WITH_CUDA=1)
endif()

# One well-known location per build tree, so server/bridge.py can discover the module by
# preferring build-cuda/python/ over build/python/ without guessing at CMake internals.
set_target_properties(_atlas_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
