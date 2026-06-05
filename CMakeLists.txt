# Atlas — top-level build.
# Phase 1 is CPU-only (MSVC, C++17). The CUDA path is guarded off here and only
# turned on in Phase 2 (engine/cuda/), so nothing CUDA is referenced for now.
cmake_minimum_required(VERSION 3.20)
project(atlas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ATLAS_USE_CUDA "Build CUDA kernels (Phase 2)" OFF)

enable_testing()

add_subdirectory(engine)
