# Building the Atlas C++ Engine (Windows / MinGW-w64)

> Scope: how to configure, build, and test the `engine/` C++ project on this Windows dev
> machine. Companion to the per-step design docs (see
> [01-tensor-foundation.md](01-tensor-foundation.md)). This is the canonical build reference
> and records the toolchain decision made while landing Phase 1 · Step 1.

## What this step delivered

The first compilable, tested slice of the inference engine:

```
CMakeLists.txt                 root project — C++17, ATLAS_USE_CUDA OFF, enable_testing()
engine/
  CMakeLists.txt               atlas_engine static lib + test_tensor (CTest) + MinGW static-link
  include/tensor.h             Tensor: owns-or-views, move-only, free-function ops
  src/tensor.cpp               strides, reshape, matmul / add / mul
  tests/test_tensor.cpp        zero-dependency CHECK harness, 5 cases
```

Build is green and `ctest` reports `test_tensor … Passed` (1/1). Definition of done from
[01-tensor-foundation.md](01-tensor-foundation.md) is met.

## Toolchain decision: MinGW-w64 GCC (not MSVC)

The original step doc assumed **MSVC**. This machine has **no Visual Studio / Build Tools
installed**, but does have **MSYS2 UCRT64 GCC 15.2** plus **CMake 4.2**. We build with GCC:

- conda-forge on Windows offers no better native compiler here — its native option is an
  **MSVC wrapper** (`vs20xx_win-64`, needs VS installed) or a **stale GCC 8.x**
  (`m2w64-toolchain`). System MSYS2 GCC 15.2 is newer and already present.
- Phase 1 is CPU-only and correctness-first; a modern C++17 GCC is fully sufficient.
- MSVC remains optional later — the CMake files are generator-agnostic, so a VS build can be
  added without code changes if we ever want to validate both compilers.

**conda boundary:** the C++ engine compiles with the **system MSYS2 GCC**; **conda is for the
Python layers** (Step 0's `download_weights.py`, and Phases 2–5 — FastAPI, and `pybind11`
which builds the bridge against the conda env's Python). Don't reach for conda to build the
C++.

## Prerequisites

- **MSYS2** with the UCRT64 toolchain at `C:\msys64\ucrt64\bin` (provides `gcc`, `g++ 15.2`,
  `mingw32-make`, `gdb`). Install via `pacman -S mingw-w64-ucrt-x86_64-gcc` in MSYS2.
- **CMake ≥ 3.20** (tested with 4.2) on PATH.

## Build & test

Run from the repo root in PowerShell:

```powershell
# 1. Put UCRT64 on PATH for this session so g++ can load its runtime DLLs (see gotcha below)
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"

# 2. Configure with the MinGW Makefiles generator (single-config)
cmake -S . -B build -G "MinGW Makefiles"

# 3. Build
cmake --build build

# 4. Run the tests
ctest --test-dir build --output-on-failure
```

`MinGW Makefiles` is **single-config**, so there is no `--config Debug` and `ctest` needs no
`-C`. The test binary lands at `build/engine/test_tensor.exe` (running via `ctest` avoids
hard-coding that path).

## Gotchas learned landing this step

1. **"The C++ compiler is broken" = PATH/DLL, not code.** CMake invokes `g++.exe` by full
   path, so it *finds* the exe, but g++ then can't locate its own dependency DLLs
   (`libstdc++-6.dll`, `libwinpthread-1.dll`, …) unless `C:\msys64\ucrt64\bin` is on PATH. It
   crashes silently and CMake reports a broken compiler. Fix = step 1 above. To make it
   permanent (no per-session prepend):
   ```powershell
   [Environment]::SetEnvironmentVariable("Path","C:\msys64\ucrt64\bin;"+[Environment]::GetEnvironmentVariable("Path","User"),"User")
   ```
   …or just launch the **MSYS2 UCRT64** shortcut, which sets the environment up.
2. **Stale generator in `build/`.** The very first configure (before PATH was fixed) cached
   `CMAKE_GENERATOR = NMake Makefiles` and a failed-compiler result. CMake reuses a cached
   generator, so switching toolchains requires deleting `build/` first:
   `Remove-Item -Recurse -Force build`.
3. **`MinGW Makefiles` + `sh.exe`.** If `sh.exe` (e.g. from Git for Windows) is on PATH, this
   generator errors with *"sh.exe was found in your PATH"*. It is not on PATH here; keep it
   that way, or don't run the build from a shell/conda env that injects one.
4. **Self-contained test exe.** `engine/CMakeLists.txt` adds
   `-static -static-libgcc -static-libstdc++` to `test_tensor` under `if (MINGW)`, so the
   exe runs from any shell without UCRT64 on PATH at *runtime* (the compiler still needs it
   at *build* time).

## Repo hygiene

`build/`, `*.o`, `*.obj`, `*.exe` are gitignored. `__cmake_systeminformation` (a stray
`cmake --system-information` artifact) is also ignored. `reference/` stays tracked (test
oracles). Nothing built or generated is committed.

## Status & next

- **Phase 1 · Step 1 (tensor foundation + C++ build): done.**
- Next in the chain: **tokenizer** (`engine/{include,src}/tokenizer.*`,
  `engine/tests/test_tokenizer.cpp`), validated against `reference/token_ids.npy`.
