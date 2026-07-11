# Phase 2 CUDA target — built only when ATLAS_USE_CUDA=ON (lab A6000 box, sm_86; off for the
# CPU-only build). Step 1 is infra-only: the DeviceTensor primitive + copy helpers +
# a no-math payload kernel. Real kernels (matmul, ...) join atlas_cuda from Step 2 on.
# See docs/07-cuda-build-matmul.md.

add_library(atlas_cuda STATIC
    device_tensor.cu
    matmul.cu
    rmsnorm.cu
    kernels.cu
)
# Links atlas_engine so CUDA code can use the host Tensor (and, from Step 2, linear() as
# the validation oracle). Headers are reached via relative include paths from the .cu files.
target_link_libraries(atlas_cuda PUBLIC atlas_engine)
set_target_properties(atlas_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# --- test (blob-free; round-trip + diff harness + CUDA_CHECK death case) ---
# Registered with CTest like the Phase 1 targets, run on the lab A6000 box via
# scripts/test_cuda.sh (`ctest -R test_device`). No MinGW static-link options — that is a
# Windows/GCC concern; the CUDA build is Linux/nvcc.
add_executable(test_device tests/test_device.cu)
target_link_libraries(test_device PRIVATE atlas_cuda)
add_test(NAME test_device COMMAND test_device)

# Step 2 matmul validation vs the linear() oracle, same harness as test_device.
add_executable(test_matmul tests/test_matmul.cu)
target_link_libraries(test_matmul PRIVATE atlas_cuda)
add_test(NAME test_matmul COMMAND test_matmul)

# Step 3 RMSNorm validation vs the rmsnorm() oracle, same harness.
add_executable(test_rmsnorm tests/test_rmsnorm.cu)
target_link_libraries(test_rmsnorm PRIVATE atlas_cuda)
add_test(NAME test_rmsnorm COMMAND test_rmsnorm)

# Step 4 utility kernels (embed/add/swiglu/rope) vs their CPU oracles, same harness.
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE atlas_cuda)
add_test(NAME test_kernels COMMAND test_kernels)
