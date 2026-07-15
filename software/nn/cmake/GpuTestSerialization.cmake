# -----------------------------------------------------------------------------
# GpuTestSerialization.cmake
#
# A hard, reboot-requiring hang hit this project's dev hardware (an AMD
# Renoir/Lucienne integrated GPU) via AdaptiveCpp's HIP backend under
# concurrent SYCL ctest workers. A second freeze followed a full-suite
# `ctest -j$(nproc)` run under NN_BACKEND=OpenCL, but the user determined
# that one was residual fallout from the SYCL/HIP incident, not OpenCL
# concurrency itself — OpenCL-touching tests had already run fine at high
# parallelism earlier in the same session (mixed into the default XTensor
# preset's full suite). So the automatic per-preset lock now only applies to
# NN_BACKEND=SYCL. `-march=native`/LTO flags are irrelevant either way — this
# is a driver/runtime limit, not a build setting.
#
# nn_gtest_discover_tests() wraps gtest_discover_tests() and applies a shared
# RESOURCE_LOCK to every discovered test so `ctest -jN` can never run two
# GPU-touching tests concurrently, regardless of N — this is enforced by
# CTest itself, not by remembering to pass a lower -j by hand. Every
# gtest_discover_tests() call in this project should go through this wrapper.
#
# The lock applies automatically only when NN_BACKEND is SYCL (every test
# binary in that preset uses it as nn::Backend, and it's the driver stack
# that actually reproduced the hang). Pass FORCE_GPU_LOCK for a target that
# explicitly instantiates SYCLTensorBackend regardless of the selected
# nn::Backend (e.g. pytorch_parity_gtest, sycl_backend_parity_gtest) — those
# touch a real SYCL device even under NN_BACKEND=XTensor/OpenCL/Device.
# Targets that only ever touch OpenCL directly (opencl_tensor_backend_gtest,
# gpu_buffer_pool_gtest, tensor_backend_switchability_gtest,
# backend_parity_gtest, tensor_all_backends_gtest) are intentionally NOT
# force-locked — OpenCL concurrency is not currently believed to be the hang
# trigger.
# -----------------------------------------------------------------------------

function(nn_gtest_discover_tests target)
    cmake_parse_arguments(NN_GTD "FORCE_GPU_LOCK" "" "" ${ARGN})

    if(NN_GTD_FORCE_GPU_LOCK OR NN_BACKEND STREQUAL "SYCL")
        gtest_discover_tests(${target} PROPERTIES RESOURCE_LOCK nn_gpu_device)
    else()
        gtest_discover_tests(${target})
    endif()
endfunction()
