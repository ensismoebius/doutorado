# -----------------------------------------------------------------------------
# OpenCLGpuCapabilityCheck.cmake
#
# Verifies, at configure time, that a real OpenCL device is present when
# NN_BACKEND=OpenCL is selected, and refuses to configure otherwise.
#
# OpenCLTensorBackend.cpp already has no runtime CPU fallback: every compute
# method either takes the OpenCL kernel path or throws (see
# `throw_opencl_only_failure` / `can_use_opencl()` in that file) — there is no
# host-math substitute hidden behind those checks despite the historical
# "cpu_fallback" naming. This CMake check exists so the failure surfaces at
# configure time with an explicit, actionable message instead of only at
# first-op runtime.
#
# Unlike the SYCL backend (see SyclGpuCapabilityCheck.cmake), there is no
# known-bad-hardware denylist here: this project's dev machine already runs
# the OpenCL backend successfully through Mesa's rusticl driver (a completely
# different, unrelated driver stack from AdaptiveCpp/HIP — see
# results/opencl_lhs_transposed_benchmark_2026-05-02.md), so the only failure
# mode worth gating on is "no OpenCL device at all."
# -----------------------------------------------------------------------------

function(nn_check_opencl_gpu_capability)
    option(NN_OPENCL_ACKNOWLEDGE_NO_GPU
        "I understand no OpenCL device was detected and accept that the OpenCL backend will throw at runtime instead of silently running on CPU"
        OFF)

    if(NN_OPENCL_ACKNOWLEDGE_NO_GPU)
        message(WARNING
            "NN_BACKEND=OpenCL: NN_OPENCL_ACKNOWLEDGE_NO_GPU is ON — skipping the "
            "device capability gate.")
        return()
    endif()

    if(NOT OpenCL_FOUND)
        message(FATAL_ERROR
"\n"
"################################################################################\n"
"###                                                                          ###\n"
"###   BIG FAT WARNING: NO OPENCL RUNTIME AVAILABLE FOR THE OPENCL BACKEND    ###\n"
"###                                                                          ###\n"
"################################################################################\n"
"\n"
"find_package(OpenCL) did not find an OpenCL implementation on this machine.\n"
"\n"
"NN_BACKEND=OpenCL exists to run tensor math on a GPU. Without an OpenCL\n"
"runtime, OpenCLTensorBackend cannot do the thing it is for — its compute\n"
"methods already throw rather than silently falling back to CPU math (see\n"
"OpenCLGpuCapabilityCheck.cmake header comment), so configure refuses to\n"
"proceed rather than building a program guaranteed to throw on first tensor op.\n"
"Options:\n"
"  1. Install an OpenCL implementation (e.g. Arch: pacman -S opencl-mesa, or\n"
"     your GPU vendor's ICD) and reconfigure.\n"
"  2. Use a different backend: cmake --preset=max-performance (XTensor).\n"
"  3. If you intend to build anyway (e.g. compiling on a machine that isn't\n"
"     the target runtime host), re-run configure with:\n"
"       -DNN_OPENCL_ACKNOWLEDGE_NO_GPU=ON\n"
"################################################################################\n"
        )
    endif()

    find_program(NN_CLINFO_EXECUTABLE NAMES clinfo)
    if(NOT NN_CLINFO_EXECUTABLE)
        message(WARNING
            "NN_BACKEND=OpenCL: OpenCL headers/libs were found but clinfo isn't "
            "installed, so device presence (as opposed to just the runtime ICD "
            "loader) could not be verified. Proceeding — OpenCLTensorBackend will "
            "throw at runtime if no device is actually present.")
        return()
    endif()

    execute_process(
        COMMAND "${NN_CLINFO_EXECUTABLE}" -l
        OUTPUT_VARIABLE NN_CLINFO_OUTPUT
        ERROR_QUIET
        RESULT_VARIABLE NN_CLINFO_RESULT
        TIMEOUT 15
    )

    if(NOT NN_CLINFO_RESULT EQUAL 0 OR NN_CLINFO_OUTPUT MATCHES "^[ \t\r\n]*$")
        message(FATAL_ERROR
"\n"
"################################################################################\n"
"###                                                                          ###\n"
"###   BIG FAT WARNING: NO OPENCL DEVICE AVAILABLE FOR THE OPENCL BACKEND     ###\n"
"###                                                                          ###\n"
"################################################################################\n"
"\n"
"clinfo -l reported no OpenCL platforms/devices on this machine, even though\n"
"an OpenCL runtime is installed.\n"
"\n"
"Configure is refusing to proceed with NN_BACKEND=OpenCL without a device.\n"
"Options:\n"
"  1. Use a different backend: cmake --preset=max-performance (XTensor).\n"
"  2. If you intend to build anyway, re-run configure with:\n"
"       -DNN_OPENCL_ACKNOWLEDGE_NO_GPU=ON\n"
"################################################################################\n"
        )
    endif()

    message(STATUS "NN_BACKEND=OpenCL: GPU capability check passed.")
endfunction()

nn_check_opencl_gpu_capability()
