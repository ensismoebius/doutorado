# -----------------------------------------------------------------------------
# SyclGpuCapabilityCheck.cmake
#
# Verifies, at configure time, that the machine building NN_BACKEND=SYCL has
# GPU hardware AdaptiveCpp can actually drive safely, and refuses to configure
# otherwise. No silent CPU fallback: if the SYCL backend can't be trusted on
# this device, the build fails loudly here instead of the program silently
# running host math while the user believes it ran on GPU (or, worse, hanging
# the GPU at runtime).
#
# Background (found 2026-07-15, this project's dev machine): AMD Renoir/
# Lucienne/Cezanne/Barcelo mobile APUs (gfx90c) are NOT on ROCm's officially
# supported hardware list. AdaptiveCpp still happily routes SYCL compute to
# them via its HIP backend, where it reproducibly triggered a genuine GPU hang
# ("HW Exception ... reason: GPU Hang" from the ROCm HSA runtime) under
# concurrent kernel submission — which on an integrated GPU takes the display
# compositor down with it. The commonly-cited workaround, HSA_OVERRIDE_GFX_VERSION
# to impersonate a supported chip, is itself documented by other users to
# sometimes crash the GPU hard enough to require a reboot — not a fix, just a
# different failure mode. See:
#   https://github.com/ROCm/ROCm/issues/2216
#   https://github.com/ROCm/ROCm/issues/5121
#   https://github.com/lamikr/rocm_sdk_builder/issues/112
#
# IMPORTANT: acpp-info -l reports a generic marketing name for AMD devices
# ("AMD Radeon Graphics" for every Renoir/Lucienne/Cezanne/Vega-iGPU alike) —
# it carries no codename or gfx-architecture info, so matching against it
# cannot distinguish a known-bad APU from a normal discrete GPU. rocminfo's
# per-agent `Name:` field (e.g. "gfx90c") is the actual ISA/codename and is
# what this check matches against. Where ROCm/rocminfo isn't present (e.g.
# non-AMD hardware, or a SYCL implementation not using HIP), this specific
# denylist check is skipped, but the "is there any GPU at all" check below
# still runs via acpp-info.
#
# This check only reads device metadata (rocminfo, acpp-info -l) — it never
# submits a kernel, so it cannot itself trigger the hang.
# -----------------------------------------------------------------------------

function(nn_check_sycl_gpu_capability)
    option(NN_SYCL_ACKNOWLEDGE_UNSUPPORTED_GPU
        "I understand this GPU is not officially supported for SYCL compute and accept the risk of driver hangs/crashes (see cmake/SyclGpuCapabilityCheck.cmake)"
        OFF)

    if(NN_SYCL_ACKNOWLEDGE_UNSUPPORTED_GPU)
        message(WARNING
            "NN_BACKEND=SYCL: NN_SYCL_ACKNOWLEDGE_UNSUPPORTED_GPU is ON — "
            "skipping the hardware safety gate. You have accepted the risk of "
            "GPU driver hangs/crashes on this machine.")
        return()
    endif()

    # ── Step 1: rocminfo-based check for known-unsupported ROCm gfx codes ──────
    # Denylist of confirmed-unsupported chips, not an allowlist — anything not
    # matched here is assumed usable. Extend as more bad combinations are
    # confirmed elsewhere.
    set(NN_SYCL_UNSUPPORTED_GFX_CODES
        "gfx90c" # Renoir / Lucienne / Cezanne / Barcelo mobile APUs
    )

    find_program(NN_ROCMINFO_EXECUTABLE NAMES rocminfo PATHS /opt/rocm/bin)
    if(NN_ROCMINFO_EXECUTABLE)
        execute_process(
            COMMAND "${NN_ROCMINFO_EXECUTABLE}"
            OUTPUT_VARIABLE NN_ROCMINFO_OUTPUT
            ERROR_QUIET
            RESULT_VARIABLE NN_ROCMINFO_RESULT
            TIMEOUT 15
        )
        if(NN_ROCMINFO_RESULT EQUAL 0)
            # rocminfo emits ANSI color codes ("ROCk module is loaded" banner)
            # unconditionally, even when stdout is a pipe (not a tty). CMake's
            # list splitting silently collapses to a single element if those
            # raw ESC (0x1B) bytes survive into the string used with
            # `foreach(... IN LISTS ...)` — so strip them first or the block
            # parse below silently never matches anything.
            string(ASCII 27 NN_ESC)
            string(REGEX REPLACE "${NN_ESC}\\[[0-9;]*m" "" NN_ROCMINFO_OUTPUT "${NN_ROCMINFO_OUTPUT}")

            # Each HSA agent block looks like:
            #   Agent 2
            #   *******
            #     Name:                    gfx90c
            #     ...
            #     Device Type:             GPU
            # Track the most recent "Name:" seen; when a block's Device Type
            # is GPU, that cached name is the agent's gfx ISA code.
            string(REPLACE "\n" ";" NN_ROCMINFO_LINES "${NN_ROCMINFO_OUTPUT}")
            set(NN_SYCL_PENDING_GFX_NAME "")
            foreach(NN_LINE IN LISTS NN_ROCMINFO_LINES)
                if(NN_LINE MATCHES "^  Name:[ \t]+([A-Za-z0-9_-]+)")
                    set(NN_SYCL_PENDING_GFX_NAME "${CMAKE_MATCH_1}")
                elseif(NN_LINE MATCHES "Device Type:[ \t]+GPU")
                    foreach(NN_BAD_GFX IN LISTS NN_SYCL_UNSUPPORTED_GFX_CODES)
                        if(NN_SYCL_PENDING_GFX_NAME STREQUAL NN_BAD_GFX)
                            set(NN_SYCL_UNSUPPORTED_GFX_FOUND "${NN_SYCL_PENDING_GFX_NAME}")
                        endif()
                    endforeach()
                endif()
            endforeach()
        endif()
    endif()

    if(DEFINED NN_SYCL_UNSUPPORTED_GFX_FOUND)
        message(FATAL_ERROR
"\n"
"################################################################################\n"
"###                                                                          ###\n"
"###   BIG FAT WARNING: SYCL GPU ON THIS MACHINE IS NOT OFFICIALLY SUPPORTED  ###\n"
"###                                                                          ###\n"
"################################################################################\n"
"\n"
"Detected GPU ISA: ${NN_SYCL_UNSUPPORTED_GFX_FOUND} (via rocminfo)\n"
"\n"
"This chip is on ROCm's known-unsupported list (confirmed on this project's\n"
"own dev machine: running SYCL compute on it via AdaptiveCpp's HIP backend\n"
"caused a genuine GPU hang under concurrent kernel submission, which froze the\n"
"whole desktop compositor along with it, requiring a hard recovery).\n"
"\n"
"Configure is refusing to proceed with NN_BACKEND=SYCL on this hardware.\n"
"Options:\n"
"  1. Use a different backend: cmake --preset=max-performance (XTensor) or\n"
"     cmake --preset=max-performance-opencl.\n"
"  2. If you have verified THIS EXACT machine handles concurrent SYCL kernel\n"
"     submission on this GPU without hanging, re-run configure with:\n"
"       -DNN_SYCL_ACKNOWLEDGE_UNSUPPORTED_GPU=ON\n"
"     understanding that upstream reports (see cmake/SyclGpuCapabilityCheck.cmake)\n"
"     include cases of this workaround crashing the GPU badly enough to need a\n"
"     full reboot.\n"
"################################################################################\n"
        )
    endif()

    # ── Step 2: is there any GPU at all? (vendor-agnostic, via acpp-info) ──────
    find_program(NN_ACPP_INFO_EXECUTABLE NAMES acpp-info)
    if(NOT NN_ACPP_INFO_EXECUTABLE)
        message(WARNING
            "NN_BACKEND=SYCL: could not find acpp-info to verify a GPU is present. "
            "Proceeding without that check — device availability is unverified.")
        return()
    endif()

    execute_process(
        COMMAND "${NN_ACPP_INFO_EXECUTABLE}" -l
        OUTPUT_VARIABLE NN_ACPP_INFO_OUTPUT
        ERROR_VARIABLE NN_ACPP_INFO_OUTPUT_ERR
        RESULT_VARIABLE NN_ACPP_INFO_RESULT
        TIMEOUT 15
    )
    set(NN_ACPP_INFO_OUTPUT "${NN_ACPP_INFO_OUTPUT}${NN_ACPP_INFO_OUTPUT_ERR}")

    if(NOT NN_ACPP_INFO_RESULT EQUAL 0)
        message(WARNING
            "NN_BACKEND=SYCL: acpp-info -l failed to run (result=${NN_ACPP_INFO_RESULT}). "
            "Proceeding without a hardware safety check — device usage is unverified.")
        return()
    endif()

    # acpp-info -l output looks like:
    #   Loaded backend 1: HIP
    #     Found device: AMD Radeon Graphics
    #   Loaded backend 3: OpenMP
    #     Found device: AdaptiveCpp OpenMP host device
    # Walk it tracking the current backend so "Found device" lines under the
    # OpenMP (host CPU) backend are never mistaken for GPU hardware.
    string(REPLACE "\n" ";" NN_ACPP_INFO_LINES "${NN_ACPP_INFO_OUTPUT}")
    set(NN_SYCL_CURRENT_BACKEND "")
    set(NN_SYCL_FOUND_NON_CPU_DEVICE FALSE)
    foreach(NN_LINE IN LISTS NN_ACPP_INFO_LINES)
        if(NN_LINE MATCHES "Loaded backend [0-9]+: ([A-Za-z0-9 ]+)")
            set(NN_SYCL_CURRENT_BACKEND "${CMAKE_MATCH_1}")
        elseif(NN_LINE MATCHES "Found device:" AND NOT NN_SYCL_CURRENT_BACKEND STREQUAL "OpenMP")
            set(NN_SYCL_FOUND_NON_CPU_DEVICE TRUE)
        endif()
    endforeach()

    if(NOT NN_SYCL_FOUND_NON_CPU_DEVICE)
        message(FATAL_ERROR
"\n"
"################################################################################\n"
"###                                                                          ###\n"
"###   BIG FAT WARNING: NO GPU AVAILABLE FOR THE SYCL BACKEND                 ###\n"
"###                                                                          ###\n"
"################################################################################\n"
"\n"
"acpp-info -l found no non-CPU SYCL device on this machine — only the OpenMP\n"
"host device backend is available.\n"
"\n"
"NN_BACKEND=SYCL exists to run tensor math on a GPU. Without one, the backend\n"
"cannot do the thing it is for, so configure refuses to proceed rather than\n"
"silently building a program that only ever runs on CPU while claiming GPU.\n"
"Use cmake --preset=max-performance (XTensor) instead, which IS the supported\n"
"CPU backend.\n"
"################################################################################\n"
        )
    endif()

    message(STATUS "NN_BACKEND=SYCL: GPU capability check passed.")
endfunction()

nn_check_sycl_gpu_capability()
