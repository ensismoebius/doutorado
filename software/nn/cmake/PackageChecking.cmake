#+#+#+#+-----------------------------------------------------------------------
# PackageChecking.cmake
#
# System dependency discovery.
#
# This module is where we require system-provided packages (OpenMP, SDL2,
# BLAS/LAPACK/OpenBLAS, etc). Vendored dependencies are handled separately via
# `Vendor*.cmake` modules.
#+#+#+#+-----------------------------------------------------------------------

# Package finder
find_package(PkgConfig REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)

# Find BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)

# Find OpenCL (for GPU tensor backend)
# Use FindOpenCL from CMake if available, otherwise try pkg-config
find_package(OpenCL QUIET)
if (NOT OpenCL_FOUND)
    pkg_check_modules(OpenCL QUIET OpenCL)
endif()

if (OpenCL_FOUND)
    if(APPLE AND OpenCL_INCLUDE_DIR)
        # Apple's OpenCL.framework ships its headers flat (cl.h, cl_platform.h,
        # ... directly under OpenCL.framework/Headers, no CL/ subdirectory),
        # but the project sources use `#include <CL/cl.h>`. Expose the Headers
        # dir and provide a tiny shim CL/cl.h (-> <cl.h>) so the Khronos-style
        # include path works. On Linux/CI the CL headers live in the system
        # include path; these global dirs replicate that on Apple for every
        # target (including tests that include CL/cl.h directly).
        set(NN_OPENCL_COMPAT_DIR "${CMAKE_BINARY_DIR}/opencl_compat")
        file(MAKE_DIRECTORY "${NN_OPENCL_COMPAT_DIR}/CL")
        file(WRITE "${NN_OPENCL_COMPAT_DIR}/CL/cl.h"
            "#pragma clang system_header\n"
            "/* Shim: macOS OpenCL.framework headers are flat (no CL/ subdir). */\n"
            "#include <cl.h>\n")
        list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
            "${OpenCL_INCLUDE_DIR}/Headers"
            "${NN_OPENCL_COMPAT_DIR}")
        list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
            "${OpenCL_INCLUDE_DIR}/Headers"
            "${NN_OPENCL_COMPAT_DIR}")
        list(APPEND OpenCL_INCLUDE_DIRS
            "${NN_OPENCL_COMPAT_DIR}"
            "${OpenCL_INCLUDE_DIR}/Headers")
        list(REMOVE_DUPLICATES OpenCL_INCLUDE_DIRS)
    endif()
    message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
else()
    message(WARNING "OpenCL not found; GPU tensor backend will not be available")
endif()