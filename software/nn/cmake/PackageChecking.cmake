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
    message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
else()
    message(WARNING "OpenCL not found; GPU tensor backend will not be available")
endif()