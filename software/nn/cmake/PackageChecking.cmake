#+#+#+#+-----------------------------------------------------------------------
# PackageChecking.cmake
#
# System dependency discovery.
#
# This module is where we require system-provided packages (OpenMP,
# BLAS/LAPACK/OpenBLAS, etc). Vendored dependencies are handled separately via
# `Vendor*.cmake` modules.
#+#+#+#+-----------------------------------------------------------------------

# Package finder
find_package(PkgConfig REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)