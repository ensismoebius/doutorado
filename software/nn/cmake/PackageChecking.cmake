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

# Eigen3: included by the 10.1117 loaders' public headers (EEGData.hpp, AudioData.hpp) and by
# the thesis PCA (ThesisPca.cpp). Until 2026-11-12 nothing declared it: the include path
# reached those targets only because matio-cpp, when ITS optional find_package(Eigen3 QUIET)
# succeeds, links Eigen publicly. On a machine without Eigen (GridUnesp's conda env) matio-cpp
# quietly built without it and the thesis failed to compile, far from the cause. Found here
# first, before the vendored dependencies, so matio-cpp's own lookup sees the same Eigen.
# No version: the code uses only Eigen 3.3+ API, and Eigen 5's version file rejects "3.3".
find_package(Eigen3 REQUIRED NO_MODULE)