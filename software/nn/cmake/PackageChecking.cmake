# Package finder
find_package(PkgConfig REQUIRED)

# Find Eigen
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3/")
find_package(Eigen3 REQUIRED NO_MODULE)
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)

# Find BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)