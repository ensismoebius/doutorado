# Package finder
find_package(PkgConfig REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)

# Find BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)