# Package finder
find_package(PkgConfig REQUIRED)

# Find Eigen using pkg-config
pkg_check_modules(EIGEN3 REQUIRED eigen3)

# Create an IMPORTED INTERFACE library to maintain compatibility with find_package
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${EIGEN3_CFLAGS_OTHER}"
    )
endif()

# Set EIGEN3_INCLUDE_DIR for backward compatibility
set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)

# Find BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)