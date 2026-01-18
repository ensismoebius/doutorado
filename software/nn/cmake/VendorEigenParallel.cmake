##
## VendorEigenParallel.cmake
##
## Purpose
## - Centralize the “Eigen + OpenMP + BLAS/LAPACK” wiring for targets that want it.
##
## What it provides
## - `configure_eigen_parallel_target(<target>)`: links OpenMP + BLAS/LAPACK and applies
##   Eigen-related compile definitions.
##
## Assumptions / pitfalls
## - Expects `OpenMP::OpenMP_CXX` and `${BLAS_LIBRARIES}` / `${LAPACK_LIBRARIES}` to be available
##   (typically established by `cmake/PackageChecking.cmake`).
## - This is a performance configuration; for deterministic experiments, pin thread counts
##   explicitly (e.g., `OMP_NUM_THREADS=1`).
##

# Configure Eigen with parallel execution and enable vectorization
# add_definitions(-DEIGEN_MAX_ALIGN_BYTES=32) # Managed by target_compile_definitions
# add_definitions(-DEIGEN_NO_DEBUG)          # Managed by target_compile_definitions
# add_definitions(-DEIGEN_VECTORIZE)         # Managed by target_compile_definitions
# add_definitions(-DEIGEN_USE_BLAS)          # Managed by target_compile_definitions
# add_definitions(-DEIGEN_USE_OPENMP)        # Managed by target_compile_definitions
# add_definitions(-DEIGEN_DONT_PARALLELIZE=0) # Managed by target_compile_definitions

# Link flags for parallel execution - handled by target_link_libraries
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Function to configure target with Eigen parallelism
function(configure_eigen_parallel_target target)
    target_link_libraries(${target} PRIVATE 
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
    )
    
    # Enable maximum compiler optimizations - handled by global CMAKE_CXX_FLAGS_RELEASE
    # if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    #     target_compile_options(${target} PRIVATE -O3 -march=native -ffast-math)
    # endif()
    
    # Enable Eigen vectorization and parallelization
    target_compile_definitions(${target} PRIVATE
        EIGEN_MAX_ALIGN_BYTES=32
        EIGEN_NO_DEBUG
        EIGEN_VECTORIZE
        EIGEN_FAST_MATH
        EIGEN_USE_BLAS
        EIGEN_USE_LAPACKE
        EIGEN_USE_OPENMP
        EIGEN_DONT_PARALLELIZE=0
    )
endfunction()
