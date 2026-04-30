# cmake/VendorXtensorParallel.cmake
# Provides configure_xtensor_parallel_target(<target>).
# Links OpenMP + BLAS/LAPACK. Replaces VendorEigenParallel.cmake.

function(configure_xtensor_parallel_target target)
    target_link_libraries(${target} PRIVATE
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        xtensor
        xtensor-blas)

    target_compile_definitions(${target} PRIVATE
        XTENSOR_USE_OPENMP=1)
endfunction()
