##
## Serialization unit tests (GoogleTest)
##
## Builds:
## - `core_gtest`: currently focused on NPZ serialization round-trips.
##
## Notes:
## - Links against `cnpy-static` + zlib and uses BLAS/LAPACK/OpenBLAS variables
##   as configured in the top-level package checks.
##
add_executable(core_gtest
    NetworkSerializer_gtest.cpp
    StateIO_gtest.cpp
    state_io_gtest.cpp
)

target_link_libraries(core_gtest
    PRIVATE 
        google_test 
        xtensor xtensor-blas
        cnpy-static
        z
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${OPENBLAS_LINK_LIBRARIES}
        tensor
)

target_include_directories(core_gtest
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/src/core/tensor 
    ${CMAKE_SOURCE_DIR}/include
        ${cnpy_SOURCE_DIR}
)

include(GoogleTest)
nn_gtest_discover_tests(core_gtest)