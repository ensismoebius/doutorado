##
## Utility unit tests (GoogleTest)
##
## Builds:
## - `util_gtest`: tests for batching, synthetic spike generators, and misc helpers.
##
# Add util_gtest
add_executable(util_gtest
    util_gtest.cpp
    transforms_gtest.cpp
    progress_gtest.cpp
)
target_link_libraries(util_gtest 
    PRIVATE 
        google_test 
         xtensor xtensor-blas
        cnpy-static
        z
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${OPENBLAS_LINK_LIBRARIES}
        util
        tensor
)

target_include_directories(util_gtest 
    PRIVATE 
         ${CMAKE_SOURCE_DIR}/src 
         ${CMAKE_SOURCE_DIR}/src/core/tensor 
         ${cnpy_SOURCE_DIR}
)

include(GoogleTest)
nn_gtest_discover_tests(util_gtest)