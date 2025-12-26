# Add util_gtest
add_executable(util_gtest 
    util_gtest.cpp 
)
target_link_libraries(util_gtest 
    PRIVATE 
        google_test 
        Eigen3::Eigen
        cnpy-static
        z
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${OPENBLAS_LIBRARIES}
        util
        tensor
)

target_include_directories(util_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src 
        ${CMAKE_SOURCE_DIR}/src/core/tensor 
        ${EIGEN3_INCLUDE_DIR} 
        ${cnpy_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(util_gtest)