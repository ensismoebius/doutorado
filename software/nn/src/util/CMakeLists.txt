
# Add util_gtest
add_executable(
    util_gtest 
    tests/util_gtest.cpp 
    batching.cpp 
    vectorizationCheck.cpp 
    synthetic_spike_data.cpp
)
target_link_libraries(
    util_gtest 
    PRIVATE GTest::gtest_main Eigen3::Eigen
    cnpy-static
    z
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
)

target_include_directories(
    util_gtest 
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core/tensor ${EIGEN3_INCLUDE_DIR} ${cnpy_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(util_gtest)
