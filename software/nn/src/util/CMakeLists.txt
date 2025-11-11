
# Add util_gtest
add_executable(
    util_gtest 
    util_gtest.cpp 
    NetworkSerializer_gtest.cpp
    batching.cpp 
    ../tensor/Tensor.hpp
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
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/lib/cnpy
)

include(GoogleTest)
gtest_discover_tests(util_gtest)
