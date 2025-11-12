add_executable(tensor_gtest tests/tensor_gtest.cpp Tensor.hpp)
target_link_libraries(tensor_gtest PRIVATE GTest::gtest_main Eigen3::Eigen)
target_include_directories(tensor_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src
                                                ${EIGEN3_INCLUDE_DIR})
include(GoogleTest)
gtest_discover_tests(tensor_gtest)

configure_eigen_parallel_target(tensor_gtest)
