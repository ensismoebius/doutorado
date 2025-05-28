add_executable(
    util_gtest 
    util_gtest.cpp 
    batching.cpp 
    vectorizationCheck.hpp 
    ../tensor/Tensor.hpp
)
target_link_libraries(util_gtest PRIVATE GTest::gtest_main Eigen3::Eigen)
target_include_directories(util_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
include(GoogleTest)
gtest_discover_tests(util_gtest)
