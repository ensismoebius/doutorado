
# Add util_gtest
add_executable(
    util_gtest 
    util_gtest.cpp 
    batching.cpp 
    ../tensor/Tensor.hpp
    vectorizationCheck.cpp 
    synthetic_spike_data.cpp
)
target_link_libraries(
    util_gtest 
    PRIVATE GTest::gtest_main Eigen3::Eigen
)

target_include_directories(
    util_gtest 
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR}
)

include(GoogleTest)
gtest_discover_tests(util_gtest)
