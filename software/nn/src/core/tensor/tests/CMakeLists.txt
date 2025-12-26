add_executable(tensor_gtest tensor_gtest.cpp)

target_link_libraries(tensor_gtest 
    PRIVATE 
        google_test 
        Eigen3::Eigen 
        tensor
)

target_include_directories(tensor_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
        ${EIGEN3_INCLUDE_DIR}
)

include(GoogleTest)
gtest_discover_tests(tensor_gtest)
configure_eigen_parallel_target(tensor_gtest)