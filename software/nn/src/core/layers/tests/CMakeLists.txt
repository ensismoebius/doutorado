add_executable(layers_gtest
    layers_gtest.cpp
)

target_link_libraries(layers_gtest 
    PRIVATE 
        google_test 
        Eigen3::Eigen 
        layers
        tensor
)

target_include_directories(
  layers_gtest 
    PRIVATE 
        ${EIGEN3_INCLUDE_DIR}
        ${cnpy_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(layers_gtest PROPERTIES CXX_CLANG_TIDY "")

include(GoogleTest)
gtest_discover_tests(layers_gtest)
configure_eigen_parallel_target(layers_gtest)
