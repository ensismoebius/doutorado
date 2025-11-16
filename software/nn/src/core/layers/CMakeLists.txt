enable_testing()

add_executable(layers_gtest
    tests/layers_gtest.cpp
)

target_link_libraries(layers_gtest PRIVATE GTest::gtest_main Eigen3::Eigen)

target_include_directories(
  layers_gtest PRIVATE ${SRC_DIR} ${EIGEN3_INCLUDE_DIR}
                       ${cnpy_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(layers_gtest)

configure_eigen_parallel_target(layers_gtest)
