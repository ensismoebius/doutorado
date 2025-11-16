add_executable(initializers_gtest tests/initializers_gtest.cpp)
target_link_libraries(initializers_gtest PRIVATE GTest::gtest_main
                                                 Eigen3::Eigen)
target_include_directories(initializers_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src
                                                      ${EIGEN3_INCLUDE_DIR})
include(GoogleTest)
gtest_discover_tests(initializers_gtest)

configure_eigen_parallel_target(initializers_gtest)
