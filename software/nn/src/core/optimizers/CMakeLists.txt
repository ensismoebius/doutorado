add_executable(optimizers_gtest tests/optimizers_gtest.cpp)
target_link_libraries(optimizers_gtest PRIVATE GTest::gtest_main Eigen3::Eigen)
target_include_directories(optimizers_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src
                                                    ${EIGEN3_INCLUDE_DIR})
include(GoogleTest)
gtest_discover_tests(optimizers_gtest)

configure_eigen_parallel_target(optimizers_gtest)
