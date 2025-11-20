add_executable(Experiment01_utils_gtest Experiment01_utils_gtest.cpp)

target_link_libraries(Experiment01_utils_gtest
    PRIVATE
        GTest::gtest_main
        Eigen3::Eigen
        tensor
        dataLoaders
        util
        experiment01_utils_lib # Link against the new library
)

target_include_directories(Experiment01_utils_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/experiments/01
        ${CMAKE_SOURCE_DIR}/src/core/tensor
        ${EIGEN3_INCLUDE_DIR}
)

include(GoogleTest)
gtest_discover_tests(Experiment01_utils_gtest)
configure_eigen_parallel_target(Experiment01_utils_gtest)