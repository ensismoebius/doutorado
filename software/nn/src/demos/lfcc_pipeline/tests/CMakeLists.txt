add_executable(lfcc_pipeline_utils_gtest lfcc_pipeline_utils_gtest.cpp)

target_link_libraries(lfcc_pipeline_utils_gtest
    PRIVATE
        GTest::gtest_main
        Eigen3::Eigen
        tensor
        dataLoaders
        util
        lfcc_pipeline_utils_lib # Link against the new library
        waveCoreLib # Link against the new waveCoreLib
)

target_include_directories(lfcc_pipeline_utils_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/demos/lfcc_pipeline
        ${CMAKE_SOURCE_DIR}/src/core/tensor
        ${EIGEN3_INCLUDE_DIR}
)

include(GoogleTest)
gtest_discover_tests(lfcc_pipeline_utils_gtest)
configure_eigen_parallel_target(lfcc_pipeline_utils_gtest)