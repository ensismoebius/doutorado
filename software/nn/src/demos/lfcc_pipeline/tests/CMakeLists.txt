##
## LFCC pipeline unit tests (GoogleTest)
##
## Builds:
## - `lfcc_pipeline_utils_gtest`: tests for feature extraction helpers used by the LFCC demo.
##
## Notes:
## - This links core libs only; LFCC helpers now live in `waveCoreLib`.
##
add_executable(lfcc_pipeline_utils_gtest lfcc_pipeline_utils_gtest.cpp)

target_link_libraries(lfcc_pipeline_utils_gtest
    PRIVATE
        google_test
        Eigen3::Eigen
        tensor
        dataLoaders
        util
        waveCoreLib # Link against the new waveCoreLib
)

target_include_directories(lfcc_pipeline_utils_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/tensor
        ${EIGEN3_INCLUDE_DIR}
)

include(GoogleTest)
gtest_discover_tests(lfcc_pipeline_utils_gtest)
configure_eigen_parallel_target(lfcc_pipeline_utils_gtest)