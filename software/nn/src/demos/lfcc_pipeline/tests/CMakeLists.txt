##
## LFCC pipeline unit tests (GoogleTest)
##
## Builds:
## - `lfcc_pipeline_utils_gtest`: tests for feature extraction helpers used by the LFCC demo.
##
## Notes:
## - This links both core libs (tensor/dataLoaders/util/waveCoreLib) and the demo-specific
##   helper library `lfcc_pipeline_utils_lib`.
##
add_executable(lfcc_pipeline_utils_gtest lfcc_pipeline_utils_gtest.cpp)

target_link_libraries(lfcc_pipeline_utils_gtest
    PRIVATE
        google_test
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