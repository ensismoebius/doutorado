add_executable(experiment_04_comparative_gtest
    ComparativeExperiment_gtest.cpp
)

target_include_directories(experiment_04_comparative_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(experiment_04_comparative_gtest
    PRIVATE
        google_test
        experiment03_lib
        tensor
        layers
        dataLoaders
        util
        Eigen3::Eigen
        statistics
        matioCpp
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        cnpy-static
)

include(GoogleTest)
gtest_discover_tests(experiment_04_comparative_gtest)
configure_eigen_parallel_target(experiment_04_comparative_gtest)
