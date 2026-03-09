add_executable(experiment_02_data_gtest
    Experiment02Data_gtest.cpp
    ../Experiment02Data.cpp
)

target_include_directories(experiment_02_data_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(experiment_02_data_gtest
    PRIVATE
        google_test
        waveCoreLib
        tensor
        matioCpp
        dataLoaders
        util
        Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(experiment_02_data_gtest)
configure_eigen_parallel_target(experiment_02_data_gtest)
