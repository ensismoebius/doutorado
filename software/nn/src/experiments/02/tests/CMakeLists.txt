add_executable(experiment_02_data_gtest
    Experiment02Data_gtest.cpp
    ../Experiment02Data.cpp
)

add_executable(experiment_02_wavelets_gtest
    Experiment02Wavelets_gtest.cpp
    ../Experiment02Wavelets.cpp
)

add_executable(experiment_02_reporting_gtest
    Experiment02Reporting_gtest.cpp
    ../Experiment02Reporting.cpp
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
        data_loaders
        util
        xtensor xtensor-blas
)

target_include_directories(experiment_02_wavelets_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(experiment_02_wavelets_gtest
    PRIVATE
        google_test
        wavelet
)

target_include_directories(experiment_02_reporting_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(experiment_02_reporting_gtest
    PRIVATE
        google_test
)

include(GoogleTest)
gtest_discover_tests(experiment_02_data_gtest)
gtest_discover_tests(experiment_02_wavelets_gtest)
gtest_discover_tests(experiment_02_reporting_gtest)
configure_xtensor_parallel_target(experiment_02_data_gtest)
configure_xtensor_parallel_target(experiment_02_wavelets_gtest)
configure_xtensor_parallel_target(experiment_02_reporting_gtest)
