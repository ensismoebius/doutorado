add_executable(waveletAE_data_gtest
    WaveletAEData_gtest.cpp
    ../WaveletAEData.cpp
)

add_executable(waveletAE_wavelets_gtest
    WaveletAEWavelets_gtest.cpp
    ../WaveletAEWavelets.cpp
)

add_executable(waveletAE_reporting_gtest
    WaveletAEReporting_gtest.cpp
    ../WaveletAEReporting.cpp
)

target_include_directories(waveletAE_data_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(waveletAE_data_gtest
    PRIVATE
        google_test
        waveCoreLib
        tensor
        matioCpp
        data_loaders
        util
        xtensor xtensor-blas
)

target_include_directories(waveletAE_wavelets_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(waveletAE_wavelets_gtest
    PRIVATE
        google_test
        wavelet
)

target_include_directories(waveletAE_reporting_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(waveletAE_reporting_gtest
    PRIVATE
        google_test
)

include(GoogleTest)
nn_gtest_discover_tests(waveletAE_data_gtest)
nn_gtest_discover_tests(waveletAE_wavelets_gtest)
nn_gtest_discover_tests(waveletAE_reporting_gtest)
configure_xtensor_parallel_target(waveletAE_data_gtest)
configure_xtensor_parallel_target(waveletAE_wavelets_gtest)
configure_xtensor_parallel_target(waveletAE_reporting_gtest)
