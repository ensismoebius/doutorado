##
## Wavelet unit tests (GoogleTest)
##
## Builds:
## - `wavelet_gtest`: tests for wavelet traits, transforms, and result extraction.
##
add_executable(wavelet_gtest 
    wavelet_gtest.cpp
)

target_link_libraries(wavelet_gtest
    PRIVATE 
       google_test
       wavelet
       waveCoreLib
       linear_algebra
)

target_include_directories(wavelet_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
nn_gtest_discover_tests(wavelet_gtest)