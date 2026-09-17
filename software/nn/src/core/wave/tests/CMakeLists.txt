##
## Wave (audio) unit tests (GoogleTest)
##
## Builds:
## - `wave_gtest`: tests for WAV I/O and basic DSP helpers.
##
add_executable(wave_gtest 
    wave_gtest.cpp
)

target_link_libraries(wave_gtest 
    PRIVATE 
       google_test
       waveCoreLib
       linear_algebra
)

target_include_directories(wave_gtest 
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(wave_gtest)
