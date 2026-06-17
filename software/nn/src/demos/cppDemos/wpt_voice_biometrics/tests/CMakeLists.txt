##
## WPT voice biometrics unit tests (GoogleTest)
## Functions replicated from main.cpp (no external linkage there).
##
find_package(GTest REQUIRED)

add_executable(wpt_voice_biometrics_gtest
    wpt_voice_biometrics_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../snn_speaker_demo/codificacao.cpp
)

target_link_libraries(wpt_voice_biometrics_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
        wavelet
        waveCoreLib
        util
        linear_algebra
)

target_include_directories(wpt_voice_biometrics_gtest
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../snn_speaker_demo
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${LIB_DIR}/util
        ${OpenMP_INCLUDE_DIRS}
)

configure_xtensor_parallel_target(wpt_voice_biometrics_gtest)

include(GoogleTest)
gtest_discover_tests(wpt_voice_biometrics_gtest)
