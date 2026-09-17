##
## Wavelet demo unit tests (GoogleTest)
##
if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

add_executable(wavelet_demo_gtest
    wavelet_demo_gtest.cpp
)

target_link_libraries(wavelet_demo_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
        wavelet
        util
        linear_algebra
)

target_include_directories(wavelet_demo_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${LIB_DIR}/util
        ${OpenMP_INCLUDE_DIRS}
)

configure_xtensor_parallel_target(wavelet_demo_gtest)

include(GoogleTest)
gtest_discover_tests(wavelet_demo_gtest)
