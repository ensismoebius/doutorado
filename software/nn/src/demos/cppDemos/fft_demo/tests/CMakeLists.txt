##
## FFT demo unit tests (GoogleTest)
##
if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

add_executable(fft_demo_gtest
    fft_demo_gtest.cpp
)

target_link_libraries(fft_demo_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
        util
        FFTW::FFTW
)

target_include_directories(fft_demo_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${LIB_DIR}/util
        ${OpenMP_INCLUDE_DIRS}
)

configure_xtensor_parallel_target(fft_demo_gtest)

include(GoogleTest)
nn_gtest_discover_tests(fft_demo_gtest)
