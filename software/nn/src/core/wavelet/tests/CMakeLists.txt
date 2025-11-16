add_executable(wavelet_gtest 
    wavelet_gtest.cpp
)

target_link_libraries(wavelet_gtest
    PRIVATE 
       GTest::gtest_main
       wavelet
       filtersOperationsLibStatic
       linearAlgebra
)

target_include_directories(wavelet_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(wavelet_gtest)