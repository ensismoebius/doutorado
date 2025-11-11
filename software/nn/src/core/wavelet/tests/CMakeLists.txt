add_library(waveletTestLibStatic
    STATIC
        ../Types.cpp
        ../waveletOperations.cpp
        ../WaveletTransformResults.cpp
)

target_include_directories(waveletTestLibStatic
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

add_executable(wavelet_gtest 
    wavelet_gtest.cpp
)

target_link_libraries(wavelet_gtest
    PRIVATE GTest::gtest_main
    PRIVATE waveletTestLibStatic
    PRIVATE filtersOperationsLibStatic
    PRIVATE linearAlgebra
)

include(GoogleTest)
gtest_discover_tests(wavelet_gtest)