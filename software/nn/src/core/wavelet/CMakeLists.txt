add_library(wavelet
    STATIC
        Types.cpp
        waveletOperations.cpp
        WaveletTransformResults.cpp
)

target_include_directories(wavelet
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)
