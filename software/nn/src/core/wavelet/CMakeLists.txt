add_library(wavelet
    STATIC
        Types.cpp
        waveletOperations.cpp
        WaveletTransformResults.cpp
)

target_include_directories(wavelet
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(wavelet PRIVATE linearAlgebra)

add_subdirectory(tests)
