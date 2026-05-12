add_library(wavelet
    STATIC
        waveletOperations.cpp
        WaveletTransformResults.cpp
)

target_include_directories(wavelet
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(wavelet PRIVATE linear_algebra)

add_subdirectory(tests)
