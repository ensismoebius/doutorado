add_library(waveCoreLib STATIC
    filter_operations.cpp
    signal_operations.cpp
    Wav.cpp
    audioFeatureExtraction.cpp
    lfcc_pipeline_utils.cpp
)

target_include_directories(waveCoreLib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(waveCoreLib
    PUBLIC xtensor xtensor-blas FFTW::FFTW FFTW::FFTWF tensor dataLoaders_10_1117
)

nn_enable_target_pch(waveCoreLib)

add_subdirectory(tests)
