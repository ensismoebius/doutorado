add_library(waveCoreLib STATIC
    filter_operations.cpp
    signal_operations.cpp
    Wav.cpp
    audioFeatureExtraction.cpp
)

target_include_directories(waveCoreLib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(waveCoreLib
    PUBLIC Eigen3::Eigen FFTW::FFTW fftw3f tensor
)

add_subdirectory(tests)