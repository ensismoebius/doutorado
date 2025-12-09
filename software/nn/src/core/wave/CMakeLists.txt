add_library(waveCoreLib STATIC
    filtersOperations.cpp
    simpleSignalOperations.cpp
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
    PRIVATE Eigen3::Eigen FFTW::FFTW
)

add_subdirectory(tests)