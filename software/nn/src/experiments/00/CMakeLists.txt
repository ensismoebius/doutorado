include(../../../cmake/SanitizerFlags.cmake)

add_executable(M0_wavelet_baseline
    phase00.cpp
)

target_include_directories(M0_wavelet_baseline
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(M0_wavelet_baseline
    PRIVATE
        experiments_config
        wavelet
        statistics
        paraconsistent
        layers
        tensor
        matioCpp
        optimizers
        dataLoaders
        dataLoaders_10_1117  # Added to link the 10.1117 specific data loaders
        Eigen3::Eigen
        yaml-cpp
        OpenMP::OpenMP_CXX
)