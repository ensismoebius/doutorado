include(../../../cmake/SanitizerFlags.cmake)

add_executable(phase1_wavelet_baseline
    main.cpp
)

target_include_directories(phase1_wavelet_baseline
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(phase1_wavelet_baseline
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
        Eigen3::Eigen
        yaml-cpp
        OpenMP::OpenMP_CXX
)