include(../../../cmake/SanitizerFlags.cmake)

add_executable(paraconsistentBaseline
    phase00.cpp
    phase00_data.cpp
    phase00_features.cpp
    phase00_training.cpp
)

target_include_directories(paraconsistentBaseline
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(paraconsistentBaseline
    PRIVATE
        experiments_config
        wavelet
        statistics
        paraconsistent
        layers
        tensor
        matioCpp
        optimizers
        data_loaders
        dataLoaders_10_1117  # Added to link the 10.1117 specific data loaders
        xtensor xtensor-blas
        nlohmann_json_wrapper
        OpenMP::OpenMP_CXX
)