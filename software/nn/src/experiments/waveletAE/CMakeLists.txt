include(../../../cmake/SanitizerFlags.cmake)

add_executable(waveletAE
    WaveletAEConfig.cpp
    WaveletAEData.cpp
    WaveletAEEvaluation.cpp
    WaveletAEPipeline.cpp
    WaveletAEReporting.cpp
    WaveletAETraining.cpp
    WaveletAEWavelets.cpp
    waveletAE.cpp
)

target_link_libraries(waveletAE
    PRIVATE
        experiments_config
        wavelet
        statistics
        paraconsistent
        layers
        tensor
        util
        matioCpp
        optimizers
        data_loaders
        linear_algebra
        waveCoreLib
        xtensor xtensor-blas
        nlohmann_json_wrapper
        OpenMP::OpenMP_CXX
)

# Add Callgrind profiling target for performance analysis
include(../../../cmake/Tooling.cmake)
add_callgrind_target(profile_experiment_02 waveletAE)

add_subdirectory(tests)