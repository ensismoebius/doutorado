include(../../../cmake/SanitizerFlags.cmake)

add_executable(experiment_02
    Experiment02Config.cpp
    Experiment02Data.cpp
    Experiment02Evaluation.cpp
    Experiment02Pipeline.cpp
    Experiment02Reporting.cpp
    Experiment02Training.cpp
    Experiment02Wavelets.cpp
    experiment_02.cpp
)

target_link_libraries(experiment_02
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
add_callgrind_target(profile_experiment_02 experiment_02)

add_subdirectory(tests)