include(../../../cmake/SanitizerFlags.cmake)

add_executable(experiment_02
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
        dataLoaders
        linearAlgebra
        waveCoreLib
        Eigen3::Eigen
        yaml-cpp
        OpenMP::OpenMP_CXX
)