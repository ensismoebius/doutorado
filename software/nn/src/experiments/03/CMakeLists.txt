# Experiment03 shared library and integrated LSTM sources

set(EXPERIMENT03_LIB_SOURCES
    ${SRC_DIR}/experiments/03/lib/src/cli.cpp
    ${SRC_DIR}/experiments/04/lib/src/LSTMAutoencoder.cpp
    ${SRC_DIR}/experiments/04/lib/src/comparative_experiment.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/AudioWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/AudioWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/EegWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/EegWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/FusedWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/FusedWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/ProtocolAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/autoencoder/ProtocolSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/ProfileLoader.cpp
    ${SRC_DIR}/experiments/03/lib/src/ResultsWriter.cpp
    ${SRC_DIR}/experiments/03/lib/src/RunSummaryBuilder.cpp
    ${SRC_DIR}/experiments/03/lib/src/DatasetBuilder.cpp
    ${SRC_DIR}/experiments/03/lib/src/TrialFoldSelector.cpp
    ${SRC_DIR}/experiments/03/lib/src/experiment03_helpers.cpp
    ${SRC_DIR}/experiments/03/lib/src/experiment03.cpp
    ${SRC_DIR}/experiments/04/lib/src/experiment04.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    add_library(experiment03_lib OBJECT ${EXPERIMENT03_LIB_SOURCES})
else()
    add_library(experiment03_lib STATIC ${EXPERIMENT03_LIB_SOURCES})
endif()

target_include_directories(experiment03_lib
    PUBLIC
        ${SRC_DIR}/experiments/03/lib/include
        ${SRC_DIR}/experiments/03/lib/include/autoencoder
        ${SRC_DIR}/experiments/04/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(experiment03_lib
    PUBLIC
        CLI11::CLI11
        util
        nlohmann_json_wrapper
        dataLoaders_10_1117
        dataLoaders_10_1117_windowing
        dataLoaders
        matioCpp
        z
        Eigen3::Eigen
        statistics
)

nn_enable_target_pch(experiment03_lib)

# Enable Eigen parallelization with OpenMP
configure_eigen_parallel_target(experiment03_lib)

add_executable(experiment03
    ${SRC_DIR}/experiments/03/experiment03.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    target_sources(experiment03
        PRIVATE
            $<TARGET_OBJECTS:experiment03_lib>
    )
endif()

# Link libraries
target_link_libraries(experiment03
    PRIVATE
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        OpenMP::OpenMP_CXX
        experiment03_lib
        Eigen3::Eigen
        cnpy-static
        dataLoaders
        matioCpp
        z
)

# Enable Eigen parallelization with OpenMP
configure_eigen_parallel_target(experiment03)

file(GLOB EXPERIMENT03_PROFILE_JSON "${SRC_DIR}/experiments/03/profiles/*.json")
file(GLOB EXPERIMENT04_PROFILE_JSON "${SRC_DIR}/experiments/04/profiles/*.json")
set(EXPERIMENT_PROFILE_JSON ${EXPERIMENT03_PROFILE_JSON} ${EXPERIMENT04_PROFILE_JSON})
if(EXPERIMENT_PROFILE_JSON)
    add_custom_command(TARGET experiment03 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:experiment03>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${EXPERIMENT_PROFILE_JSON}
            $<TARGET_FILE_DIR:experiment03>/profiles
        VERBATIM
    )
endif()

add_subdirectory(tests)
