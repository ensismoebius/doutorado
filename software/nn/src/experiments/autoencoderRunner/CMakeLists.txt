# AutoencoderRunner shared library and integrated LSTM sources

set(EXPERIMENT03_LIB_SOURCES
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/cli.cpp
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/ProfileLoader.cpp
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/ResultsWriter.cpp
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/RunSummaryBuilder.cpp
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/DatasetBuilder.cpp
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/TrialFoldSelector.cpp
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/autoencoderRunner_helpers.cpp
    ${SRC_DIR}/experiments/autoencoderRunner/lib/src/autoencoderRunner.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    add_library(autoencoderRunner_lib OBJECT ${EXPERIMENT03_LIB_SOURCES})
else()
    add_library(autoencoderRunner_lib STATIC ${EXPERIMENT03_LIB_SOURCES})
endif()

target_include_directories(autoencoderRunner_lib
    PUBLIC
        ${SRC_DIR}/experiments/autoencoderRunner/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(autoencoderRunner_lib
    PUBLIC
        CLI11::CLI11
        util
        nlohmann_json_wrapper
        dataLoaders_10_1117
        dataLoaders_10_1117_windowing
        data_loaders
        matioCpp
        z
        xtensor xtensor-blas
        statistics
        layers
        lstm_models
        autoencoder_models
        nn_progress
        waveCoreLib
    )


# Experiment04 implementation library (used by both autoencoderRunner and guayaquil executables).
add_library(guayaquil_lib STATIC
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilRunner.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilLstmAutoencoder.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilExperiment.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilCli.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilConfig.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilDataset.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilEncoding.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilMetrics.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilEvaluation.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilTraining.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilOutput.cpp
    ${SRC_DIR}/experiments/guayaquil/lib/src/GuayaquilCheckpoint.cpp
)

target_include_directories(guayaquil_lib
    PUBLIC
        ${SRC_DIR}/experiments/guayaquil/lib/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(guayaquil_lib
    PUBLIC
        autoencoderRunner_lib
        data_loaders_fsdd
)

nn_enable_target_pch(autoencoderRunner_lib)

# Enable xtensor parallelization with OpenMP
configure_xtensor_parallel_target(autoencoderRunner_lib)

add_executable(autoencoderRunner
    ${SRC_DIR}/experiments/autoencoderRunner/autoencoderRunner.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    target_sources(autoencoderRunner
        PRIVATE
            $<TARGET_OBJECTS:autoencoderRunner_lib>
    )
endif()

# Link libraries
target_link_libraries(autoencoderRunner
    PRIVATE
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        OpenMP::OpenMP_CXX
        autoencoderRunner_lib
        guayaquil_lib
        xtensor xtensor-blas
        cnpy-static
        data_loaders
        matioCpp
        z
)

# Enable xtensor parallelization with OpenMP
configure_xtensor_parallel_target(autoencoderRunner)

file(GLOB EXPERIMENT03_PROFILE_JSON "${SRC_DIR}/experiments/autoencoderRunner/profiles/*.json")
file(GLOB EXPERIMENT04_PROFILE_JSON "${SRC_DIR}/experiments/guayaquil/profiles/*.json")
set(EXPERIMENT_PROFILE_JSON ${EXPERIMENT03_PROFILE_JSON} ${EXPERIMENT04_PROFILE_JSON})
if(EXPERIMENT_PROFILE_JSON)
    add_custom_command(TARGET autoencoderRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:autoencoderRunner>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${EXPERIMENT_PROFILE_JSON}
            $<TARGET_FILE_DIR:autoencoderRunner>/profiles
        VERBATIM
    )
endif()

add_subdirectory(tests)
