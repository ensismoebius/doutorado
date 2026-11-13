# meeting01: the conference paper's experiment.
# meeting01_lib holds the implementation; the thesis links it too (Meeting01Energy).

add_library(meeting01_lib STATIC
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Runner.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01LstmAutoencoder.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Experiment.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Cli.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Config.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Dataset.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01MitBih.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Eeg.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Encoding.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Metrics.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Energy.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Evaluation.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Training.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Output.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01Events.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01GaGenome.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01GaFitness.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01GaCheckpoint.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01GaSearch.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01RecurrentGaGenome.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01RecurrentGaFitness.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01TransformerGaGenome.cpp
    ${SRC_DIR}/experiments/meeting01/lib/src/Meeting01TransformerGaFitness.cpp
)

target_include_directories(meeting01_lib
    PUBLIC
        ${SRC_DIR}/experiments/meeting01/lib/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(meeting01_lib
    PUBLIC
        CLI11::CLI11
        util
        nlohmann_json_wrapper
        dataLoaders_10_1117
        dataLoaders_10_1117_windowing
        data_loaders
        data_loaders_fsdd
        matioCpp
        z
        xtensor xtensor-blas
        statistics
        layers
        lstm_models
        gru_models
        transformer_models
        autoencoder_models
        nn_progress
        waveCoreLib
        cnpy
)

# The meeting01 executable.

add_executable(meeting01
    ${SRC_DIR}/experiments/meeting01/meeting01.cpp
)

target_include_directories(meeting01
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(meeting01
    PRIVATE
        OpenMP::OpenMP_CXX
        meeting01_lib
        xtensor xtensor-blas
        cnpy-static
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        data_loaders
        data_loaders_fsdd
        matioCpp
        z
)

# Enable xtensor parallelization with OpenMP
configure_xtensor_parallel_target(meeting01)

# Copy profile JSON files to build directory
file(GLOB EXPERIMENT04_PROFILE_JSON "${SRC_DIR}/experiments/meeting01/profiles/*.json")
if(EXPERIMENT04_PROFILE_JSON)
    add_custom_command(TARGET meeting01 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:meeting01>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${EXPERIMENT04_PROFILE_JSON}
            $<TARGET_FILE_DIR:meeting01>/profiles
        VERBATIM
    )
endif()

add_subdirectory(tests)
