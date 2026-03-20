## exec_experiment04.cmake
# Target for experiment04 utility

set(EXPERIMENT03_LIB_SOURCES
    ${SRC_DIR}/experiments/03/lib/src/cli.cpp
    ${SRC_DIR}/experiments/03/lib/src/progress.cpp
    ${SRC_DIR}/experiments/03/lib/src/dataset_info.cpp
    ${SRC_DIR}/experiments/03/lib/src/experiment03.cpp
    # ANN autoencoders
    ${SRC_DIR}/experiments/03/lib/src/ProtocolAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/EegWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/AudioWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/FusedWindowAutoencoder.cpp
    # SNN autoencoders
    ${SRC_DIR}/experiments/03/lib/src/ProtocolSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/EegWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/AudioWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/FusedWindowSpikingAutoencoder.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    add_library(experiment03_lib OBJECT ${EXPERIMENT03_LIB_SOURCES})
else()
    add_library(experiment03_lib STATIC ${EXPERIMENT03_LIB_SOURCES})
endif()

target_include_directories(experiment03_lib
    PUBLIC
        ${SRC_DIR}/experiments/03/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(experiment03_lib
    PUBLIC
        CLI11::CLI11
        dataLoaders_10_1117
        dataLoaders_10_1117_windowing
        dataLoaders
        matioCpp
        z
        Eigen3::Eigen
)

nn_enable_target_pch(experiment03_lib)

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
    experiment03_lib
    Eigen3::Eigen
    cnpy-static
    dataLoaders_10_1117
    dataLoaders_10_1117_windowing
    dataLoaders
    matioCpp
    z
)

# ---------------------------------------------------------------------------
# Autoencoder scaffold example executable
# ---------------------------------------------------------------------------
add_executable(autoencoder_scaffold_example
    ${SRC_DIR}/experiments/03/autoencoder_scaffold_example.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    target_sources(autoencoder_scaffold_example
        PRIVATE
            $<TARGET_OBJECTS:experiment03_lib>
    )
endif()

target_link_libraries(autoencoder_scaffold_example
    PRIVATE
    experiment03_lib
    Eigen3::Eigen
)

add_executable(experiment03_redesign_gtest
    ${SRC_DIR}/experiments/03/tests/AutoencoderRedesign_gtest.cpp
)

target_include_directories(experiment03_redesign_gtest
    PRIVATE
        ${SRC_DIR}/experiments/03/lib/include
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(experiment03_redesign_gtest
    PRIVATE
        experiment03_lib
        google_test
        Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(experiment03_redesign_gtest)
configure_eigen_parallel_target(experiment03_redesign_gtest)
