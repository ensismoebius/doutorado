## exec_experiment04.cmake
# Target for experiment04 utility

set(EXPERIMENT03_LIB_SOURCES
    ${SRC_DIR}/experiments/03/lib/src/cli.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/AudioWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/AudioWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/EegWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/EegWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/FusedWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/FusedWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/ProtocolAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/autoencoder/src/ProtocolSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/ProfileLoader.cpp
    ${SRC_DIR}/experiments/03/lib/src/ResultsWriter.cpp
    ${SRC_DIR}/experiments/03/lib/src/experiment03.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    add_library(experiment03_lib OBJECT ${EXPERIMENT03_LIB_SOURCES})
else()
    add_library(experiment03_lib STATIC ${EXPERIMENT03_LIB_SOURCES})
endif()

target_include_directories(experiment03_lib
    PUBLIC
        ${SRC_DIR}/experiments/03/lib/autoencoder/include
        ${SRC_DIR}/experiments/03/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(experiment03_lib
    PUBLIC
        CLI11::CLI11
        util
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

file(GLOB EXPERIMENT03_PROFILE_JSON "${SRC_DIR}/experiments/03/profiles/*.json")
if(EXPERIMENT03_PROFILE_JSON)
    add_custom_command(TARGET experiment03 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:experiment03>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${EXPERIMENT03_PROFILE_JSON}
            $<TARGET_FILE_DIR:experiment03>/profiles
        VERBATIM
    )
endif()

add_subdirectory(tests)
