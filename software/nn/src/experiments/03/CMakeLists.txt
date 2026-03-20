## exec_experiment04.cmake
# Target for experiment04 utility

set(EXPERIMENT03_LIB_SOURCES
    ${SRC_DIR}/experiments/03/lib/src/cli.cpp
    ${SRC_DIR}/experiments/03/lib/src/AudioWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/AudioWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/EegWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/EegWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/FusedWindowAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/FusedWindowSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/ProtocolAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/ProtocolSpikingAutoencoder.cpp
    ${SRC_DIR}/experiments/03/lib/src/experiment03.cpp
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

add_subdirectory(tests)
