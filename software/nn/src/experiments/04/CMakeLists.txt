# Standalone experiment04 executable
# Note: experiment04 reuses LSTM autoencoder and comparative runner components
# that are compiled as part of experiment03_lib. This executable provides an
# independent entry point for running LSTM-AE or comparative experiments.

add_executable(experiment04
    ${SRC_DIR}/experiments/04/experiment04_launcher.cpp
)

target_include_directories(experiment04
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Link libraries - depends on experiment03_lib for LSTM/comparative components
target_link_libraries(experiment04
    PRIVATE
        OpenMP::OpenMP_CXX
        experiment03_lib
        Eigen3::Eigen
        cnpy-static
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        dataLoaders
        matioCpp
        z
)

# Enable Eigen parallelization with OpenMP
configure_eigen_parallel_target(experiment04)

# Copy profile JSON files to build directory
# Note: profiles are already copied by experiment03, but we also copy them
# here for standalone experiment04 builds
file(GLOB EXPERIMENT04_PROFILE_JSON "${SRC_DIR}/experiments/04/profiles/*.json")
if(EXPERIMENT04_PROFILE_JSON)
    add_custom_command(TARGET experiment04 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:experiment04>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${EXPERIMENT04_PROFILE_JSON}
            $<TARGET_FILE_DIR:experiment04>/profiles
        VERBATIM
    )
endif()

add_subdirectory(tests)
