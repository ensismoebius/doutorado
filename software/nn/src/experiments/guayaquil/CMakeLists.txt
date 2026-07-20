# Standalone guayaquil executable.
# The guayaquil implementation lives in guayaquil_lib (defined in Experiment03 CMake).

add_executable(guayaquil
    ${SRC_DIR}/experiments/guayaquil/guayaquil.cpp
)

target_include_directories(guayaquil
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Link libraries - depends on autoencoderRunner_lib for LSTM/comparative components
target_link_libraries(guayaquil
    PRIVATE
        OpenMP::OpenMP_CXX
        guayaquil_lib
        autoencoderRunner_lib
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
configure_xtensor_parallel_target(guayaquil)

# Copy profile JSON files to build directory
# Note: profiles are already copied by autoencoderRunner, but we also copy them
# here for standalone guayaquil builds
file(GLOB EXPERIMENT04_PROFILE_JSON "${SRC_DIR}/experiments/guayaquil/profiles/*.json")
if(EXPERIMENT04_PROFILE_JSON)
    add_custom_command(TARGET guayaquil POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:guayaquil>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${EXPERIMENT04_PROFILE_JSON}
            $<TARGET_FILE_DIR:guayaquil>/profiles
        VERBATIM
    )
endif()

add_subdirectory(tests)
