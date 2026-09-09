# Standalone meeting01 executable.
# The meeting01 implementation lives in meeting01_lib (defined in AutoencoderRunner CMake).

add_executable(meeting01
    ${SRC_DIR}/experiments/meeting01/meeting01.cpp
)

target_include_directories(meeting01
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Link libraries - depends on autoencoderRunner_lib for LSTM/comparative components
target_link_libraries(meeting01
    PRIVATE
        OpenMP::OpenMP_CXX
        meeting01_lib
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
configure_xtensor_parallel_target(meeting01)

# Copy profile JSON files to build directory
# Note: profiles are already copied by autoencoderRunner, but we also copy them
# here for standalone meeting01 builds
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
