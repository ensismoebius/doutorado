# CMakeLists.txt for experiment04
# LSTM Autoencoder for 1-D temporal signals
#
# Build targets:
#   experiment04_lib   — reusable static/object library containing the LSTM
#                        implementation and trainer (mirrors experiment03_lib).
#   experiment04       — thin executable launcher.
#
# Dependencies (all already available from parent CMake context):
#   - Eigen3::Eigen     (tensor backend)
#   - nlohmann_json     (results serialisation)
#   - util              (progress / batching helpers)
#   - statistics        (fold splitters, if needed later)
#
# Optional (linked if available, experiment still compiles without them):
#   - dataLoaders_10_1117          (real windowed dataset)
#   - dataLoaders_10_1117_windowing

set(EXPERIMENT04_LIB_SOURCES
    ${SRC_DIR}/experiments/04/lib/src/LSTMAutoencoder.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    add_library(experiment04_lib OBJECT ${EXPERIMENT04_LIB_SOURCES})
else()
    add_library(experiment04_lib STATIC ${EXPERIMENT04_LIB_SOURCES})
endif()

target_include_directories(experiment04_lib
    PUBLIC
        ${SRC_DIR}/experiments/04/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(experiment04_lib
    PUBLIC
        Eigen3::Eigen
        util
        nlohmann_json_wrapper
)

nn_enable_target_pch(experiment04_lib)

# Enable Eigen parallelisation with OpenMP if available
if(TARGET configure_eigen_parallel_target)
    configure_eigen_parallel_target(experiment04_lib)
endif()

# ---- Executable ----
add_executable(experiment04
    ${SRC_DIR}/experiments/04/experiment04.cpp
)

if(NN_USE_OBJECT_LIBRARIES)
    target_sources(experiment04
        PRIVATE
            $<TARGET_OBJECTS:experiment04_lib>
    )
endif()

target_link_libraries(experiment04
    PRIVATE
        experiment04_lib
        Eigen3::Eigen
        nlohmann_json_wrapper
        util
)

# Optionally link real dataset loaders when present
if(TARGET dataLoaders_10_1117)
    target_link_libraries(experiment04 PRIVATE dataLoaders_10_1117)
    target_compile_definitions(experiment04 PRIVATE EXPERIMENT04_HAS_10_1117_LOADERS)
endif()
if(TARGET dataLoaders_10_1117_windowing)
    target_link_libraries(experiment04 PRIVATE dataLoaders_10_1117_windowing)
endif()

# ---- Tests ----
if(EXISTS ${SRC_DIR}/experiments/04/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()

# ---- Profile copy (for future JSON profiles) ----
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
