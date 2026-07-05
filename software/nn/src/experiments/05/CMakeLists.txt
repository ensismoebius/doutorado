set(E05_LIB_SOURCES
    lib/src/E05Config.cpp
    lib/src/E05Dataset.cpp
    lib/src/E05FeatureExtraction.cpp
    lib/src/E05Paraconsistent.cpp
    lib/src/E05Classifiers.cpp
    lib/src/E05Output.cpp
)

add_library(experiment05_lib STATIC ${E05_LIB_SOURCES})

target_include_directories(experiment05_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/experiments/03/lib/include  # Protocol (SNN/ANN) autoencoders
)

target_link_libraries(experiment05_lib
    PUBLIC
        tensor
        layers
        data_loaders
        dataLoaders_10_1117
        dataLoaders_10_1117_windowing
        wavelet
        waveCoreLib
        paraconsistent
        statistics
        util
        optimizers
        serialization
        nn_progress
        lstm_models
        experiment03_lib   # ProtocolAutoencoder (ANN-AE) + ProtocolSpikingAutoencoder (SNN-AE)
        nlohmann_json_wrapper
        matioCpp
        cnpy-static
        xtensor xtensor-blas
)

configure_xtensor_parallel_target(experiment05_lib)

# ── executable ──────────────────────────────────────────────────────────────

add_executable(experiment05
    ${CMAKE_CURRENT_SOURCE_DIR}/experiment05.cpp
)

target_include_directories(experiment05
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(experiment05
    PRIVATE
        experiment05_lib
        OpenMP::OpenMP_CXX
        xtensor xtensor-blas
        matioCpp
        cnpy-static
        z
)

configure_xtensor_parallel_target(experiment05)

# ── copy profiles to build dir ───────────────────────────────────────────────

file(GLOB E05_PROFILE_JSON "${CMAKE_CURRENT_SOURCE_DIR}/profiles/*.json")
if(E05_PROFILE_JSON)
    add_custom_command(TARGET experiment05 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:experiment05>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${E05_PROFILE_JSON}
            $<TARGET_FILE_DIR:experiment05>/profiles
        VERBATIM
    )
endif()

# ── smoke-test profile mirroring ─────────────────────────────────────────────
# Regenerate profiles/smoke/ whenever any source profile (or the generator)
# changes. make_smoke_profiles.py is stdlib-only (no torch/numpy), so any
# python3 works. CONFIGURE_DEPENDS re-globs when profiles are added/removed.
# Only the source phase directories are globbed (never profiles/smoke/) so the
# generated mirror is not an input to its own generation.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE NAMES python3 python)
endif()

if(Python3_EXECUTABLE)
    file(GLOB_RECURSE E05_SOURCE_PROFILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/profiles/phase00/*.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/profiles/phase01/*.json")
    list(APPEND E05_SOURCE_PROFILES "${CMAKE_CURRENT_SOURCE_DIR}/profiles/debug.json")

    set(E05_SMOKE_GEN "${CMAKE_SOURCE_DIR}/scripts/testing/make_smoke_profiles.py")
    set(E05_SMOKE_STAMP "${CMAKE_CURRENT_BINARY_DIR}/e05_smoke_profiles.stamp")

    add_custom_command(
        OUTPUT ${E05_SMOKE_STAMP}
        COMMAND ${Python3_EXECUTABLE} ${E05_SMOKE_GEN}
        COMMAND ${CMAKE_COMMAND} -E touch ${E05_SMOKE_STAMP}
        DEPENDS ${E05_SOURCE_PROFILES} ${E05_SMOKE_GEN}
        COMMENT "Mirroring Experiment05 smoke profiles (profiles/smoke/)"
        VERBATIM)

    add_custom_target(e05_smoke_profiles ALL DEPENDS ${E05_SMOKE_STAMP})
else()
    message(WARNING "python3 not found — Experiment05 smoke profiles will not auto-regenerate")
endif()

add_subdirectory(tests)
