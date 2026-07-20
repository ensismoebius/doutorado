set(THESIS_LIB_SOURCES
    lib/src/ThesisConfig.cpp
    lib/src/ThesisDataset.cpp
    lib/src/ThesisFeatureExtraction.cpp
    lib/src/ThesisParaconsistent.cpp
    lib/src/ThesisClassifiers.cpp
    lib/src/ThesisOutput.cpp
)

add_library(thesis_lib STATIC ${THESIS_LIB_SOURCES})

target_include_directories(thesis_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/experiments/autoencoderRunner/lib/include  # Protocol (SNN/ANN) autoencoders
)

target_link_libraries(thesis_lib
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
        autoencoderRunner_lib   # ProtocolAutoencoder (ANN-AE) + ProtocolSpikingAutoencoder (SNN-AE)
        nlohmann_json_wrapper
        matioCpp
        cnpy-static
        xtensor xtensor-blas
)

configure_xtensor_parallel_target(thesis_lib)

# ── executable ──────────────────────────────────────────────────────────────

add_executable(thesis
    ${CMAKE_CURRENT_SOURCE_DIR}/thesis.cpp
)

target_include_directories(thesis
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(thesis
    PRIVATE
        thesis_lib
        OpenMP::OpenMP_CXX
        xtensor xtensor-blas
        matioCpp
        cnpy-static
        z
)

configure_xtensor_parallel_target(thesis)

# ── copy profiles to build dir ───────────────────────────────────────────────

file(GLOB THESIS_PROFILE_JSON "${CMAKE_CURRENT_SOURCE_DIR}/profiles/*.json")
if(THESIS_PROFILE_JSON)
    add_custom_command(TARGET thesis POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:thesis>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${THESIS_PROFILE_JSON}
            $<TARGET_FILE_DIR:thesis>/profiles
        VERBATIM
    )
endif()

# ── smoke-test profile mirroring ─────────────────────────────────────────────
# Regenerate profiles/smoke/ whenever any source profile (or the generator)
# changes. thesis_make_smoke_profiles.py is stdlib-only (no torch/numpy), so any
# python3 works. CONFIGURE_DEPENDS re-globs when profiles are added/removed.
# Only the source phase directories are globbed (never profiles/smoke/) so the
# generated mirror is not an input to its own generation.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE NAMES python3 python)
endif()

if(Python3_EXECUTABLE)
    file(GLOB_RECURSE THESIS_SOURCE_PROFILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/profiles/phase00/*.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/profiles/phase01/*.json")
    list(APPEND THESIS_SOURCE_PROFILES "${CMAKE_CURRENT_SOURCE_DIR}/profiles/debug.json")

    set(THESIS_SMOKE_GEN "${CMAKE_SOURCE_DIR}/scripts/testing/thesis_make_smoke_profiles.py")
    set(THESIS_SMOKE_STAMP "${CMAKE_CURRENT_BINARY_DIR}/thesis_smoke_profiles.stamp")

    add_custom_command(
        OUTPUT ${THESIS_SMOKE_STAMP}
        COMMAND ${Python3_EXECUTABLE} ${THESIS_SMOKE_GEN}
        COMMAND ${CMAKE_COMMAND} -E touch ${THESIS_SMOKE_STAMP}
        DEPENDS ${THESIS_SOURCE_PROFILES} ${THESIS_SMOKE_GEN}
        COMMENT "Mirroring Thesis smoke profiles (profiles/smoke/)"
        VERBATIM)

    add_custom_target(thesis_smoke_profiles ALL DEPENDS ${THESIS_SMOKE_STAMP})
else()
    message(WARNING "python3 not found — Thesis smoke profiles will not auto-regenerate")
endif()

# ── phase00 paraconsistent ranking tables (thesis) ───────────────────────────
# Rebuilds the ranked LaTeX longtables in the thesis whenever any phase00
# result JSON changes. NOT part of ALL: results/phase00/*_summary.json only
# exist after an actual phase00 run (scripts/testing/run_thesis_profiles.sh
# phase00), so this must be invoked explicitly:
#   cmake --build <preset-dir> --target e05_phase00_paraconsistent_tables
if(Python3_EXECUTABLE)
    file(GLOB THESIS_PHASE00_RESULTS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/results/phase00/*_summary.json")

    set(THESIS_PHASE00_TABLES_GEN "${CMAKE_SOURCE_DIR}/scripts/pipeline/thesis/thesis_build_phase00_paraconsistent_tables.py")
    set(THESIS_PHASE00_RESULTS_DIR "${CMAKE_SOURCE_DIR}/results/phase00")
    set(THESIS_PHASE00_TABLES_DIR "${CMAKE_SOURCE_DIR}/../../documentation/00-thesis/monography/tables")
    set(THESIS_PHASE00_TABLES_STAMP "${CMAKE_CURRENT_BINARY_DIR}/e05_phase00_paraconsistent_tables.stamp")

    add_custom_command(
        OUTPUT ${THESIS_PHASE00_TABLES_STAMP}
        COMMAND ${Python3_EXECUTABLE} ${THESIS_PHASE00_TABLES_GEN}
            --results-dir ${THESIS_PHASE00_RESULTS_DIR}
            --tables-dir ${THESIS_PHASE00_TABLES_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${THESIS_PHASE00_TABLES_STAMP}
        DEPENDS ${THESIS_PHASE00_RESULTS} ${THESIS_PHASE00_TABLES_GEN}
        COMMENT "Ranking Experiment05 phase00 results by d_truth (thesis LaTeX tables)"
        VERBATIM)

    add_custom_target(e05_phase00_paraconsistent_tables DEPENDS ${THESIS_PHASE00_TABLES_STAMP})
endif()

add_subdirectory(tests)
