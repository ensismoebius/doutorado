set(PGA_LIB_SOURCES
    lib/src/GaConfig.cpp
    lib/src/GaGenome.cpp
    lib/src/GaFitness.cpp
    lib/src/GaNsga2.cpp
    lib/src/GaOutput.cpp
    lib/src/GaCheckpoint.cpp
    lib/src/GaModelSnapshot.cpp
)

add_library(paraconsistentGA_lib STATIC ${PGA_LIB_SOURCES})

target_include_directories(paraconsistentGA_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/experiments/thesis/lib/include  # ThesisConfig / extract_features / score_feature_set
    PRIVATE
        ${cnpy_SOURCE_DIR}  # GaModelSnapshot.cpp: <cnpy.h> for the weight-snapshot .npz writer
)

target_link_libraries(paraconsistentGA_lib
    PUBLIC
        thesis_lib          # reused: dataset, AE feature extraction, paraconsistent scoring, output
        nlohmann_json_wrapper
        nn_progress
        xtensor xtensor-blas
)

configure_xtensor_parallel_target(paraconsistentGA_lib)

# ── executable ──────────────────────────────────────────────────────────────

add_executable(paraconsistentGA
    ${CMAKE_CURRENT_SOURCE_DIR}/paraconsistentGA.cpp
)

target_include_directories(paraconsistentGA
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(paraconsistentGA
    PRIVATE
        paraconsistentGA_lib
        OpenMP::OpenMP_CXX
        xtensor xtensor-blas
        matioCpp
        cnpy-static
        z
)

configure_xtensor_parallel_target(paraconsistentGA)

# ── copy profiles to build dir ───────────────────────────────────────────────

file(GLOB PGA_PROFILE_JSON "${CMAKE_CURRENT_SOURCE_DIR}/profiles/*.json")
if(PGA_PROFILE_JSON)
    add_custom_command(TARGET paraconsistentGA POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:paraconsistentGA>/profiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PGA_PROFILE_JSON}
            $<TARGET_FILE_DIR:paraconsistentGA>/profiles
        VERBATIM
    )
endif()

add_subdirectory(tests)
