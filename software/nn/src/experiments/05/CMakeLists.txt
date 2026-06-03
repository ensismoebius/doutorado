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

add_subdirectory(tests)
