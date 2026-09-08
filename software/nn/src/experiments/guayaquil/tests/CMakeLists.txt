add_executable(guayaquil_comparative_gtest
    ComparativeExperiment_gtest.cpp
)

target_include_directories(guayaquil_comparative_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(guayaquil_comparative_gtest
    PRIVATE
        google_test
        autoencoderRunner_lib
        tensor
        layers
        data_loaders
        util
        xtensor xtensor-blas
        statistics
        matioCpp
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        cnpy-static
)

include(GoogleTest)
nn_gtest_discover_tests(guayaquil_comparative_gtest)
configure_xtensor_parallel_target(guayaquil_comparative_gtest)

# DAT writer tests
add_executable(dat_writer_gtest
    dat_writer_gtest.cpp
)

target_include_directories(dat_writer_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dat_writer_gtest
    PRIVATE
        google_test
        guayaquil_lib
        autoencoderRunner_lib
        tensor
        layers
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(dat_writer_gtest)
configure_xtensor_parallel_target(dat_writer_gtest)

# Checkpoint round-trip / resilience tests
add_executable(checkpoint_gtest
    checkpoint_gtest.cpp
)

target_include_directories(checkpoint_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(checkpoint_gtest
    PRIVATE
        google_test
        guayaquil_lib
        autoencoderRunner_lib
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(checkpoint_gtest)
configure_xtensor_parallel_target(checkpoint_gtest)

# Profile audit: every shipping JSON profile must parse, validate, and
# populate every config field the harness consumes (catches silent drift).
add_executable(profile_audit_gtest
    profile_audit_gtest.cpp
)

target_include_directories(profile_audit_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(profile_audit_gtest
    PRIVATE
        google_test
        guayaquil_lib
        autoencoderRunner_lib
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(profile_audit_gtest)
configure_xtensor_parallel_target(profile_audit_gtest)

# Grouped-window dataset sources (FSDD / AudioMNIST / MIT-BIH): fast real-loader checks.
add_executable(loaders_gtest
    loaders_gtest.cpp
)

target_include_directories(loaders_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/src/experiments/guayaquil/lib/src
)

target_link_libraries(loaders_gtest
    PRIVATE
        google_test
        guayaquil_lib
        autoencoderRunner_lib
        data_loaders_fsdd
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(loaders_gtest)
configure_xtensor_parallel_target(loaders_gtest)

# Structured JSONL event sink for the live monitor (schema helpers + round-trip).
add_executable(event_sink_gtest
    event_sink_gtest.cpp
)

target_include_directories(event_sink_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/src/experiments/guayaquil/lib/src
)

target_link_libraries(event_sink_gtest
    PRIVATE
        google_test
        guayaquil_lib
        autoencoderRunner_lib
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(event_sink_gtest)
configure_xtensor_parallel_target(event_sink_gtest)

# Parameter-count audit: documented P for LSTM-/GRU-/Transformer-AE at the article config.
add_executable(parameter_count_gtest
    parameter_count_gtest.cpp
)

target_include_directories(parameter_count_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(parameter_count_gtest
    PRIVATE
        google_test
        guayaquil_lib
        autoencoderRunner_lib
        gru_models
        transformer_models
        tensor
        layers
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(parameter_count_gtest)
configure_xtensor_parallel_target(parameter_count_gtest)

# Split audit: nested LOSO fold assignment is speaker/recording/window disjoint.
add_executable(guayaquil_split_audit_gtest
    split_audit_gtest.cpp
)

target_include_directories(guayaquil_split_audit_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(guayaquil_split_audit_gtest
    PRIVATE
        google_test
        guayaquil_lib
        autoencoderRunner_lib
        tensor
        layers
        data_loaders
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(guayaquil_split_audit_gtest)
configure_xtensor_parallel_target(guayaquil_split_audit_gtest)

# LSTM microbench (ops breakdown)
add_executable(lstm_ops_microbench
    lstm_ops_microbench.cpp
)

target_include_directories(lstm_ops_microbench
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(lstm_ops_microbench
    PRIVATE
        tensor
        layers
        xtensor xtensor-blas
)

configure_xtensor_parallel_target(lstm_ops_microbench)
