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
