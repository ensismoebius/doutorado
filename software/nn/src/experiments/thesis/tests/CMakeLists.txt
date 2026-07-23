add_executable(thesis_profile_audit_gtest
    thesis_profile_audit_gtest.cpp
)

target_include_directories(thesis_profile_audit_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(thesis_profile_audit_gtest
    PRIVATE
        google_test
        thesis_lib
        xtensor xtensor-blas
        cnpy-static
)

include(GoogleTest)
nn_gtest_discover_tests(thesis_profile_audit_gtest)
configure_xtensor_parallel_target(thesis_profile_audit_gtest)

# ── Feature extraction unit tests ────────────────────────────────────────────
add_executable(thesis_feature_extraction_gtest
    thesis_feature_extraction_gtest.cpp
)

target_include_directories(thesis_feature_extraction_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(thesis_feature_extraction_gtest
    PRIVATE
        google_test
        thesis_lib
        xtensor xtensor-blas
        cnpy-static
)

nn_gtest_discover_tests(thesis_feature_extraction_gtest)
configure_xtensor_parallel_target(thesis_feature_extraction_gtest)

# ── Classifier unit tests ────────────────────────────────────────────────────
add_executable(thesis_classifiers_gtest
    thesis_classifiers_gtest.cpp
)

target_include_directories(thesis_classifiers_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(thesis_classifiers_gtest
    PRIVATE
        google_test
        thesis_lib
        xtensor xtensor-blas
        cnpy-static
        OpenMP::OpenMP_CXX
)

nn_gtest_discover_tests(thesis_classifiers_gtest)
configure_xtensor_parallel_target(thesis_classifiers_gtest)

# ── Output-artifact contract tests ──────────────────────────────────────────
add_executable(thesis_output_gtest
    thesis_output_gtest.cpp
)

target_include_directories(thesis_output_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(thesis_output_gtest
    PRIVATE
        google_test
        thesis_lib
        xtensor xtensor-blas
        cnpy-static
        OpenMP::OpenMP_CXX
)

nn_gtest_discover_tests(thesis_output_gtest)
configure_xtensor_parallel_target(thesis_output_gtest)

# ── Spike-loss end-to-end smoke (SpikeCount / SpikeTime actually train) ──────
add_executable(thesis_spike_loss_e2e_gtest
    thesis_spike_loss_e2e_gtest.cpp
)

target_include_directories(thesis_spike_loss_e2e_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(thesis_spike_loss_e2e_gtest
    PRIVATE
        google_test
        thesis_lib
        xtensor xtensor-blas
        cnpy-static
        OpenMP::OpenMP_CXX
)

nn_gtest_discover_tests(thesis_spike_loss_e2e_gtest)
configure_xtensor_parallel_target(thesis_spike_loss_e2e_gtest)
