add_executable(e05_profile_audit_gtest
    e05_profile_audit_gtest.cpp
)

target_include_directories(e05_profile_audit_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(e05_profile_audit_gtest
    PRIVATE
        google_test
        experiment05_lib
        xtensor xtensor-blas
        cnpy-static
)

include(GoogleTest)
gtest_discover_tests(e05_profile_audit_gtest)
configure_xtensor_parallel_target(e05_profile_audit_gtest)

# ── Feature extraction unit tests ────────────────────────────────────────────
add_executable(e05_feature_extraction_gtest
    e05_feature_extraction_gtest.cpp
)

target_include_directories(e05_feature_extraction_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(e05_feature_extraction_gtest
    PRIVATE
        google_test
        experiment05_lib
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(e05_feature_extraction_gtest)
configure_xtensor_parallel_target(e05_feature_extraction_gtest)

# ── Classifier unit tests ────────────────────────────────────────────────────
add_executable(e05_classifiers_gtest
    e05_classifiers_gtest.cpp
)

target_include_directories(e05_classifiers_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(e05_classifiers_gtest
    PRIVATE
        google_test
        experiment05_lib
        xtensor xtensor-blas
        cnpy-static
        OpenMP::OpenMP_CXX
)

gtest_discover_tests(e05_classifiers_gtest)
configure_xtensor_parallel_target(e05_classifiers_gtest)
