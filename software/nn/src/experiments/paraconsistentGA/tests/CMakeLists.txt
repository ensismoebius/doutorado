add_executable(paraconsistent_ga_gtest
    paraconsistent_ga_gtest.cpp
)

target_include_directories(paraconsistent_ga_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${cnpy_SOURCE_DIR}  # reads back the weight-snapshot .npz directly to verify contents
)

target_link_libraries(paraconsistent_ga_gtest
    PRIVATE
        google_test
        paraconsistentGA_lib
        thesis_lib
        xtensor xtensor-blas
        cnpy-static
)

include(GoogleTest)
nn_gtest_discover_tests(paraconsistent_ga_gtest)
configure_xtensor_parallel_target(paraconsistent_ga_gtest)
