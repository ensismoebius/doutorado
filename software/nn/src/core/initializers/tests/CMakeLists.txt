##
## Initializer unit tests (GoogleTest)
##
## Builds:
## - `initializers_gtest`: tests for weight initialization helpers.
##
## Notes:
## - Linked against `initializers` + `tensor` and uses xtensor/OpenMP settings.
##
add_executable(
    initializers_gtest 
        initializers_gtest.cpp
)
target_link_libraries(initializers_gtest 
    PRIVATE 
        google_test 
         xtensor xtensor-blas 
        initializers
        tensor
)

target_include_directories(initializers_gtest 
    PRIVATE 
)
include(GoogleTest)
nn_gtest_discover_tests(initializers_gtest)
configure_xtensor_parallel_target(initializers_gtest)