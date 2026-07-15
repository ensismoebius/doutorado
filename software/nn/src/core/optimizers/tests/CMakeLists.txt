##
## Optimizer unit tests (GoogleTest)
##
## Builds:
## - `optimizers_gtest`: tests for optimizer implementations (Adam/SGD variants).
##
## Notes:
## - Linked against `optimizers` + `tensor` and uses xtensor/OpenMP via
##   `configure_xtensor_parallel_target`.
##
add_executable(optimizers_gtest 
    optimizers_gtest.cpp
    state_io_gtest.cpp
)

target_link_libraries(optimizers_gtest 
    PRIVATE 
        google_test 
         xtensor xtensor-blas 
        optimizers
        tensor
)

target_include_directories(optimizers_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
nn_gtest_discover_tests(optimizers_gtest)
configure_xtensor_parallel_target(optimizers_gtest)
