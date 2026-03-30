##
## Optimizer unit tests (GoogleTest)
##
## Builds:
## - `optimizers_gtest`: tests for optimizer implementations (Adam/SGD variants).
##
## Notes:
## - Linked against `optimizers` + `tensor` and uses Eigen/OpenMP via
##   `configure_eigen_parallel_target`.
##
add_executable(optimizers_gtest 
    optimizers_gtest.cpp
    state_io_gtest.cpp
)

target_link_libraries(optimizers_gtest 
    PRIVATE 
        google_test 
        Eigen3::Eigen 
        optimizers
        tensor
)

target_include_directories(optimizers_gtest 
    PRIVATE 
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(optimizers_gtest)
configure_eigen_parallel_target(optimizers_gtest)
