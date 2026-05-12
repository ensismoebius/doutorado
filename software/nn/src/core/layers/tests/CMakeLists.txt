##
## Core layers unit tests (GoogleTest)
##
## Builds:
## - `layers_gtest`: broad coverage across dense layers and SNN components.
##
## Notes:
## - clang-tidy is disabled for this binary (test convenience; avoids noisy
##   diagnostics in third-party include stacks).
##
add_executable(layers_gtest
    layers_gtest.cpp
)

target_link_libraries(layers_gtest 
    PRIVATE 
        google_test 
         xtensor xtensor-blas 
        layers
        tensor
)

target_include_directories(
  layers_gtest 
    PRIVATE 
         ${cnpy_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(layers_gtest PROPERTIES CXX_CLANG_TIDY "")

include(GoogleTest)
gtest_discover_tests(layers_gtest)
configure_xtensor_parallel_target(layers_gtest)

##
## fundamental_mechanisms_gtest: mathematical correctness of every layer mechanism
##
add_executable(fundamental_mechanisms_gtest
    fundamental_mechanisms_gtest.cpp
)

target_link_libraries(fundamental_mechanisms_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        layers
        tensor
)

target_include_directories(fundamental_mechanisms_gtest
    PRIVATE
        ${cnpy_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(fundamental_mechanisms_gtest PROPERTIES CXX_CLANG_TIDY "")

gtest_discover_tests(fundamental_mechanisms_gtest)
configure_xtensor_parallel_target(fundamental_mechanisms_gtest)

##
## spiking_mechanisms_gtest: comprehensive spiking-specific behaviour tests
## Covers: Lif, LifIntegrator, LifBPTT (temporal integration, adaptation,
##          hard/soft reset, state persistence), ThresholdDependentBatchNorm,
##          ExponentialSurrogate, BoxcarSurrogate.
##
add_executable(spiking_mechanisms_gtest
    spiking_mechanisms_gtest.cpp
)

target_link_libraries(spiking_mechanisms_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        layers
        tensor
)

target_include_directories(spiking_mechanisms_gtest
    PRIVATE
        ${cnpy_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(spiking_mechanisms_gtest PROPERTIES CXX_CLANG_TIDY "")

gtest_discover_tests(spiking_mechanisms_gtest)
configure_xtensor_parallel_target(spiking_mechanisms_gtest)
