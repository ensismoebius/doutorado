##
## Autoencoder LeakyReLU unit tests (GoogleTest)
## Tests dense MSE autoencoder and spike autoencoder output shape.
##
if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

add_executable(autoencoder_leakyrelu_gtest
    autoencoder_leakyrelu_gtest.cpp
)

target_link_libraries(autoencoder_leakyrelu_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
        util
        layers
        optimizers
        initializers
        cnpy
)

target_include_directories(autoencoder_leakyrelu_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${LIB_DIR}/util
        ${OpenMP_INCLUDE_DIRS}
)

configure_xtensor_parallel_target(autoencoder_leakyrelu_gtest)

include(GoogleTest)
nn_gtest_discover_tests(autoencoder_leakyrelu_gtest)
