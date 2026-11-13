##
## Autoencoder config unit tests (GoogleTest)
##
## Builds:
## - `autoencoder_config_gtest`: tests for AutoencoderConfig
##
## - `autoencoder_models_gtest`: the fused-window and protocol (ANN/SNN) autoencoders
##

add_executable(autoencoder_config_gtest
    autoencoder_config_gtest.cpp
)

target_link_libraries(autoencoder_config_gtest 
    PRIVATE 
        google_test 
        tensor
        layers
)

target_include_directories(
    autoencoder_config_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
)

gtest_discover_tests(autoencoder_config_gtest)

add_executable(autoencoder_models_gtest
    autoencoder_models_gtest.cpp
)

target_link_libraries(autoencoder_models_gtest
    PRIVATE
        google_test
        autoencoder_models
        tensor
        layers
        util
        xtensor xtensor-blas
)

target_include_directories(autoencoder_models_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

gtest_discover_tests(autoencoder_models_gtest)
configure_xtensor_parallel_target(autoencoder_models_gtest)
