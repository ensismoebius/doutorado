##
## Autoencoder config unit tests (GoogleTest)
##
## Builds:
## - `autoencoder_config_gtest`: tests for AutoencoderConfig
##
## Note: AutoencoderBuilders tests are in experiments/autoencoderRunner: the create()
## factory stays experiment-specific, while the models it builds are core.
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

nn_gtest_discover_tests(autoencoder_config_gtest)