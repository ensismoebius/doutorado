##
## Autoencoder config unit tests (GoogleTest)
##
## Builds:
## - `autoencoder_config_gtest`: tests for AutoencoderConfig
##
## Note: AutoencoderBuilders tests are in experiments/autoencoderRunner since the create()
## factory function is implemented there, not in core.
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
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/models
)

nn_gtest_discover_tests(autoencoder_config_gtest)