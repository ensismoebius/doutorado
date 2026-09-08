add_executable(transformer_autoencoder_gtest
    transformer_autoencoder_gtest.cpp
)

target_link_libraries(transformer_autoencoder_gtest
    PRIVATE
        google_test
        transformer_models
        tensor
        layers
        optimizers
)

target_include_directories(transformer_autoencoder_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

nn_gtest_discover_tests(transformer_autoencoder_gtest)
