##
## GRU autoencoder unit tests (GoogleTest)
##

add_executable(gru_autoencoder_gtest
    gru_autoencoder_gtest.cpp
)

target_link_libraries(gru_autoencoder_gtest
    PRIVATE
        google_test
        gru_models
        tensor
        layers
        optimizers
)

target_include_directories(
    gru_autoencoder_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/models
)

nn_gtest_discover_tests(gru_autoencoder_gtest)
