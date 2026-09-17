##
## LSTM autoencoder unit tests (GoogleTest)
##
## Builds:
## - `lstm_autoencoder_gtest`: tests for LSTM autoencoder interface
##

add_executable(lstm_autoencoder_gtest
    lstm_autoencoder_gtest.cpp
)

target_link_libraries(lstm_autoencoder_gtest
    PRIVATE
        google_test
        lstm_models
        tensor
        layers
        optimizers
)

target_include_directories(
    lstm_autoencoder_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/models
)

gtest_discover_tests(lstm_autoencoder_gtest)

##
## lstm_batch_gtest: tests for LSTMLayer batch support (forward_batch / backward_batch)
##
add_executable(lstm_batch_gtest
    lstm_batch_gtest.cpp
)

target_link_libraries(lstm_batch_gtest
    PRIVATE
        google_test
        lstm_models
        tensor
        layers
        optimizers
)

target_include_directories(
    lstm_batch_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/models
)

gtest_discover_tests(lstm_batch_gtest)