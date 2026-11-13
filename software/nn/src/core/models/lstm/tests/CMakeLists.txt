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

##
## lstm_ae_pytorch_parity_gtest: the WHOLE LSTM autoencoder (stacked encoder, tanh bottleneck,
## constant-latent decoder, masked MSE) against an independent PyTorch implementation, from
## identical weights: reconstruction, loss and every parameter gradient. Loads committed
## fixtures (fixtures/lstm_ae_refs.npz, from scripts/testing/gen_lstm_ae_refs.py), so no
## Python/torch is needed at build or CI time.
##
add_executable(lstm_ae_pytorch_parity_gtest
    lstm_ae_pytorch_parity_gtest.cpp
)

target_link_libraries(lstm_ae_pytorch_parity_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        lstm_models
        layers
        tensor
        cnpy-static
        z
)

target_include_directories(lstm_ae_pytorch_parity_gtest
    PRIVATE
        ${cnpy_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/models
)

set_target_properties(lstm_ae_pytorch_parity_gtest PROPERTIES CXX_CLANG_TIDY "")

gtest_discover_tests(lstm_ae_pytorch_parity_gtest)
configure_xtensor_parallel_target(lstm_ae_pytorch_parity_gtest)


##
## lstm_autoencoder_training_gtest: LSTMLayer shapes and gradients, and an LSTM autoencoder
## trained through the core Trainer
##
add_executable(lstm_autoencoder_training_gtest
    lstm_autoencoder_training_gtest.cpp
)

target_link_libraries(lstm_autoencoder_training_gtest
    PRIVATE
        google_test
        lstm_models
        tensor
        layers
        optimizers
        util
        nn_progress
        xtensor xtensor-blas
)

target_include_directories(lstm_autoencoder_training_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

gtest_discover_tests(lstm_autoencoder_training_gtest)
configure_xtensor_parallel_target(lstm_autoencoder_training_gtest)
