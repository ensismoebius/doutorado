/**
 * @file autoEncoderLeakyReLUTest.cpp
 * @brief Dense (non-spiking) autoencoder demo using `Sequential` + Adam.
 *
 * This demo exercises the classic module/loss/optimizer loop without spiking dynamics:
 * - model is an MLP built from Linear + LeakyReLU
 * - loss is `MSELoss`
 * - optimization is `Adam`
 *
 * It serves as a baseline for comparing with the spiking autoencoder demo.
 */

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>

#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/LeakyReLU.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/MSELoss.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/saver/NetworkSerializer.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/testing.hpp"
#include "nn/utility/EigenParallel.hpp"
#include "nn/utility/batching.hpp"
#include "nn/utility/synthetic_spike_data.hpp"
#include "nn/utility/vectorizationCheck.hpp"

using std::cout;
using std::fixed;
using std::flush;
using std::make_shared;
using std::scientific;
using std::setprecision;
using std::string;
using std::tie;
using std::vector;

// If DEBUG is defined then show the debug information
#ifdef DEBUG
namespace
{
// Format for printing tensors (simple CSV-like format)
auto print_tensor = [](const nn::Tensor& t)
{
    for (size_t i = 0; i < t.rows(); ++i)
    {
        for (size_t j = 0; j < t.cols(); ++j)
        {
            if (j > 0) std::cout << ",";
            std::cout << t.at(i, j);
        }
        std::cout << "\n";
    }
};

auto debug(const Batch& batch, const nn::Tensor& y_pred, const nn::Tensor& loss_tensor) -> void
{
    cout << "Input dimensions: " << batch.inputs.rows() << "x" << batch.inputs.cols() << '\n';
    cout << "Output dimensions: " << y_pred.rows() << "x" << y_pred.cols() << '\n';
    cout << "Target values:\n";
    print_tensor(batch.targets);
    cout << "Output values:\n";
    print_tensor(y_pred);
    cout << "Loss values:\n";
    print_tensor(loss_tensor);
}
} // namespace
#endif

constexpr int OUTPUT_PRECISION = 40; // Precision for floating-point output

auto main(int /*argc*/, char* /*argv*/[]) -> int
{
    util::initializeEigenParallel();

    cout << fixed << scientific << setprecision(OUTPUT_PRECISION);

    printVectorizationSupport();
    // ==== Data Generation ====

    // Network parameters
    constexpr float learning_rate = 0.001;  // Learning rate for the optimizer - low for stability
    constexpr float target_loss = 1.0e-14F; // Target loss value for early stopping
    constexpr int input_dim = 500;          // Input dimension for synthetic data
    constexpr int hidden_dim1 = 250;        // First hidden layer dimension
    constexpr int hidden_dim2 = 125;        // Second hidden layer dimension
    constexpr int hidden_dim3 = 63;         // Third hidden layer dimension
    constexpr int hidden_dim4 = 31;         // Fourth hidden layer dimension
    constexpr int hidden_dim5 = 10;         // Fifth hidden layer dimension
    constexpr int bottleneck_dim = 5;       // bottleneck layer size
    constexpr int epochs = 100000; // Number of training epochs in which n_samples is presented
    const string encoder_weights_file_path =
        "weights/encoder_model_weights.npz"; // Model weights file
    const string decoder_weights_file_path =
        "weights/decoder_model_weights.npz"; // Model weights file

    // Batch parameters
    constexpr int batch_size = 32; // Batch size for training

    // Parameters for synthetic spike train
    constexpr int n_samples = 5;      // Number of samples for synthetic data the higher the better
    constexpr int n_steps = 10;       // Number of time steps in the spike train
    constexpr float max_rate = 1.0F;  // Maximum firing rate
    constexpr float time_step = 1.0F; // Time step duration

    // Create input and target tensors
    vector<nn::Tensor> inputs;
    vector<nn::Tensor> targets;

    // Generate synthetic spike data
    tie(inputs, targets) =
        generate_autoencoder_spike_data(n_samples, input_dim, n_steps, max_rate, time_step);

    // ==== Model Definition ====

    // Encoder layers

    auto encoder1 = make_shared<Linear>(input_dim, hidden_dim1);
    auto encoder_act1 = make_shared<LeakyReLU>(0.01F);
    auto encoder2 = make_shared<Linear>(hidden_dim1, hidden_dim2);
    auto encoder_act2 = make_shared<LeakyReLU>(0.01F);
    auto encoder3 = make_shared<Linear>(hidden_dim2, hidden_dim3);
    auto encoder_act3 = make_shared<LeakyReLU>(0.01F);
    auto encoder4 = make_shared<Linear>(hidden_dim3, hidden_dim4);
    auto encoder_act4 = make_shared<LeakyReLU>(0.01F);
    auto encoder5 = make_shared<Linear>(hidden_dim4, hidden_dim5);
    auto encoder_act5 = make_shared<LeakyReLU>(0.01F);
    auto encoder6 = make_shared<Linear>(hidden_dim5, bottleneck_dim);
    auto encoder_act6 = make_shared<LeakyReLU>(0.01F);

    // Decoder layers

    auto decoder1 = make_shared<Linear>(bottleneck_dim, hidden_dim5);
    auto decoder_act1 = make_shared<LeakyReLU>(0.01F);
    auto decoder2 = make_shared<Linear>(hidden_dim5, hidden_dim4);
    auto decoder_act2 = make_shared<LeakyReLU>(0.01F);
    auto decoder3 = make_shared<Linear>(hidden_dim4, hidden_dim3);
    auto decoder_act3 = make_shared<LeakyReLU>(0.01F);
    auto decoder4 = make_shared<Linear>(hidden_dim3, hidden_dim2);
    auto decoder_act4 = make_shared<LeakyReLU>(0.01F);
    auto decoder5 = make_shared<Linear>(hidden_dim2, hidden_dim1);
    auto decoder_act5 = make_shared<LeakyReLU>(0.01F);
    auto decoder6 = make_shared<Linear>(hidden_dim1, input_dim);
    // ==== Loss Layer ====
    auto mse_loss = std::make_shared<MSELoss>();

    Sequential encoders({encoder1,
        encoder_act1,
        encoder2,
        encoder_act2,
        encoder3,
        encoder_act3,
        encoder4,
        encoder_act4,
        encoder5,
        encoder_act5,
        encoder6,
        encoder_act6});

    Sequential decoders({decoder1,
        decoder_act1,
        decoder2,
        decoder_act2,
        decoder3,
        decoder_act3,
        decoder4,
        decoder_act4,
        decoder5,
        decoder_act5,
        decoder6});

    // ==== Initialization ====

    // Try to load existing weights, if they exist
    // otherwise initialize encoder and decoder weights and biases
    // Try to load weights for both encoder and decoder networks
    const bool loaded_weights =
        NetworkSerializer::loadNetwork(encoders, encoder_weights_file_path) &&
        NetworkSerializer::loadNetwork(decoders, decoder_weights_file_path);

    if (!loaded_weights)
    {
        std::cerr << "Failed to load weights, initializing with Kaiming "
                     "initialization\n";

        // Initialize encoder weights and biases
        kaimingSNNInitializer(encoder1, nn::testing::SEED);
        kaimingSNNInitializer(encoder2, nn::testing::SEED);
        kaimingSNNInitializer(encoder3, nn::testing::SEED);
        kaimingSNNInitializer(encoder4, nn::testing::SEED);
        kaimingSNNInitializer(encoder5, nn::testing::SEED);
        kaimingSNNInitializer(encoder6, nn::testing::SEED);

        // Initialize decoder weights and biases if no saved weights exist
        kaimingSNNInitializer(decoder1, nn::testing::SEED);
        kaimingSNNInitializer(decoder2, nn::testing::SEED);
        kaimingSNNInitializer(decoder3, nn::testing::SEED);
        kaimingSNNInitializer(decoder4, nn::testing::SEED);
        kaimingSNNInitializer(decoder5, nn::testing::SEED);
        kaimingSNNInitializer(decoder6, nn::testing::SEED);
    }

    // ==== Optimizer ====
    vector<nn::Tensor*> params;
    vector<nn::Tensor*> encoder_params = encoders.params();
    vector<nn::Tensor*> decoder_params = decoders.params();

    params.insert(params.end(), encoder_params.begin(), encoder_params.end());
    params.insert(params.end(), decoder_params.begin(), decoder_params.end());

    Adam optimizer(learning_rate);
    optimizer.attach(params);

    // ==== Training Loop ====
    float epoch_loss = std::numeric_limits<float>::max();

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // Create batches
        auto batches = create_batches(inputs, targets, batch_size);

        // Parallelize batch processing using OpenMP
        for (const auto& batch : batches)
        {
            // For autoencoder, the target is the input itself
            mse_loss->set_target(batch.inputs);

            // Forward pass through encoder and decoder - backend will handle
            // internal parallelization
            nn::Tensor encoded = encoders.forward(batch.inputs);
            nn::Tensor decoded = decoders.forward(encoded);
            nn::Tensor loss_tensor = mse_loss->forward(decoded);

            // Compute gradients using the improved MSELoss backward pass
            nn::Tensor grad_loss = mse_loss->backward(decoded);

            // Backward pass through decoder first, then encoder
            nn::Tensor decoder_grad = decoders.backward(grad_loss);
            encoders.backward(decoder_grad);

            // Update parameters
            optimizer.step(params);

            // Track best loss in epoch
            float tmp = loss_tensor(0, 0);
            epoch_loss = tmp < epoch_loss ? tmp : epoch_loss;

// If DEBUG is defined then show the debug information
#ifdef DEBUG
            debug(batch, decoded, loss_tensor);
#endif
        }

        // Print progress
        constexpr int progress_interval = 10; // Print progress every N epochs
        if (epoch % progress_interval == 0)
        {
            cout << "Epoch: " << epoch << " - Loss: " << epoch_loss << "\r" << flush;
        }

        // Stop training when target loss is achieved
        if (epoch_loss < target_loss)
        {
            break;
        }
    }

    // ==== End of Training ====
    cout << "Training complete. Final loss: " << epoch_loss << "\n";
    bool enc_saved = NetworkSerializer::saveNetwork(encoders, encoder_weights_file_path);
    bool dec_saved = NetworkSerializer::saveNetwork(decoders, decoder_weights_file_path);
    if (enc_saved && dec_saved)
    {
        cout << "Network weights saved successfully.\n";
    }
    else
    {
        cout << "Failed to save network weights.\n";
    }

    return 0;
}
