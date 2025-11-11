#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>

#include "core/initializers/kaiming_snn.hpp"
#include "core/layers/Leaky.hpp"
#include "core/layers/Linear.hpp"
#include "core/layers/MSELoss.hpp"
#include "core/layers/Sequential.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/tensor/Tensor.hpp"
#include "util/EigenParallel.hpp"
#include "util/NetworkSerializer.hpp"
#include "util/batching.hpp"
#include "util/synthetic_spike_data.hpp"
#include "util/vectorizationCheck.hpp"

using Eigen::MatrixXf;
using std::cout;
using std::fixed;
using std::make_shared;
using std::scientific;
using std::setprecision;
using std::string;
using std::tie;
using std::vector;

// Initialize Eigen parallel execution is now called inside main

// If DEBUG is defined then show the debug information
#ifdef DEBUG
// Define a nice format for debugging eigen matrices
const Eigen::IOFormat CleanFmt(0,                // number of decimals
                               Eigen::Unaligned, // flags
                               ",",              // string between numbers
                               "\n",             // string between rows
                               "|",              // opening bracket
                               "|",              // closing bracket
                               "\n",             // string between matrices
                               "\n"              // closing bracket for the matrix
);

namespace
{
auto debug(const Batch& batch, const Tensor& y_pred, const Tensor& loss_tensor) -> void
{
    cout << "Input dimensions: " << batch.inputs.data.rows() << "x" << batch.inputs.data.cols()
         << '\n';
    cout << "Output dimensions: " << y_pred.data.rows() << "x" << y_pred.data.cols() << '\n';
    cout << "Target values: " << batch.targets.data.format(CleanFmt) << '\n';
    cout << "Output values: " << y_pred.data.format(CleanFmt) << '\n';
    cout << "Loss values: " << loss_tensor.data.format(CleanFmt) << '\n';
}
} // namespace
#endif

constexpr int OUTPUT_PRECISION = 20; // Precision for floating-point output

auto main(int /*argc*/, char* /*argv*/[]) -> int
{
    util::initializeEigenParallel();

    cout << fixed << scientific << setprecision(OUTPUT_PRECISION);

    printVectorizationSupport();
    // ==== Data Generation ====

    // Network parameters
    constexpr float learning_rate = 0.0001;  // Learning rate for the optimizer - low for stability
    constexpr float target_loss = -1.0e-14F; // Target loss value for early stopping
    constexpr int input_dim = 1000;          // Input dimension for synthetic data
    constexpr int hidden_dim1 = 500;         // First hidden layer dimension
    constexpr int hidden_dim2 = 250;         // Second hidden layer dimension
    constexpr int hidden_dim3 = 125;         // Third hidden layer dimension
    constexpr int hidden_dim4 = 63;          // Fourth hidden layer dimension
    constexpr int hidden_dim5 = 32;          // Fifth hidden layer dimension
    constexpr int bottleneck_dim = 15;       // bottleneck layer size
    constexpr int epochs = 200; // Number of training epochs in which n_samples is presented
    const string encoder_weights_file_path =
        "weights/encoder_spike_model_weights.npz"; // Model weights file
    const string decoder_weights_file_path =
        "weights/decoder_spike_model_weights.npz"; // Model weights file

    // Batch parameters
    constexpr int batch_size = 32; // Batch size for training

    // Parameters for synthetic spike train
    constexpr int n_samples = 10; // Number of samples for synthetic data the higher the better
    constexpr int n_steps = 100;  // Number of time steps in the spike train
    constexpr float time_step = 0.001F; // Time step duration

    constexpr float resist = 5.0F;      // Resistance R
    constexpr float capct = 1.0F;       // Capacitance C
    constexpr float v_thresh = 0.0001F; // Membrane potential threshold

    // Create input and target tensors
    vector<Tensor> inputs;
    vector<Tensor> targets;

    // Generate synthetic spike data
    // float max_rate = 500.0F; // Maximum firing rate
    // tie(inputs, targets) =
    //     generate_autoencoder_spike_data(n_samples, input_dim, n_steps,
    //     max_rate, time_step);
    tie(inputs, targets) = generate_autoencoder_spike_data_of_ones(n_samples, input_dim, n_steps);

    // ==== Model Definition ====

    ///////////////////////////////
    //////// Encoder layers ///////
    ///////////////////////////////
    auto enc1 = make_shared<Linear>(input_dim, hidden_dim1);
    auto enc_act1 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto enc2 = make_shared<Linear>(hidden_dim1, hidden_dim2);
    auto enc_act2 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto enc3 = make_shared<Linear>(hidden_dim2, hidden_dim3);
    auto enc_act3 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto enc4 = make_shared<Linear>(hidden_dim3, hidden_dim4);
    auto enc_act4 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto enc5 = make_shared<Linear>(hidden_dim4, hidden_dim5);
    auto enc_act5 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto enc6 = make_shared<Linear>(hidden_dim5, bottleneck_dim);
    auto enc_act6 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    ///////////////////////////////
    //////// Decoder layers ///////
    ///////////////////////////////
    auto dec1 = make_shared<Linear>(bottleneck_dim, hidden_dim5);
    auto dec_act1 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto dec2 = make_shared<Linear>(hidden_dim5, hidden_dim4);
    auto dec_act2 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto dec3 = make_shared<Linear>(hidden_dim4, hidden_dim3);
    auto dec_act3 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto dec4 = make_shared<Linear>(hidden_dim3, hidden_dim2);
    auto dec_act4 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto dec5 = make_shared<Linear>(hidden_dim2, hidden_dim1);
    auto dec_act5 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    auto dec6 = make_shared<Linear>(hidden_dim1, input_dim);
    auto dec_act6 = make_shared<Leaky>(time_step,
                                       resist,
                                       capct,
                                       v_thresh,
                                       true,
                                       0.0F,
                                       std::make_shared<ExponentialSurrogate>(1.0F));

    //////////////////////////
    // ==== Loss Layer ====///
    //////////////////////////
    auto mse_loss = std::make_shared<MSELoss>();

    Sequential encoders({
        enc1,
        enc_act1, // First encoder block
        enc2,
        enc_act2, // Second encoder block
        enc3,
        enc_act3, // Third encoder block
        enc4,
        enc_act4, // Fourth encoder block
        enc5,
        enc_act5, // Fifth encoder block
        enc6,
        enc_act6, // Sixth encoder block (to bottleneck)
    });

    Sequential decoders({
        dec1,
        dec_act1, // First decoder block
        dec2,
        dec_act2, // Second decoder block
        dec3,
        dec_act3, // Third decoder block
        dec4,
        dec_act4, // Fourth decoder block
        dec5,
        dec_act5, // Fifth decoder block
        dec6,
        dec_act6 // Final decoder layer (no activation)
    });

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
        kaimingSNNInitializer(enc1);
        kaimingSNNInitializer(enc2);
        kaimingSNNInitializer(enc3);
        kaimingSNNInitializer(enc4);
        kaimingSNNInitializer(enc5);
        kaimingSNNInitializer(enc6);

        // Initialize decoder weights and biases if no saved weights exist
        kaimingSNNInitializer(dec1);
        kaimingSNNInitializer(dec2);
        kaimingSNNInitializer(dec3);
        kaimingSNNInitializer(dec4);
        kaimingSNNInitializer(dec5);
        kaimingSNNInitializer(dec6);
    }

    // ==== Optimizer ====
    vector<Tensor*> params;
    vector<Tensor*> encoder_params = encoders.params();
    vector<Tensor*> decoder_params = decoders.params();

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

        for (const auto& batch : batches)
        {
            // For autoencoder, the target is the input itself
            mse_loss->set_target(batch.inputs);

            // Forward pass through encoder and decoder - Eigen will handle
            // internal parallelization
            Tensor encoded = encoders.forward(batch.inputs);
            Tensor decoded = decoders.forward(encoded);
            Tensor loss_tensor = mse_loss->forward(decoded);

            // Compute gradients using the improved MSELoss backward pass
            Tensor grad_loss = mse_loss->backward(decoded);

            // Backward pass through decoder first, then encoder
            Tensor decoder_grad = decoders.backward(grad_loss);
            encoders.backward(decoder_grad);

            // Update parameters
            optimizer.step(params);

            // Track best loss in epoch
            float tmp = loss_tensor.data(0, 0);
            epoch_loss = tmp < epoch_loss ? tmp : epoch_loss;

// If DEBUG is defined then show the debug information
#ifdef DEBUG
            debug(batch, y_pred, loss_tensor);
#endif
        }

        // Print progress
        constexpr int progress_interval = 1; // Print progress every N epochs
        if (epoch % progress_interval == 0)
        {
            cout << "Epoch: " << epoch << " - Loss: " << epoch_loss << "\r" << std::flush;
        }

        // Stop training when target loss is achieved
        if (epoch_loss < target_loss)
        {
            break;
        }
    }

    // ==== End of Training ====
    cout << "Training complete. Final loss: " << epoch_loss << "\n";

    // Save both encoder and decoder networks
    if (NetworkSerializer::saveNetwork(encoders, encoder_weights_file_path) &&
        NetworkSerializer::saveNetwork(decoders, decoder_weights_file_path))
    {
        cout << "Network weights saved successfully.\n";
    }
    else
    {
        cout << "Failed to save network weights.\n";
    }

    return 0;
}
