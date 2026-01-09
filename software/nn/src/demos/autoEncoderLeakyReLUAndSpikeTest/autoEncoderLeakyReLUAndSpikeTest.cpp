#include <cnpy.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/LeakyBPTT.hpp"
#include "nn/layers/LeakyIntegrator.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/MSELoss.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/saver/NetworkSerializer.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/utility/EigenParallel.hpp"
#include "nn/utility/batching.hpp"
#include "nn/utility/reset.hpp"
#include "nn/utility/synthetic_spike_data.hpp"
#include "nn/utility/vectorizationCheck.hpp"

using std::cout;
using std::fixed;
using std::make_shared;
using std::scientific;
using std::setprecision;
using std::string;
using std::tie;
using std::vector;

// If DEBUG is defined then show the debug information
#ifdef DEBUG
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

namespace
{
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

constexpr int OUTPUT_PRECISION = 20; // Precision for floating-point output

auto main(int /*argc*/, char* /*argv*/[]) -> int
{
    util::initializeEigenParallel();

    cout << fixed << scientific << setprecision(OUTPUT_PRECISION);

    printVectorizationSupport();
    // ==== Data Generation ====

    // Network parameters
    constexpr float learning_rate = 0.001;   // Reduced learning rate
    constexpr float target_loss = -1.0e-14F; // Target loss value for early stopping
    constexpr int input_dim = 100;           // Input dimension for synthetic data
    constexpr int hidden_dim1 = 50;          // First hidden layer dimension
    constexpr int hidden_dim2 = 40;          // Second hidden layer dimension
    constexpr int hidden_dim3 = 30;          // Third hidden layer dimension
    constexpr int hidden_dim4 = 20;          // Fourth hidden layer dimension
    constexpr int hidden_dim5 = 10;          // Fifth hidden layer dimension
    constexpr int bottleneck_dim = 10;       // bottleneck layer size
    constexpr int epochs = 4000; // Number of training epochs in which n_samples is presented
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

    constexpr float resist = 5.0F;    // Resistance R
    constexpr float capct = 1.0F;     // Capacitance C
    constexpr float v_thresh = 0.01F; // Membrane potential threshold

    // Create input and target tensors
    vector<nn::Tensor> input_sequence;
    vector<nn::Tensor> target_sequence;

    // Generate synthetic spike data
    tie(input_sequence, target_sequence) =
        generate_autoencoder_spike_data_of_ones(n_samples, input_dim, n_steps);

    // Stack sequence into single Tensor (Time*Batch, Features)
    // input_sequence is vector of n_steps, each (n_samples, input_dim)
    nn::Tensor inputs(n_samples * n_steps, input_dim);
    nn::Tensor targets(n_samples * n_steps, input_dim);

    for (int t = 0; t < n_steps; ++t)
    {
        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < input_dim; ++j)
            {
                float val = input_sequence[t].at(i, j);
                inputs.at(t * n_samples + i, j) = val;
                targets.at(t * n_samples + i, j) = val;
            }
        }
    }

    // Use SINGLE BATCH containing all data (simplified for this test)
    // Or we should adapt create_batches to handle 3D Logic.
    // Given the small data size, full batch training is fine.
    // inputs is now (1000, 100).

    // ==== Model Definition ====

    // Use LeakyBPTT for comparison with snnTorch
    auto EncLayer = [&](int in, int out) { return make_shared<Linear>(in, out); };

    auto ActLayer = [&](int steps)
    {
        return make_shared<LeakyBPTT>(steps,
                                      time_step,
                                      resist,
                                      capct,
                                      v_thresh,
                                      true,
                                      0.0F,
                                      false,
                                      std::make_shared<ExponentialSurrogate>(1.0F));
    };

    ///////////////////////////////
    //////// Encoder layers ///////
    ///////////////////////////////
    auto enc1 = EncLayer(input_dim, hidden_dim1);
    auto enc_act1 = ActLayer(n_steps);

    auto enc2 = EncLayer(hidden_dim1, hidden_dim2);
    auto enc_act2 = ActLayer(n_steps);

    auto enc3 = EncLayer(hidden_dim2, hidden_dim3);
    auto enc_act3 = ActLayer(n_steps);

    auto enc4 = EncLayer(hidden_dim3, hidden_dim4);
    auto enc_act4 = ActLayer(n_steps);

    auto enc5 = EncLayer(hidden_dim4, hidden_dim5);
    auto enc_act5 = ActLayer(n_steps);

    auto enc6 = EncLayer(hidden_dim5, bottleneck_dim);
    auto enc_act6 = ActLayer(n_steps);

    ///////////////////////////////
    //////// Decoder layers ///////
    ///////////////////////////////
    auto dec1 = EncLayer(bottleneck_dim, hidden_dim5);
    auto dec_act1 = ActLayer(n_steps);

    auto dec2 = EncLayer(hidden_dim5, hidden_dim4);
    auto dec_act2 = ActLayer(n_steps);

    auto dec3 = EncLayer(hidden_dim4, hidden_dim3);
    auto dec_act3 = ActLayer(n_steps);

    auto dec4 = EncLayer(hidden_dim3, hidden_dim2);
    auto dec_act4 = ActLayer(n_steps);

    auto dec5 = EncLayer(hidden_dim2, hidden_dim1);
    auto dec_act5 = ActLayer(n_steps);

    auto dec6 = EncLayer(hidden_dim1, input_dim);
    // Final Layer: Readout Mode (Leaky Integrator BPTT)
    // readout_mode = true
    auto dec_act6 =
        make_shared<LeakyBPTT>(n_steps, time_step, resist, capct, v_thresh, true, 0.0F, true);

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

    // Helper to load weights explicitly from NPZ files generated by Python script
    auto load_weights_explicit = [&](Sequential& seq, const string& file) -> bool
    {
        if (!std::filesystem::exists(file)) return false;
        try
        {
            cnpy::npz_t data = cnpy::npz_load(file);
            for (size_t i = 0; i < seq.layers.size(); ++i)
            {
                if (auto lin = std::dynamic_pointer_cast<Linear>(seq.layers[i]))
                {
                    string key_prefix = std::to_string(i);
                    string w_key = key_prefix + ".weight";
                    if (data.count(w_key))
                    {
                        auto& arr = data[w_key];
                        float* w_ptr = arr.data<float>();
                        // Check dimensions loosely (total size)
                        size_t total_size = 1;
                        for (auto d : arr.shape) total_size *= d;

                        if (lin->weight.rows() * lin->weight.cols() != total_size)
                        {
                            std::cerr << "Shape mismatch loading " << w_key << "\n";
                            continue;
                        }

                        for (int r = 0; r < lin->weight.rows(); ++r)
                        {
                            for (int c = 0; c < lin->weight.cols(); ++c)
                            {
                                lin->weight.at(r, c) = w_ptr[r * lin->weight.cols() + c];
                            }
                        }
                    }

                    string b_key = key_prefix + ".bias";
                    if (data.count(b_key))
                    {
                        auto& arr = data[b_key];
                        float* b_ptr = arr.data<float>();
                        for (int r = 0; r < lin->bias.rows(); ++r)
                        {
                            for (int c = 0; c < lin->bias.cols(); ++c)
                            {
                                lin->bias.at(r, c) = b_ptr[r * lin->bias.cols() + c];
                            }
                        }
                    }
                }
            }
            std::cout << "Loaded weights from " << file << "\n";
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error loading weights: " << e.what() << "\n";
            return false;
        }
    };

    const bool loaded_weights = load_weights_explicit(encoders, encoder_weights_file_path) &&
                                load_weights_explicit(decoders, decoder_weights_file_path);

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

        // Initialize decoder weights
        kaimingSNNInitializer(dec1);
        kaimingSNNInitializer(dec2);
        kaimingSNNInitializer(dec3);
        kaimingSNNInitializer(dec4);
        kaimingSNNInitializer(dec5);
        kaimingSNNInitializer(dec6);
    }

    // ==== Optimizer ====
    vector<nn::Tensor*> params;
    vector<nn::Tensor*> encoder_params = encoders.params();
    vector<nn::Tensor*> decoder_params = decoders.params();

    params.insert(params.end(), encoder_params.begin(), encoder_params.end());
    params.insert(params.end(), decoder_params.begin(), decoder_params.end());

    // Scale down weights to prevent explosion with high Tau
    if (!loaded_weights)
    {
        for (auto* p : params)
        {
            // Only scale weights, not biases (though biases are usually 0 init)
            // KaimingSNN might give large values for this depth.
            // Multiply by 0.01f
            for (int i = 0; i < p->rows(); ++i)
            {
                for (int j = 0; j < p->cols(); ++j)
                {
                    p->at(i, j) *= 0.01f;
                }
            }
        }
    }

    Adam optimizer(learning_rate);
    optimizer.attach(params);

    // ==== Training Loop ====
    float epoch_loss = std::numeric_limits<float>::max();
    std::ofstream log_file("cpp_loss_log.txt");

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // Zero gradients!
        optimizer.zero_grad(params);

        // Reset state (utils.reset)
        nn::utility::reset(encoders);
        nn::utility::reset(decoders);

        // For autoencoder, the target is the input itself
        mse_loss->set_target(inputs);

        // Forward pass (Time Unrolled internally)
        nn::Tensor encoded = encoders.forward(inputs);
        nn::Tensor decoded = decoders.forward(encoded);

        nn::Tensor loss_tensor = mse_loss->forward(decoded);

        // Compute gradients using the improved MSELoss backward pass
        nn::Tensor grad_loss = mse_loss->backward(decoded);

        // Backward pass (BPTT internally)
        nn::Tensor decoder_grad = decoders.backward(grad_loss);
        encoders.backward(decoder_grad);

        // Gradient Clipping
        float max_grad_norm = 1.0f;
        float total_norm_sq = 0.0f;
        for (auto* p : params)
        {
            nn::Tensor g = p->grad();
            // Simple sum of squares
            if (g.size() > 0)
            {
                // Access data directly if possible or iterate
                for (int i = 0; i < g.rows(); ++i)
                    for (int j = 0; j < g.cols(); ++j)
                    {
                        float val = g.at(i, j);
                        total_norm_sq += val * val;
                    }
            }
        }
        float total_norm = std::sqrt(total_norm_sq);

        if (epoch % 10 == 0 || epoch < 5)
        {
            std::cout << "Epoch " << epoch << " Grad Norm: " << total_norm << std::endl;
        }

        if (total_norm > max_grad_norm)
        {
            float scale = max_grad_norm / (total_norm + 1e-6f);
            for (auto* p : params)
            {
                nn::Tensor g = p->grad();
                // Scale gradient
                for (int i = 0; i < g.rows(); ++i)
                    for (int j = 0; j < g.cols(); ++j)
                    {
                        g.at(i, j) *= scale;
                    }
                p->set_grad(g);
            }
            if (epoch == 0) std::cout << "Gradients clipped. Scale: " << scale << std::endl;
        }

        // Update parameters
        optimizer.step(params);

        // Track best loss in epoch
        float current_loss = loss_tensor.at(0, 0);
        epoch_loss = current_loss; // Update final loss tracker

        if (epoch % 10 == 0)
        {
            std::cout << "Epoch " << epoch << " Loss: " << current_loss << std::endl;
        }
        log_file << epoch << "," << current_loss << "\n";
    }

    log_file.close();
    std::cout << "Final Loss: " << epoch_loss << std::endl;
    return 0;
}