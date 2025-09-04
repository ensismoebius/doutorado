#include "initializers/kaiming_snn.hpp"
#include "layers/LeakyReLU.hpp"
#include "layers/Linear.hpp"
#include "layers/MSELoss.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include "util/EigenParallel.hpp"
#include "util/NetworkSerializer.hpp"
#include "util/batching.hpp"
#include "util/synthetic_spike_data.hpp"
#include "util/vectorizationCheck.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>

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

// If DEBUG is defined then show the debug information
#ifdef DEBUG
namespace {
auto debug(const Batch &batch, const Tensor &y_pred, const Tensor &loss_tensor) -> void {
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

auto main(int /*argc*/, char * /*argv*/[]) -> int {

  util::initializeEigenParallel();

  cout << fixed << scientific << setprecision(OUTPUT_PRECISION);

  printVectorizationSupport();
  // ==== Data Generation ====

  // Network parameters
  constexpr float learning_rate = 0.001;  // Learning rate for the optimizer - reduced for stability
  constexpr float target_loss = 1.0e-20F; // Target loss value for early stopping
  constexpr int input_dim = 500;          // Input dimension for synthetic data
  constexpr int hidden_dim1 = 250;        // First hidden layer dimension
  constexpr int hidden_dim2 = 125;        // Second hidden layer dimension
  constexpr int hidden_dim3 = 63;         // Third hidden layer dimension
  constexpr int hidden_dim4 = 31;         // Fourth hidden layer dimension
  constexpr int hidden_dim5 = 10;         // Fifth hidden layer dimension
  constexpr int bottleneck_dim = 5;       // bottleneck layer size
  constexpr int epochs = 100000; // Number of training epochs in which n_samples is presented
  const string weights_file_path = "weights/model_weights.npz"; // Model weights file

  // Batch parameters
  constexpr int batch_size = 32; // Batch size for training

  // Parameters for synthetic spike train
  constexpr int n_samples = 5;      // Number of samples for synthetic data the higher the better
  constexpr int n_steps = 10;       // Number of time steps in the spike train
  constexpr float max_rate = 1.0F;  // Maximum firing rate
  constexpr float time_step = 1.0F; // Time step duration

  // Create input and target tensors
  vector<Tensor> inputs;
  vector<Tensor> targets;

  // Generate synthetic spike data
  tie(inputs, targets) =
      generate_autoencoder_spike_data(n_samples, input_dim, n_steps, max_rate, time_step);

  // ==== Model Definition ====

  // Constants for model saving/loading
  const vector<string> layer_names = {"encoder1", "encoder2", "encoder3", "encoder4",
                                      "encoder5", "encoder6", "decoder1", "decoder2",
                                      "decoder3", "decoder4", "decoder5", "decoder6"};

  // Encoder layers
  auto encoder1 = make_shared<Linear>(input_dim, hidden_dim1);
  auto encoder_act1 = make_shared<LeakyReLU>();
  auto encoder2 = make_shared<Linear>(hidden_dim1, hidden_dim2);
  auto encoder_act2 = make_shared<LeakyReLU>();
  auto encoder3 = make_shared<Linear>(hidden_dim2, hidden_dim3);
  auto encoder_act3 = make_shared<LeakyReLU>();
  auto encoder4 = make_shared<Linear>(hidden_dim3, hidden_dim4);
  auto encoder_act4 = make_shared<LeakyReLU>();
  auto encoder5 = make_shared<Linear>(hidden_dim4, hidden_dim5);
  auto encoder_act5 = make_shared<LeakyReLU>();
  auto encoder6 = make_shared<Linear>(hidden_dim5, bottleneck_dim);
  auto encoder_act6 = make_shared<LeakyReLU>();

  // Decoder layers
  auto decoder1 = make_shared<Linear>(bottleneck_dim, hidden_dim5);
  auto decoder_act1 = make_shared<LeakyReLU>();
  auto decoder2 = make_shared<Linear>(hidden_dim5, hidden_dim4);
  auto decoder_act2 = make_shared<LeakyReLU>();
  auto decoder3 = make_shared<Linear>(hidden_dim4, hidden_dim3);
  auto decoder_act3 = make_shared<LeakyReLU>();
  auto decoder4 = make_shared<Linear>(hidden_dim3, hidden_dim2);
  auto decoder_act4 = make_shared<LeakyReLU>();
  auto decoder5 = make_shared<Linear>(hidden_dim2, hidden_dim1);
  auto decoder_act5 = make_shared<LeakyReLU>();
  auto decoder6 = make_shared<Linear>(hidden_dim1, input_dim);

  // ==== Loss Layer ====
  auto mse_loss = std::make_shared<MSELoss>();

  Sequential encoders({
      encoder1, encoder_act1, // First encoder block
      encoder2, encoder_act2, // Second encoder block
      encoder3, encoder_act3, // Third encoder block
      encoder4, encoder_act4, // Fourth encoder block
      encoder5, encoder_act5, // Fifth encoder block
      encoder6, encoder_act6, // Sixth encoder block (to bottleneck)
  });

  Sequential decoders({
      decoder1, decoder_act1, // First decoder block
      decoder2, decoder_act2, // Second decoder block
      decoder3, decoder_act3, // Third decoder block
      decoder4, decoder_act4, // Fourth decoder block
      decoder5, decoder_act5, // Fifth decoder block
      decoder6                // Final decoder layer (no activation)
  });

  // ==== Initialization ====

  // Try to load existing weights, if they exist
  // otherwise initialize encoder and decoder weights and biases
  // Try to load weights for both encoder and decoder networks
  const bool loaded_weights =
      NetworkSerializer::loadNetwork(encoders, weights_file_path, layer_names) &&
      NetworkSerializer::loadNetwork(decoders, weights_file_path + ".decoder", layer_names);

  if (!loaded_weights) {
    std::cerr << "Failed to load weights, initializing with Kaiming initialization\n";

    // Initialize encoder weights and biases
    kaimingSNNInitializer(encoder1);
    kaimingSNNInitializer(encoder2);
    kaimingSNNInitializer(encoder3);
    kaimingSNNInitializer(encoder4);
    kaimingSNNInitializer(encoder5);
    kaimingSNNInitializer(encoder6);

    // Initialize decoder weights and biases if no saved weights exist
    kaimingSNNInitializer(decoder1);
    kaimingSNNInitializer(decoder2);
    kaimingSNNInitializer(decoder3);
    kaimingSNNInitializer(decoder4);
    kaimingSNNInitializer(decoder5);
    kaimingSNNInitializer(decoder6);
  }

  vector<Tensor *> params = {
      &encoder1->weight, &encoder1->bias,   &encoder2->weight, &encoder2->bias,   &encoder3->weight,
      &encoder3->bias,   &encoder4->weight, &encoder4->bias,   &encoder5->weight, &encoder5->bias,
      &encoder6->weight, &encoder6->bias,   &decoder1->weight, &decoder1->bias,   &decoder2->weight,
      &decoder2->bias,   &decoder3->weight, &decoder3->bias,   &decoder4->weight, &decoder4->bias,
      &decoder5->weight, &decoder5->bias,   &decoder6->weight, &decoder6->bias};
  Adam optimizer(learning_rate);
  optimizer.attach(params);

  // ==== Training Loop ====
  float epoch_loss = std::numeric_limits<float>::max();

  for (size_t epoch = 0; epoch < epochs; ++epoch) {

    // Create batches
    auto batches = create_batches(inputs, targets, batch_size);

// Parallelize batch processing using OpenMP
#pragma omp parallel for schedule(dynamic) reduction(min : epoch_loss)
    for (const auto &batch : batches) {
      // For autoencoder, the target is the input itself
      mse_loss->set_target(batch.inputs);

      // Forward pass through encoder and decoder - Eigen will handle internal parallelization
      Tensor encoded = encoders.forward(batch.inputs);
      Tensor decoded = decoders.forward(encoded);
      Tensor loss_tensor = mse_loss->forward(decoded);

#pragma omp critical
      {
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
      } // End of critical section

// If DEBUG is defined then show the debug information
#ifdef DEBUG
      debug(batch, y_pred, loss_tensor);
#endif
    }

    // Print progress
    constexpr int progress_interval = 1; // Print progress every N epochs
    if (epoch % progress_interval == 0) {
      cout << "Epoch: " << epoch << " - Loss: " << epoch_loss << "\r";
    }

    // Stop training when target loss is achieved
    if (epoch_loss < target_loss) {
      break;
    }
  }

  // ==== End of Training ====
  cout << "Training complete. Final loss: " << epoch_loss << "\n";

  // Save both encoder and decoder networks
  if (NetworkSerializer::saveNetwork(encoders, weights_file_path, layer_names) &&
      NetworkSerializer::saveNetwork(decoders, weights_file_path + ".decoder", layer_names)) {
    cout << "Network weights saved successfully.\n";
  } else {
    cout << "Failed to save network weights.\n";
  }

  return 0;
}
