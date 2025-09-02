#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>

#include "initializers/kaiming_snn.hpp"
#include "layers/Leaky.hpp"
#include "layers/LeakyReLU.hpp"
#include "layers/Linear.hpp"
#include "layers/ReLU.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include "util/EigenParallel.hpp"
#include "util/batching.hpp"
#include "util/synthetic_spike_data.hpp"
#include "util/vectorizationCheck.hpp"

using Eigen::MatrixXf;
using std::cout;
using std::fixed;
using std::make_shared;
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

auto main(int /*argc*/, char * /*argv*/[]) -> int {

  util::initializeEigenParallel();

  cout << fixed << setprecision(20);

  printVectorizationSupport();
  // ==== Data Generation ====

  // Network parameters
  constexpr float learning_rate = 0.001; // Learning rate for the optimizer - reduced for stability
  constexpr int input_dim = 2000;        // Input dimension for synthetic data
  constexpr int hidden_dim1 = 1600;      // First hidden layer dimension
  constexpr int hidden_dim2 = 1200;      // Second hidden layer dimension
  constexpr int hidden_dim3 = 800;       // Third hidden layer dimension
  constexpr int hidden_dim4 = 400;       // Fourth hidden layer dimension
  constexpr int hidden_dim5 = 200;       // Fifth hidden layer dimension
  constexpr int bottleneck_dim = 20;     // bottleneck layer size
  constexpr int epochs = 10000000; // Number of training epochs in which n_samples is presented

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

  Sequential model({
      encoder1, encoder_act1, // First encoder block
      encoder2, encoder_act2, // Second encoder block
      encoder3, encoder_act3, // Third encoder block
      encoder4, encoder_act4, // Fourth encoder block
      encoder5, encoder_act5, // Fifth encoder block
      encoder6, encoder_act6, // Sixth encoder block (to bottleneck)
      decoder1, decoder_act1, // First decoder block
      decoder2, decoder_act2, // Second decoder block
      decoder3, decoder_act3, // Third decoder block
      decoder4, decoder_act4, // Fourth decoder block
      decoder5, decoder_act5, // Fifth decoder block
      decoder6                // Final decoder layer (no activation)
  });

  // ==== Initialization ====
  // Initialize encoder weights and biases
  kaimingSNNInitializer(input_dim, hidden_dim1, encoder1->weight, encoder1->bias);
  kaimingSNNInitializer(hidden_dim1, hidden_dim2, encoder2->weight, encoder2->bias);
  kaimingSNNInitializer(hidden_dim2, hidden_dim3, encoder3->weight, encoder3->bias);
  kaimingSNNInitializer(hidden_dim3, hidden_dim4, encoder4->weight, encoder4->bias);
  kaimingSNNInitializer(hidden_dim4, hidden_dim5, encoder5->weight, encoder5->bias);
  kaimingSNNInitializer(hidden_dim5, bottleneck_dim, encoder6->weight, encoder6->bias);

  // Initialize decoder weights and biases
  kaimingSNNInitializer(bottleneck_dim, hidden_dim5, decoder1->weight, decoder1->bias);
  kaimingSNNInitializer(hidden_dim5, hidden_dim4, decoder2->weight, decoder2->bias);
  kaimingSNNInitializer(hidden_dim4, hidden_dim3, decoder3->weight, decoder3->bias);
  kaimingSNNInitializer(hidden_dim3, hidden_dim2, decoder4->weight, decoder4->bias);
  kaimingSNNInitializer(hidden_dim2, hidden_dim1, decoder5->weight, decoder5->bias);
  kaimingSNNInitializer(hidden_dim1, input_dim, decoder6->weight, decoder6->bias);

  vector<Tensor *> params = {
      &encoder1->weight, &encoder1->bias,   &encoder2->weight, &encoder2->bias,   &encoder3->weight,
      &encoder3->bias,   &encoder4->weight, &encoder4->bias,   &encoder5->weight, &encoder5->bias,
      &encoder6->weight, &encoder6->bias,   &decoder1->weight, &decoder1->bias,   &decoder2->weight,
      &decoder2->bias,   &decoder3->weight, &decoder3->bias,   &decoder4->weight, &decoder4->bias,
      &decoder5->weight, &decoder5->bias,   &decoder6->weight, &decoder6->bias};
  Adam optimizer(learning_rate);
  optimizer.attach(params);

  // ==== Loss Function ====
  auto mse_loss = [](const Tensor &y_pred, const Tensor &y_target) -> Tensor {
    // Use Eigen's parallelized array operations with numerical stability checks
    MatrixXf diff = y_pred.data - y_target.data;

    // Check for invalid values in the predictions
    if (!y_pred.data.allFinite()) {
      std::cerr << "Warning: Non-finite values detected in predictions\n";
      // Return a very large but finite loss
      MatrixXf out(1, 1);
      out(0, 0) = std::numeric_limits<float>::max() / 2.0f;
      return {out};
    }

    // Compute MSE with careful reduction
    float sum_squared = 0.0f;
    long count = diff.size();

#pragma omp parallel for reduction(+ : sum_squared)
    for (long i = 0; i < count; ++i) {
      float val = diff(i);
      sum_squared += val * val;
    }

    float mse = sum_squared / static_cast<float>(count);

    // Clip extremely large values to prevent overflow
    mse = std::min(mse, std::numeric_limits<float>::max() / 2.0f);

    MatrixXf out(1, 1);
    out(0, 0) = mse;
    return {out};
  };

  // ==== Training Loop ====
  float epoch_loss = std::numeric_limits<float>::max();

  for (size_t epoch = 0; epoch < epochs; ++epoch) {

    // Create batches
    auto batches = create_batches(inputs, targets, batch_size);

// Parallelize batch processing using OpenMP
#pragma omp parallel for schedule(dynamic) reduction(min : epoch_loss)
    for (const auto &batch : batches) {
      // Forward pass - Eigen will handle internal parallelization
      Tensor y_pred = model.forward(batch.inputs);

      // Compute loss - Using Eigen's parallelized operations
      Tensor loss_tensor = mse_loss(y_pred, batch.targets);

      // Backward pass (dL/dy_pred = 2*(y_pred - y_true))
      constexpr float mse_gradient_factor = 2.0F; // Factor for MSE gradient
      // Use Eigen's array operations which are automatically parallelized
      MatrixXf grad =
          (mse_gradient_factor * (y_pred.data - batch.targets.data)).array() / y_pred.data.size();
      Tensor grad_loss(grad);

      // Normalize the gradient to prevent explosion
      float grad_norm = grad.norm();
      if (grad_norm > 1.0F) {
        grad *= 1.0F / grad_norm;
      }

#pragma omp critical
      {
        model.backward(grad_loss);

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
  }

  // ==== End of Training ====
  cout << "Training complete. Final loss: " << epoch_loss << "\n";
  return 0;
}
