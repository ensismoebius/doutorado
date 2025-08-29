#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <tuple>

#include "initializers/kaiming_snn.hpp"
#include "layers/Leaky.hpp"
#include "layers/Linear.hpp"
#include "layers/ReLU.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
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

  cout << fixed << setprecision(10);

  printVectorizationSupport();

  // ==== Data Generation ====

  // Network parameters
  constexpr float learning_rate = 0.001; // Learning rate for the optimizer
  constexpr int bottleneck_dim = 4;      // bottleneck layer size
  constexpr int input_dim = 4;           // Input dimension for synthetic data
  constexpr int epochs = 100000; // Number of training epochs in which n_samples is presented

  // Batch parameters
  constexpr int batch_size = 5; // Batch size for training

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
  auto encoder = make_shared<Linear>(input_dim, bottleneck_dim);
  auto encoder_act = make_shared<Leaky>();

  auto decoder = make_shared<Linear>(bottleneck_dim, input_dim);
  auto decoder_act = make_shared<Leaky>();
  // auto decoder_out = make_shared<ReLU>();

  Sequential model({encoder, encoder_act, decoder, decoder_act /*, decoder_out*/});

  // ==== Initialization ====
  kaimingSNNInitializer(input_dim, bottleneck_dim, encoder->weight, encoder->bias);
  kaimingSNNInitializer(bottleneck_dim, input_dim, decoder->weight, decoder->bias);

  vector<Tensor *> params = {&encoder->weight, &encoder->bias, &decoder->weight, &decoder->bias};
  Adam optimizer(learning_rate);
  optimizer.attach(params);

  // ==== Loss Function ====
  auto mse_loss = [](const Tensor &y_pred, const Tensor &y_target) -> Tensor {
    MatrixXf diff = y_pred.data - y_target.data;
    float mse = diff.array().square().mean();
    MatrixXf out(1, 1);
    out(0, 0) = mse;
    return {out};
  };

  // ==== Training Loop ====
  float epoch_loss = std::numeric_limits<float>::max();

  for (size_t epoch = 0; epoch < epochs; ++epoch) {

    // Create batches
    auto batches = create_batches(inputs, targets, batch_size);

    for (const auto &batch : batches) {
      // Forward pass
      Tensor y_pred = model.forward(batch.inputs);

      // Compute loss
      Tensor loss_tensor = mse_loss(y_pred, batch.targets);

      // Backward pass (dL/dy_pred = 2*(y_pred - y_true)/N)
      MatrixXf grad = 2.0F * (y_pred.data - batch.targets.data) / y_pred.data.size();
      Tensor grad_loss(grad);
      model.backward(grad_loss);

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
    if (epoch % 100 == 0) {
      cout << "Epoch: " << epoch << " - Loss: " << epoch_loss << "\n";
    }
  }

  // ==== End of Training ====
  cout << "Training complete. Final loss: " << epoch_loss << "\n";
  return 0;
}
