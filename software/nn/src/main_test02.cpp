#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>

#include "initializers/kaiming_snn.hpp"
#include "layers/Leaky.hpp"
#include "layers/Linear.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include "util/batching.hpp"
#include "util/synthetic_spike_data.hpp"
#include "util/vectorizationCheck.hpp"

// Define a nice format
using std::vector;

const Eigen::IOFormat CleanFmt(0,                // number of decimals
                               Eigen::Unaligned, // flags
                               ",",              // string between numbers
                               "\n",             // string between rows
                               "|",              // opening bracket
                               "|",              // closing bracket
                               "\n",             // string between matrices
                               "\n"              // closing bracket for the matrix
);

// ==== Configuration ====
constexpr float learning_rate = 0.001;
constexpr int n_samples = 5;  // Number of samples for synthetic data the higher the better
constexpr int epochs = 10000; // Number of training epochs in which n_samples is presented
constexpr int input_dim = 4;
constexpr int bottleneck_dim = 4; // bottleneck layer size
constexpr int batch_size = 5;

// ==== Data Generation ====
auto main(int /*argc*/, char * /*argv*/[]) -> int {
  printVectorizationSupport();
  std::cout << std::fixed << std::setprecision(0);

  // Parameters for synthetic spike train
  const int n_steps = 10;
  const float max_rate = 1.0F;
  const float timeStep = 1.0F;
  float epoch_loss = 0.0F;

  // Generate synthetic spike data
  std::vector<Eigen::MatrixXf> spike_trains =
      generate_synthetic_spike_data(n_samples, input_dim, n_steps, max_rate, timeStep);

  // Create input tensors
  // Target: input itself (auto-encoder reconstruction)
  vector<Tensor> inputs(spike_trains.size());
  vector<Tensor> targets(spike_trains.size());

  for (int i = 0; i < spike_trains.size(); ++i) {
    std::cout << "Matrix:" << spike_trains.at(i).format(CleanFmt);
    // Take only the first timestep for now, as we need to figure out proper time handling
    Eigen::MatrixXf firstTimeStep =
        spike_trains.at(i).block(0, 0, spike_trains.at(i).rows(), input_dim);
    inputs.at(i) = Tensor(firstTimeStep);
    targets.at(i) = Tensor(firstTimeStep);
  }

  // ==== Model Definition ====
  auto encoder = std::make_shared<Linear>(input_dim, bottleneck_dim);
  auto encoder_act = std::make_shared<Leaky>();

  auto decoder = std::make_shared<Linear>(bottleneck_dim, input_dim);
  auto decoder_act = std::make_shared<Leaky>();

  Sequential model({encoder, encoder_act, decoder, decoder_act});

  // ==== Initialization ====
  kaimingSNNInitializer(input_dim, bottleneck_dim, encoder->weight, encoder->bias);
  kaimingSNNInitializer(bottleneck_dim, input_dim, decoder->weight, decoder->bias);

  std::vector<Tensor *> params = {&encoder->weight, &encoder->bias, &decoder->weight,
                                  &decoder->bias};
  Adam optimizer(learning_rate);
  optimizer.attach(params);

  // ==== Loss Function ====
  auto mse_loss = [](const Tensor &y_pred, const Tensor &y_true) -> Tensor {
    Eigen::MatrixXf diff = y_pred.data - y_true.data;
    float mse = diff.array().square().mean();
    Eigen::MatrixXf out(1, 1);
    out(0, 0) = mse;
    return {out};
  };

  // ==== Training Loop ====
  for (size_t epoch = 0; epoch < epochs; ++epoch) {

    // Create batches
    auto batches = create_batches(inputs, targets, batch_size);

    for (const auto &batch : batches) {
      // Forward pass
      std::cout << "Input dimensions: " << batch.inputs.data.rows() << "x"
                << batch.inputs.data.cols() << std::endl;
      Tensor y_pred = model.forward(batch.inputs);
      std::cout << "Output dimensions: " << y_pred.data.rows() << "x" << y_pred.data.cols()
                << std::endl;

      // Compute loss
      Tensor loss_tensor = mse_loss(y_pred, batch.targets);

      // Backward pass (dL/dy_pred = 2*(y_pred - y_true)/N)
      Eigen::MatrixXf grad = 2.0F * (y_pred.data - batch.targets.data) / y_pred.data.size();
      Tensor grad_loss(grad);
      model.backward(grad_loss);

      // Update parameters
      optimizer.step(params);

      // Track best loss in epoch
      float tmp = loss_tensor.data(0, 0);
      epoch_loss = tmp < epoch_loss ? tmp : epoch_loss;
    }

    // Print progress
    if (epoch % 10 == 0) {
      std::cout << "Epoch: " << epoch
                << " - Loss: " << epoch_loss / static_cast<float>(batches.size()) << "\n";
    }
  }

  // ==== End of Training ====
  std::cout << "Training complete. Final loss: "
            << epoch_loss / static_cast<float>(n_samples / batch_size) << "\n";
  return 0;
}
