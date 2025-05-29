#include "initializers/kaiming_snn.hpp"
#include "initializers/xavier.hpp"
#include "layers/Leaky.hpp"
#include "layers/Linear.hpp"
#include "layers/MSELoss.hpp"
#include "layers/ReLU.hpp"
#include "layers/Sequential.hpp"
#include "layers/SpikeCountLoss.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/SGD.hpp"
#include "tensor/Tensor.hpp"
#include "util/batching.hpp"
#include "util/synthetic_spike_data.hpp"
#include "util/vectorizationCheck.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>

constexpr float learning_rate = 0.001;
constexpr int n_samples = 10000;
constexpr int epochs = 200;

constexpr int input_dim = 7;
constexpr int output_dim = 1;
constexpr int batch_size = 1;

auto main(int /*argc*/, char * /*argv*/[]) -> int {
  printVectorizationSupport();

  std::cout << std::fixed << std::setprecision(50);

  // Generate synthetic spike train data using the new utility function
  int const n_steps = 10;      // number of time steps for spike train
  float const max_rate = 1.0F; // maximum firing rate
  float const timeStep = 1.0F; // time step

  // Real-valued input matrix for encoding
  Eigen::MatrixXf real_valued;

  // Generate synthetic spike data
  // This will generate spike trains for n_samples, each with input_dim features over n_steps time steps
  std::vector<Eigen::MatrixXf> spike_trains = generate_synthetic_spike_data(n_samples, input_dim, n_steps, max_rate, timeStep, &real_valued);

  // Sum the spike trains across time steps to create the input data
  // Each spike train is a matrix of shape (n_samples, input_dim)
  Eigen::MatrixXf x_data = Eigen::MatrixXf::Zero(n_samples, input_dim);
  for (const auto &spikes : spike_trains) {
    x_data += spikes;
  }

  // Target: sum of input spikes per sample (regression on spike count)
  Eigen::MatrixXf y_data = x_data.rowwise().sum();

  Tensor const input(x_data);
  Tensor const y_target(y_data);

  // Layers using Sequential
  auto linear1 = std::make_shared<Linear>(input_dim, 4);
  auto leaky1 = std::make_shared<Leaky>();
  auto linear2 = std::make_shared<Linear>(4, output_dim);
  auto leaky2 = std::make_shared<Leaky>();

  Sequential model({linear1, leaky1, linear2, leaky2});

  // Initialization
  kaimingSNNInitializer(input_dim, 4, linear1->weight, linear1->bias);
  kaimingSNNInitializer(4, output_dim, linear2->weight, linear2->bias);

  std::vector<Tensor *> params = {&linear1->weight, &linear1->bias, &linear2->weight, &linear2->bias};

  Adam optimizer(learning_rate);
  optimizer.attach(params);

  // Loss
  float epoch_loss = std::numeric_limits<float>::max();

  // Use SpikeCountLoss from layers
  SpikeCountLoss spike_loss;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    auto batches = create_batches(input, y_target, batch_size);
    optimizer.zero_grad(params); // Zeroing the gradients

// Parallelize batch processing with OpenMP
#pragma omp parallel for schedule(static)
    for (const auto &batch : batches) {
      // forward
      Tensor y_pred = model.forward(batch.inputs);

      // Loss
      spike_loss.set_target(batch.targets);
      Tensor loss_tensor = spike_loss.forward(y_pred);

      // Backward
      Tensor grad_loss = spike_loss.backward(y_pred);
      model.backward(grad_loss);

      // gradient descent
      optimizer.step(params);

      // Update epoch loss
      float tmp = loss_tensor.data(0, 0);
      epoch_loss = tmp < epoch_loss ? tmp : epoch_loss;
    }

    if (epoch % 100 == 0) {
      std::cout << "Epoch: " << epoch << " - Loss: " << epoch_loss / static_cast<float>(batches.size()) << "\n";
    }
  }

  return 0;
}
