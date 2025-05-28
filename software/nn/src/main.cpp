#include "initializers/kaiming_snn.hpp"
#include "initializers/xavier.hpp"
#include "layers/Leaky.hpp"
#include "layers/Linear.hpp"
#include "layers/MSELoss.hpp"
#include "layers/ReLU.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/SGD.hpp"
#include "tensor/Tensor.hpp"
#include "util/batching.hpp"
#include "util/vectorizationCheck.hpp"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>

constexpr float learning_rate = 0.001;
constexpr int epochs = 1000000;
constexpr int n_samples = 4;
constexpr int input_dim = 2;
constexpr int output_dim = 1;
constexpr int batch_size = 1;

auto main(int /*argc*/, char * /*argv*/[]) -> int {
  printVectorizationSupport();

  Eigen::MatrixXf x_data = Eigen::MatrixXf::Random(n_samples, input_dim);
  Eigen::MatrixXf const y_data = x_data.rowwise().sum();

  Tensor const input(x_data);
  Tensor const y_target(y_data);

  // Layers using Sequential
  auto linear1 = std::make_shared<Linear>(input_dim, output_dim);
  auto relu1 = std::make_shared<ReLU>();

  Sequential model({linear1, relu1});

  // Initialization
  kaimingSNNInitializer(input_dim, output_dim, linear1->weight, linear1->bias);

  std::vector<Tensor *> params = {&linear1->weight, &linear1->bias};

  Adam optimizer(learning_rate);
  optimizer.attach(params);

  // Loss
  float epoch_loss = std::numeric_limits<float>::max();
  std::cout << std::fixed << std::setprecision(50);

  MSELoss mse_loss;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    auto batches = create_batches(input, y_target, batch_size);
    optimizer.zero_grad(params); // Zera os gradientes

    for (const auto &batch : batches) {
      // forward
      Tensor const y_pred = model.forward(batch.inputs);

      // Loss
      mse_loss.set_target(batch.targets);
      Tensor loss_tensor = mse_loss.forward(y_pred);
      float tmp = loss_tensor.data(0, 0);
      epoch_loss = tmp < epoch_loss ? tmp : epoch_loss;

      // Backward
      Tensor const grad_loss = mse_loss.backward(y_pred);
      model.backward(grad_loss);

      // gradient descent
      optimizer.step(params);
    }

    if (epoch % 500 == 0) {
      std::cout << "Epoch: " << epoch << "-Loss: " << epoch_loss / static_cast<float>(batches.size()) << "\n";
    }
  }

  return 0;
}
