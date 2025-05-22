#include "layers/Linear.hpp"
#include "layers/ReLU.hpp"
#include "optimizers/SGDMinimal.hpp"
#include "tensor/Tensor.hpp"
#include "util/batching.hpp"
#include "util/vectorizationCheck.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <ostream>
#include <vector>

constexpr float learning_rate = 0.01;
constexpr int epochs = 1000000;
constexpr int n_samples = 4;
constexpr int input_dim = 1;
constexpr int output_dim = 1;
constexpr int batch_size = 1;

namespace {
auto compute_mse_loss(const Tensor &prediction, const Tensor &target) -> float {
  Eigen::MatrixXf diff = prediction.data - target.data;
  return diff.array().square().mean();
}

auto compute_mse_grad(const Tensor &prediction, const Tensor &target) -> Tensor {
  Eigen::MatrixXf const grad = 2.0F * (prediction.data - target.data) / prediction.data.rows();
  return {grad};
}
} // namespace

auto main(int /*argc*/, char * /*argv*/[]) -> int {
  printVectorizationSupport();

  SGDMinimal optmizer(0.01F);

  Eigen::MatrixXf x_data = Eigen::MatrixXf::Random(n_samples, input_dim);
  Eigen::MatrixXf const y_data = x_data.rowwise().sum();

  Tensor const input(x_data);
  Tensor const y_target(y_data);

  // Layers
  Linear linear1(input_dim, output_dim); // entrada 3 -> escondida 4
  ReLU relu1;

  std::vector<Tensor *> params = {&linear1.weight, &linear1.bias};

  // Loss
  float epoch_loss = std::numeric_limits<float>::max();

  for (size_t epoch = 0; epoch < epochs; ++epoch) {

    auto batches = create_batches(input, y_target, batch_size);

    for (const auto &batch : batches) {

      // forward
      Tensor const out1 = linear1.forward(batch.inputs);
      Tensor const y_pred = relu1.forward(out1);

      // Loss
      const auto tmp = compute_mse_loss(y_pred, batch.targets);
      epoch_loss = tmp < epoch_loss ? tmp : epoch_loss;

      // Backward
      Tensor const grad_loss = compute_mse_grad(y_pred, batch.targets);

      Tensor const grad_relu1 = relu1.backward(grad_loss);
      Tensor const grad_linear1 = linear1.backward(grad_relu1);

      // gradient descent
      linear1.weight.data -= learning_rate * linear1.weight.grad;
      linear1.bias.data -= learning_rate * linear1.bias.grad;
    }

    if (epoch % 100 == 0) {
      std::cout << "Epoch: " << epoch << "-Loss: " << epoch_loss / static_cast<float>(batches.size()) << "\n";
    }
  }

  return 0;
}
