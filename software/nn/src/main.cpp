#include "layers/Linear.hpp"
#include "layers/ReLU.hpp"
#include "util/batching.hpp"
#include "util/vectorizationCheck.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <ostream>

#define learning_rate 0.01
#define epochs 10000

#define n_amostras 40
#define input_dim 3
#define output_dim 1
#define batch_size 10

namespace {
auto compute_mse_loss(const Tensor &prediction, const Tensor &target) -> float {
  Eigen::MatrixXf diff = prediction.data - target.data;
  return diff.array().square().mean();
}

auto compute_mse_grad(const Tensor &prediction, const Tensor &target) -> Tensor {
  Eigen::MatrixXf grad = 2.0F * (prediction.data - target.data) / prediction.data.rows();
  return {grad};
}
} // namespace

auto main(int /*argc*/, char * /*argv*/[]) -> int {
  printVectorizationSupport();

  Eigen::MatrixXf x_data = Eigen::MatrixXf::Random(n_amostras, input_dim);
  Eigen::MatrixXf y_data = x_data.rowwise().sum();

  Tensor input(x_data);
  Tensor y_target(y_data);

  // Camadas
  Linear linear1(input_dim, 4); // entrada 3 -> escondida 4
  ReLU relu1;

  Linear linear2(4, 1); // escondida 4 -> saída 1
  ReLU relu2;

  // Loss
  float epoch_loss = std::numeric_limits<float>::max();

  for (int epoch = 0; epoch < epochs; ++epoch) {

    auto batches = create_batches(input, y_target, batch_size);

    for (auto &[x_batch, y_batch] : batches) {

      // forward
      Tensor out1 = linear1.forward(x_batch);
      Tensor act1 = relu1.forward(out1);

      Tensor out2 = linear2.forward(act1);
      Tensor y_pred = relu2.forward(out2);

      // Loss
      const auto tmp = compute_mse_loss(y_pred, y_batch);
      epoch_loss = tmp < epoch_loss ? tmp : epoch_loss;

      // Backward
      Tensor grad_loss = compute_mse_grad(y_pred, y_batch);

      Tensor grad_relu2 = relu2.backward(grad_loss);
      Tensor grad_linear2 = linear2.backward(grad_relu2);

      Tensor grad_relu1 = relu1.backward(grad_linear2);
      Tensor grad_linear1 = linear1.backward(grad_relu1);

      // gradient descent
      linear2.weight -= learning_rate * linear2.grad_weight;
      linear2.bias -= learning_rate * linear2.grad_bias;

      linear1.weight -= learning_rate * linear1.grad_weight;
      linear1.bias -= learning_rate * linear1.grad_bias;
    }

    if ((epoch % 100) == 0) {
      std::cout << "Epoch: " << epoch << "-Loss: " << epoch_loss / static_cast<float>(batches.size()) << "\n";
    }
  }

  return 0;
}
