#ifndef TENSOR
#define TENSOR

#include <Eigen/Dense>
#include <optional>

struct Tensor {
  Eigen::MatrixXf data;
  Eigen::MatrixXf grad;

  // estado auxiliar para backward, ex: usado por ReLU
  std::optional<Eigen::MatrixXf> aux;

  Tensor() = default;

  Tensor(const int rows, const int cols) : data(rows, cols), grad(rows, cols) {
    data.setZero();
    grad.setZero();
  }

  Tensor(const Eigen::MatrixXf &data) : data(data), grad(Eigen::MatrixXf::Zero(data.rows(), data.cols())) {}
};

#endif