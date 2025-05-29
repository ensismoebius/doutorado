#ifndef TENSOR
#define TENSOR

#include <Eigen/Dense>
#include <optional>

struct Tensor {
  Eigen::MatrixXf data;
  Eigen::MatrixXf grad;

  Tensor() = default;

  Tensor(const int rows, const int cols) : data(rows, cols), grad(rows, cols) {
    data.setZero();
    grad.setZero();
  }

  Tensor(const Eigen::MatrixXf &data) : data(data), grad(Eigen::MatrixXf::Zero(data.rows(), data.cols())) {}
};

#endif