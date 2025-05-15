#ifndef TENSOR
#define TENSOR

#include <Eigen/Dense> // inclui a biblioteca Eigen para operações matriciais

struct Tensor {
  Eigen::MatrixXf data;
  Eigen::MatrixXf grad;

  Tensor(int rows, int cols) : data(rows, cols), grad(rows, cols) {
    data.setZero();
    grad.setZero();
  }

  Tensor(const Eigen::MatrixXf &data)
      : data(data), grad(Eigen::MatrixXf::Zero(data.rows(), data.cols())) {}
};

#endif