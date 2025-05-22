#ifndef SGD_HPP
#define SGD_HPP

#include "Optimizer.hpp"

struct SGD : public Optimizer {
  float lr;
  float momentum;

  std::vector<Eigen::MatrixXf> velocity;

  explicit SGD(float lr = 0.01f, float momentum = 0.0f) : lr(lr), momentum(momentum) {}

  void attach(std::vector<Tensor *> &paramsList) {
    velocity.clear();
    for (auto *param : paramsList) {
      velocity.emplace_back(Eigen::MatrixXf::Zero(param->grad.rows(), param->grad.cols()));
    }
  }

  void step(std::vector<Tensor *> &paramsList) override {
    for (size_t i = 0; i < paramsList.size(); ++i) {
      auto &param = *paramsList[i];
      velocity[i] = momentum * velocity[i] - lr * param.grad;
      param.data += velocity[i];
    }
  }

  void zero_grad(std::vector<Tensor *> &paramsList) override {
    for (auto *param : paramsList) {
      param->grad.setZero();
    }
  }
};

#endif // SGD_HPP
