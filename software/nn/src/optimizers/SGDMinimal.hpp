#ifndef SGD_MINIMAL_HPP
#define SGD_MINIMAL_HPP

#include "Optimizer.hpp"

struct SGDMinimal : public Optimizer {
  float learning_rate;

  explicit SGDMinimal(float learnningRate = 0.01F) : learning_rate(learnningRate) {}

  void step(std::vector<Tensor *> &params) override {
    for (Tensor *param : params) {
      param->data -= learning_rate * param->grad;
    }
  }

  void zero_grad(std::vector<Tensor *> &params) override {
    for (Tensor *param : params) {
      param->grad.setZero();
    }
  }
};

#endif // SGD_MINIMAL_HPP
