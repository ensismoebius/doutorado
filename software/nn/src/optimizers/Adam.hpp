#include "optimizers/Optimizer.hpp"
#include "tensor/Tensor.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <vector>

struct Adam : public Optimizer {
  float lr;
  float beta1;
  float beta2;
  float eps;
  int t;

  std::vector<Eigen::MatrixXf> m;
  std::vector<Eigen::MatrixXf> v;

  explicit Adam(float learning_rate = 0.001F, float beta1 = 0.9F, float beta2 = 0.999F, float eps = 1e-8F) : lr(learning_rate), beta1(beta1), beta2(beta2), eps(eps), t(0) {}

  auto attach(std::vector<Tensor *> &paramsList) -> void {
    m.clear();
    v.clear();
    for (auto *param : paramsList) {
      m.emplace_back(Eigen::MatrixXf::Zero(param->grad.rows(), param->grad.cols()));
      v.emplace_back(Eigen::MatrixXf::Zero(param->grad.rows(), param->grad.cols()));
    }
  }

  auto step(std::vector<Tensor *> &paramsList) -> void override {
    t += 1;
    for (size_t i = 0; i < paramsList.size(); ++i) {
      auto &param = *paramsList[i];
      m[i] = beta1 * m[i] + (1 - beta1) * param.grad;
      v[i] = beta2 * v[i] + (1 - beta2) * param.grad.array().square().matrix();

      Eigen::MatrixXf m_hat = m[i] / (1 - std::pow(beta1, t));
      Eigen::MatrixXf v_hat = v[i] / (1 - std::pow(beta2, t));

      param.data -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
    }
  }

  auto zero_grad(std::vector<Tensor *> &paramsList) -> void override {
    for (auto *param : paramsList) {
      param->grad.setZero();
    }
  }
};
