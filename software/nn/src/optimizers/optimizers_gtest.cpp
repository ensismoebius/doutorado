#include "../tensor/Tensor.hpp"
#include "Adam.hpp"
#include "SGD.hpp"
#include "SGDMinimal.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(SGDMinimalOptimizerTest, StepAndZeroGrad) {
  Tensor w(Eigen::MatrixXf::Ones(2, 2));
  Tensor b(Eigen::MatrixXf::Zero(2, 1));
  std::vector<Tensor *> params = {&w, &b};
  SGDMinimal sgd_minimal(0.01F);
  w.grad = Eigen::MatrixXf::Ones(2, 2);
  b.grad = Eigen::MatrixXf::Ones(2, 1);
  sgd_minimal.step(params);
  ASSERT_NE(w.data(0, 0), 1.0F);
  sgd_minimal.zero_grad(params);
  ASSERT_EQ(w.grad(0, 0), 0.0F);
}

TEST(AdamOptimizerTest, StepAndZeroGrad) {
  Tensor weights(Eigen::MatrixXf::Ones(2, 2));
  Tensor bias(Eigen::MatrixXf::Zero(2, 1));
  std::vector<Tensor *> params = {&weights, &bias};
  Adam adam(0.01F);
  adam.attach(params);
  weights.grad = Eigen::MatrixXf::Ones(2, 2);
  bias.grad = Eigen::MatrixXf::Ones(2, 1);
  adam.step(params);
  ASSERT_NE(weights.data(0, 0), 1.0F);
  adam.zero_grad(params);
  ASSERT_EQ(weights.grad(0, 0), 0.0F);
}

TEST(SGDOptimizerTest, StepAndZeroGrad) {
  Tensor weights(Eigen::MatrixXf::Ones(2, 2));
  Tensor bias(Eigen::MatrixXf::Zero(2, 1));
  std::vector<Tensor *> params = {&weights, &bias};
  SGD sgd(0.01F);
  sgd.attach(params);
  weights.grad = Eigen::MatrixXf::Ones(2, 2);
  bias.grad = Eigen::MatrixXf::Ones(2, 1);
  sgd.step(params);
  ASSERT_NE(weights.data(0, 0), 1.0F);
  sgd.zero_grad(params);
  ASSERT_EQ(weights.grad(0, 0), 0.0F);
}
