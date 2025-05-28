#include "Adam.hpp"
#include "SGD.hpp"
#include "../tensor/Tensor.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(AdamOptimizerTest, StepAndZeroGrad) {
  Tensor w(Eigen::MatrixXf::Ones(2, 2));
  Tensor b(Eigen::MatrixXf::Zero(2, 1));
  std::vector<Tensor *> params = {&w, &b};
  Adam adam(0.01F);
  adam.attach(params);
  w.grad = Eigen::MatrixXf::Ones(2, 2);
  b.grad = Eigen::MatrixXf::Ones(2, 1);
  adam.step(params);
  ASSERT_NE(w.data(0, 0), 1.0F);
  adam.zero_grad(params);
  ASSERT_EQ(w.grad(0, 0), 0.0F);
}

TEST(SGDOptimizerTest, StepAndZeroGrad) {
  Tensor w(Eigen::MatrixXf::Ones(2, 2));
  Tensor b(Eigen::MatrixXf::Zero(2, 1));
  std::vector<Tensor *> params = {&w, &b};
  SGD sgd(0.01F);
  sgd.attach(params);
  w.grad = Eigen::MatrixXf::Ones(2, 2);
  b.grad = Eigen::MatrixXf::Ones(2, 1);
  sgd.step(params);
  ASSERT_NE(w.data(0, 0), 1.0F);
  sgd.zero_grad(params);
  ASSERT_EQ(w.grad(0, 0), 0.0F);
}
