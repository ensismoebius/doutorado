#include "kaiming_snn.hpp"
#include "xavier.hpp"
#include "tensor/Tensor.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

// Initializer: kaiming_snn
TEST(InitializerTest, KaimingSNN) {
  Tensor weights(Eigen::MatrixXf::Zero(4, 2));
  Tensor bias(Eigen::MatrixXf::Zero(4, 1));
  kaimingSNNInitializer(2, 4, weights, bias);
  ASSERT_NE(weights.data.sum(), 0.0F);
  ASSERT_EQ(bias.data.sum(), 0.0F);
}

// Initializer: Xavier
TEST(InitializerTest, Xavier) {
  Tensor weights(Eigen::MatrixXf::Zero(4, 2));
  Tensor bias(Eigen::MatrixXf::Zero(4, 1));
  xavierInitializer(2, 4, weights, bias);
  ASSERT_NE(weights.data.sum(), 0.0F);
  ASSERT_NE(bias.data.sum(), 0.0F);
}
