#include "tensor/Tensor.hpp"
#include "batching.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

// Util: batching
TEST(UtilTest, Batching) {
  Eigen::MatrixXf input_matrix = Eigen::MatrixXf::Random(4, 2);
  Eigen::MatrixXf target_matrix = Eigen::MatrixXf::Random(4, 1);
  Tensor input(input_matrix);
  Tensor target(target_matrix);
  auto batches = create_batches(input, target, 2);
  ASSERT_EQ(batches.size(), 2U);
  ASSERT_EQ(batches[0].inputs.data.rows(), 2);
}
