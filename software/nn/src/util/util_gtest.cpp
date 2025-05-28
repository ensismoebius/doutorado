#include "batching.hpp"
#include "tensor/Tensor.hpp"
#include "vectorizationCheck.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

// Util: vectorizationCheck
TEST(UtilTest, VectorizationCheck) {
  ASSERT_NO_THROW(printVectorizationSupport());
}

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
