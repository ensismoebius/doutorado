#include "Tensor.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(TensorTest, ConstructionAndAssignment) {
  Eigen::MatrixXf mat = Eigen::MatrixXf::Random(2, 2);
  Tensor t1(mat);
  Tensor t2 = t1;
  ASSERT_EQ(t1.data.rows(), 2);
  ASSERT_EQ(t1.data.cols(), 2);
  ASSERT_EQ(t1.grad.rows(), 2);
  ASSERT_EQ(t1.grad.cols(), 2);
  ASSERT_EQ(t1.grad.sum(), 0);
  ASSERT_EQ(t1.data, t2.data);
  t2.data(0, 0) += 1.0F;
  ASSERT_NE(t1.data, t2.data);
}

TEST(TensorTest, ConstructionWithDimensions) {
  Tensor t(2, 3);
  ASSERT_EQ(t.data.rows(), 2);
  ASSERT_EQ(t.data.cols(), 3);
  ASSERT_EQ(t.grad.rows(), 2);
  ASSERT_EQ(t.grad.cols(), 3);
  ASSERT_EQ(t.data.sum(), 0);
  ASSERT_EQ(t.grad.sum(), 0);
}