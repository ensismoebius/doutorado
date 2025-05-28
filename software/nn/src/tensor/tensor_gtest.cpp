#include "Tensor.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(TensorTest, ConstructionAndAssignment) {
  Eigen::MatrixXf mat = Eigen::MatrixXf::Random(2, 2);
  Tensor t1(mat);
  Tensor t2 = t1;
  ASSERT_EQ(t1.data, t2.data);
  t2.data(0, 0) += 1.0F;
  ASSERT_NE(t1.data, t2.data);
}
