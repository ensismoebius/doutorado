#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "core/tensor/Tensor.hpp"

TEST(TensorTest, ConstructionAndAssignment)
{
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(2, 2);
    nn::Tensor t1(mat);
    nn::Tensor t2 = t1;
    ASSERT_EQ(t1.get_data_ref().rows(), 2);
    ASSERT_EQ(t1.get_data_ref().cols(), 2);
    ASSERT_EQ(t1.get_grad_ref().rows(), 2);
    ASSERT_EQ(t1.get_grad_ref().cols(), 2);
    ASSERT_EQ(t1.get_grad_ref().sum(), 0);
    ASSERT_EQ(t1.get_data_ref(), t2.get_data_ref());
    t2.get_data_ref()(0, 0) += 1.0F;
    ASSERT_NE(t1.get_data_ref(), t2.get_data_ref());
}

TEST(TensorTest, ConstructionWithDimensions)
{
    nn::Tensor t(2, 3);
    ASSERT_EQ(t.get_data_ref().rows(), 2);
    ASSERT_EQ(t.get_data_ref().cols(), 3);
    ASSERT_EQ(t.get_grad_ref().rows(), 2);
    ASSERT_EQ(t.get_grad_ref().cols(), 3);
    ASSERT_EQ(t.get_data_ref().sum(), 0);
    ASSERT_EQ(t.get_grad_ref().sum(), 0);
}