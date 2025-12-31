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

TEST(TensorTest, TwoDAccess)
{
    nn::Tensor t(2, 3);
    t.at(0, 0) = 1.0f;
    t.at(1, 2) = 2.0f;
    ASSERT_EQ(t.at(0, 0), 1.0f);
    ASSERT_EQ(t.at(1, 2), 2.0f);
    ASSERT_THROW(t.at(2, 0), std::out_of_range);
    ASSERT_THROW(t.at(0, 3), std::out_of_range);
}

TEST(TensorTest, FourDAccess)
{
    nn::Tensor t(2, 3, 4, 5); // batch=2, channels=3, height=4, width=5
    t.at(0, 0, 0, 0) = 1.0f;
    t.at(1, 2, 3, 4) = 2.0f;
    ASSERT_EQ(t.at(0, 0, 0, 0), 1.0f);
    ASSERT_EQ(t.at(1, 2, 3, 4), 2.0f);
    ASSERT_THROW(t.at(2, 0, 0, 0), std::out_of_range);
    ASSERT_THROW(t.at(0, 3, 0, 0), std::out_of_range);
}

TEST(TensorTest, GeneralNDAccess)
{
    nn::Tensor t(std::vector<Eigen::Index>{2, 3, 4}); // 3D tensor
    t.at({0, 0, 0}) = 1.0f;
    t.at({1, 2, 3}) = 2.0f;
    ASSERT_EQ(t.at({0, 0, 0}), 1.0f);
    ASSERT_EQ(t.at({1, 2, 3}), 2.0f);
    ASSERT_THROW(t.at({2, 0, 0}), std::out_of_range);
    ASSERT_THROW(t.at({0, 3, 0}), std::out_of_range);
    ASSERT_THROW(t.at({0, 0}), std::invalid_argument); // wrong number of indices
}

TEST(TensorTest, AccessWrongDimensions)
{
    nn::Tensor t2d(2, 3);
    nn::Tensor t4d(1, 1, 1, 1);
    nn::Tensor t3d(std::vector<Eigen::Index>{2, 2, 2});

    ASSERT_THROW(t4d.at(0, 0), std::invalid_argument);
    ASSERT_THROW(t2d.at(0, 0, 0, 0), std::invalid_argument);
    ASSERT_THROW(t3d.at(0, 0), std::invalid_argument);
    ASSERT_THROW(t3d.at(0, 0, 0, 0), std::invalid_argument);
}

TEST(TensorTest, ShapeAndSize)
{
    nn::Tensor t2d(2, 3);
    ASSERT_EQ(t2d.get_shape(), std::vector<Eigen::Index>({2, 3}));
    ASSERT_EQ(t2d.size(), 6);

    nn::Tensor t4d(1, 2, 3, 4);
    ASSERT_EQ(t4d.get_shape(), std::vector<Eigen::Index>({1, 2, 3, 4}));
    ASSERT_EQ(t4d.size(), 24);

    nn::Tensor tnd(std::vector<Eigen::Index>{5, 6, 7});
    ASSERT_EQ(tnd.get_shape(), std::vector<Eigen::Index>({5, 6, 7}));
    ASSERT_EQ(tnd.size(), 210);
}