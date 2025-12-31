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

TEST(TensorTest, RowColAccess)
{
    nn::Tensor t(3, 4);
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = 2.0f;
    t.at(0, 2) = 3.0f;
    t.at(0, 3) = 4.0f;
    t.at(1, 0) = 5.0f;
    t.at(1, 1) = 6.0f;

    auto row0 = t.row(0);
    ASSERT_EQ(row0.get_shape(), std::vector<Eigen::Index>({1, 4}));
    EXPECT_EQ(row0.at(0, 0), 1.0f);
    EXPECT_EQ(row0.at(0, 3), 4.0f);

    auto col1 = t.col(1);
    ASSERT_EQ(col1.get_shape(), std::vector<Eigen::Index>({3, 1}));
    EXPECT_EQ(col1.at(0, 0), 2.0f);
    EXPECT_EQ(col1.at(1, 0), 6.0f);

    auto left2 = t.leftCols(2);
    ASSERT_EQ(left2.get_shape(), std::vector<Eigen::Index>({3, 2}));
    EXPECT_EQ(left2.at(0, 0), 1.0f);
    EXPECT_EQ(left2.at(0, 1), 2.0f);
}

TEST(TensorTest, BlockOperations)
{
    nn::Tensor t(4, 4);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            t.at(i, j) = static_cast<float>(i * 4 + j);
        }
    }

    // Test block extraction
    auto block = t.block(1, 1, 2, 2);
    ASSERT_EQ(block.get_shape(), std::vector<Eigen::Index>({2, 2}));
    EXPECT_EQ(block.at(0, 0), 5.0f);  // (1,1) -> 1*4+1 = 5
    EXPECT_EQ(block.at(1, 1), 10.0f); // (2,2) -> 2*4+2 = 10

    // Test block setting
    nn::Tensor small_block(2, 2);
    small_block.at(0, 0) = 100.0f;
    small_block.at(0, 1) = 101.0f;
    small_block.at(1, 0) = 102.0f;
    small_block.at(1, 1) = 103.0f;

    t.setBlock(1, 1, small_block);
    EXPECT_EQ(t.at(1, 1), 100.0f);
    EXPECT_EQ(t.at(1, 2), 101.0f);
    EXPECT_EQ(t.at(2, 1), 102.0f);
    EXPECT_EQ(t.at(2, 2), 103.0f);
}

TEST(TensorTest, ElementWiseOperations)
{
    nn::Tensor t1(2, 2);
    t1.at(0, 0) = 1.0f;
    t1.at(0, 1) = 2.0f;
    t1.at(1, 0) = 3.0f;
    t1.at(1, 1) = 4.0f;

    nn::Tensor t2(2, 2);
    t2.at(0, 0) = 5.0f;
    t2.at(0, 1) = 6.0f;
    t2.at(1, 0) = 7.0f;
    t2.at(1, 1) = 8.0f;

    // Test add
    auto sum = t1.add(t2);
    EXPECT_EQ(sum.at(0, 0), 6.0f);
    EXPECT_EQ(sum.at(1, 1), 12.0f);

    // Test multiply (element-wise)
    auto prod = t1.multiply(t2);
    EXPECT_EQ(prod.at(0, 0), 5.0f);
    EXPECT_EQ(prod.at(1, 1), 32.0f);

    // Test scalar operations
    t1.add_scalar(10.0f);
    EXPECT_EQ(t1.at(0, 0), 11.0f);
    EXPECT_EQ(t1.at(1, 1), 14.0f);

    t1.multiply_scalar(2.0f);
    EXPECT_EQ(t1.at(0, 0), 22.0f);
    EXPECT_EQ(t1.at(1, 1), 28.0f);
}

TEST(TensorTest, MatrixOperations)
{
    // Test matmul
    nn::Tensor t1(2, 3);
    t1.at(0, 0) = 1.0f;
    t1.at(0, 1) = 2.0f;
    t1.at(0, 2) = 3.0f;
    t1.at(1, 0) = 4.0f;
    t1.at(1, 1) = 5.0f;
    t1.at(1, 2) = 6.0f;

    nn::Tensor t2(3, 2);
    t2.at(0, 0) = 7.0f;
    t2.at(0, 1) = 8.0f;
    t2.at(1, 0) = 9.0f;
    t2.at(1, 1) = 10.0f;
    t2.at(2, 0) = 11.0f;
    t2.at(2, 1) = 12.0f;

    auto result = t1.matmul(t2);
    ASSERT_EQ(result.get_shape(), std::vector<Eigen::Index>({2, 2}));
    EXPECT_EQ(result.at(0, 0), 58.0f);  // 1*7 + 2*9 + 3*11
    EXPECT_EQ(result.at(0, 1), 64.0f);  // 1*8 + 2*10 + 3*12
    EXPECT_EQ(result.at(1, 0), 139.0f); // 4*7 + 5*9 + 6*11
    EXPECT_EQ(result.at(1, 1), 154.0f); // 4*8 + 5*10 + 6*12

    // Test transpose
    auto transposed = t1.transpose();
    ASSERT_EQ(transposed.get_shape(), std::vector<Eigen::Index>({3, 2}));
    EXPECT_EQ(transposed.at(0, 0), 1.0f);
    EXPECT_EQ(transposed.at(1, 0), 2.0f);
    EXPECT_EQ(transposed.at(2, 0), 3.0f);
    EXPECT_EQ(transposed.at(0, 1), 4.0f);
    EXPECT_EQ(transposed.at(1, 1), 5.0f);
    EXPECT_EQ(transposed.at(2, 1), 6.0f);

    // Test dimension mismatch error
    nn::Tensor t3(3, 2);
    ASSERT_THROW(t1.matmul(t3), std::invalid_argument);

    // Test non-2D tensor error
    nn::Tensor t4(std::vector<Eigen::Index>{2, 3, 4});
    ASSERT_THROW(t4.matmul(t2), std::invalid_argument);
    ASSERT_THROW(t4.transpose(), std::invalid_argument);
}

TEST(TensorTest, ActivationFunctions)
{
    // Test ReLU
    nn::Tensor t(2, 3);
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = -2.0f;
    t.at(0, 2) = 3.0f;
    t.at(1, 0) = -4.0f;
    t.at(1, 1) = 5.0f;
    t.at(1, 2) = -6.0f;

    auto relu_result = t.relu();
    ASSERT_EQ(relu_result.get_shape(), std::vector<Eigen::Index>({2, 3}));
    EXPECT_EQ(relu_result.at(0, 0), 1.0f); // max(1, 0) = 1
    EXPECT_EQ(relu_result.at(0, 1), 0.0f); // max(-2, 0) = 0
    EXPECT_EQ(relu_result.at(0, 2), 3.0f); // max(3, 0) = 3
    EXPECT_EQ(relu_result.at(1, 0), 0.0f); // max(-4, 0) = 0
    EXPECT_EQ(relu_result.at(1, 1), 5.0f); // max(5, 0) = 5
    EXPECT_EQ(relu_result.at(1, 2), 0.0f); // max(-6, 0) = 0
    // Test LeakyReLU
    auto leaky_result = t.leaky_relu(0.1f);
    ASSERT_EQ(leaky_result.get_shape(), std::vector<Eigen::Index>({2, 3}));
    EXPECT_EQ(leaky_result.at(0, 0), 1.0f);  // max(1, 0) = 1
    EXPECT_EQ(leaky_result.at(0, 1), -0.2f); // min(-2, 0) * 0.1 = -0.2
    EXPECT_EQ(leaky_result.at(0, 2), 3.0f);  // max(3, 0) = 3
    EXPECT_EQ(leaky_result.at(1, 0), -0.4f); // min(-4, 0) * 0.1 = -0.4
    EXPECT_EQ(leaky_result.at(1, 1), 5.0f);  // max(5, 0) = 5
    EXPECT_EQ(leaky_result.at(1, 2), -0.6f); // min(-6, 0) * 0.1 = -0.6}
}
TEST(TensorTest, LossFunctions)
{
    // Test mean_squared_error
    nn::Tensor pred(2, 2);
    pred.at(0, 0) = 1.0f; pred.at(0, 1) = 2.0f;
    pred.at(1, 0) = 3.0f; pred.at(1, 1) = 4.0f;

    nn::Tensor target(2, 2);
    target.at(0, 0) = 1.5f; target.at(0, 1) = 2.5f;
    target.at(1, 0) = 3.5f; target.at(1, 1) = 4.5f;

    float mse = pred.mean_squared_error(target);
    // Differences: 0.5, 0.5, 0.5, 0.5
    // Squared: 0.25, 0.25, 0.25, 0.25
    // Sum: 1.0, divided by 4 elements: 0.25
    EXPECT_FLOAT_EQ(mse, 0.25f);

    // Test norm
    nn::Tensor vec(1, 3);
    vec.at(0, 0) = 3.0f; vec.at(0, 1) = 4.0f; vec.at(0, 2) = 0.0f;
    float norm_val = vec.norm();
    EXPECT_FLOAT_EQ(norm_val, 5.0f);  // sqrt(3^2 + 4^2 + 0^2) = 5
}
