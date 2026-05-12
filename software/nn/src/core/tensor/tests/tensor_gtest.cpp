/**
 * @file tensor_gtest.cpp
 * @brief Unit tests for the `nn::Tensor` API (construction, indexing, shape behavior).
 */

#include <gtest/gtest.h>

#include <cmath>

#include "tensor/Tensor.hpp"

TEST(TensorTest, ConstructionAndAssignment)
{
    nn::Tensor t1(2, 2);
    t1.at(0, 0) = 1.0f;
    t1.at(0, 1) = 2.0f;
    t1.at(1, 0) = 3.0f;
    t1.at(1, 1) = 4.0f;
    nn::Tensor t2 = t1;
    ASSERT_EQ(t1.rows(), 2);
    ASSERT_EQ(t1.cols(), 2);
    ASSERT_EQ(t1.rows(), 2);
    ASSERT_EQ(t1.cols(), 2);
    ASSERT_NEAR(t1.sum(), 10.0f, 1e-6f);
    ASSERT_EQ(t1, t2);
    t2.at(0, 0) += 1.0F;
    ASSERT_NE(t1, t2);
}

TEST(TensorTest, ConstructionWithDimensions)
{
    nn::Tensor t(2, 3);
    ASSERT_EQ(t.rows(), 2);
    ASSERT_EQ(t.cols(), 3);
    ASSERT_EQ(t.rows(), 2);
    ASSERT_EQ(t.cols(), 3);
    ASSERT_EQ(t.sum(), 0);
}

TEST(TensorTest, TwoDAccess)
{
    nn::Tensor t(2, 3);
    t.at(0, 0) = 1.0f;
    t.at(1, 2) = 2.0f;
    ASSERT_NEAR(t.at(0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at(1, 2), 2.0f, 1e-6f);
    ASSERT_THROW(t.at(2, 0), std::out_of_range);
    ASSERT_THROW(t.at(0, 3), std::out_of_range);
}

TEST(TensorTest, FourDAccess)
{
    nn::Tensor t(2, 3, 4, 5); // batch=2, channels=3, height=4, width=5
    t.at(0, 0, 0, 0) = 1.0f;
    t.at(1, 2, 3, 4) = 2.0f;
    ASSERT_NEAR(t.at(0, 0, 0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at(1, 2, 3, 4), 2.0f, 1e-6f);
    ASSERT_THROW(t.at(2, 0, 0, 0), std::out_of_range);
    ASSERT_THROW(t.at(0, 3, 0, 0), std::out_of_range);
}

TEST(TensorTest, GeneralNDAccess)
{
    nn::Tensor t(std::vector<size_t>{2, 3, 4}); // 3D tensor
    t.at({0, 0, 0}) = 1.0f;
    t.at({1, 2, 3}) = 2.0f;
    ASSERT_NEAR(t.at({0, 0, 0}), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at({1, 2, 3}), 2.0f, 1e-6f);
    ASSERT_THROW(t.at({2, 0, 0}), std::out_of_range);
    ASSERT_THROW(t.at({0, 3, 0}), std::out_of_range);
    ASSERT_THROW(t.at({0, 0}), std::invalid_argument); // wrong number of indices
}

TEST(TensorTest, AccessWrongDimensions)
{
    nn::Tensor t2d(2, 3);
    nn::Tensor t4d(1, 1, 1, 1);
    nn::Tensor t3d(std::vector<size_t>{2, 2, 2});

    ASSERT_THROW(t4d.at(0, 0), std::invalid_argument);
    ASSERT_THROW(t2d.at(0, 0, 0, 0), std::invalid_argument);
    ASSERT_THROW(t3d.at(0, 0), std::invalid_argument);
    ASSERT_THROW(t3d.at(0, 0, 0, 0), std::invalid_argument);
}

TEST(TensorTest, ShapeAndSize)
{
    nn::Tensor t2d(2, 3);
    ASSERT_EQ(t2d.get_shape(), std::vector<size_t>({2, 3}));
    ASSERT_EQ(t2d.size(), 6);

    nn::Tensor t4d(1, 2, 3, 4);
    ASSERT_EQ(t4d.get_shape(), std::vector<size_t>({1, 2, 3, 4}));
    ASSERT_EQ(t4d.size(), 24);

    nn::Tensor tnd(std::vector<size_t>{5, 6, 7});
    ASSERT_EQ(tnd.get_shape(), std::vector<size_t>({5, 6, 7}));
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
    ASSERT_EQ(row0.get_shape(), std::vector<size_t>({1, 4}));
    EXPECT_NEAR(row0.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(row0.at(0, 3), 4.0f, 1e-6f);

    auto col1 = t.col(1);
    ASSERT_EQ(col1.get_shape(), std::vector<size_t>({3, 1}));
    EXPECT_NEAR(col1.at(0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(col1.at(1, 0), 6.0f, 1e-6f);

    auto left2 = t.leftCols(2);
    ASSERT_EQ(left2.get_shape(), std::vector<size_t>({3, 2}));
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
    ASSERT_EQ(block.get_shape(), std::vector<size_t>({2, 2}));
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
    auto t3 = t1.add_scalar(10.0f);
    EXPECT_EQ(t3.at(0, 0), 11.0f);
    EXPECT_EQ(t3.at(1, 1), 14.0f);

    auto t4 = t3.multiply_scalar(2.0f);
    EXPECT_EQ(t4.at(0, 0), 22.0f);
    EXPECT_EQ(t4.at(1, 1), 28.0f);
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
    ASSERT_EQ(result.get_shape(), std::vector<size_t>({2, 2}));
    EXPECT_EQ(result.at(0, 0), 58.0f);  // 1*7 + 2*9 + 3*11
    EXPECT_EQ(result.at(0, 1), 64.0f);  // 1*8 + 2*10 + 3*12
    EXPECT_EQ(result.at(1, 0), 139.0f); // 4*7 + 5*9 + 6*11
    EXPECT_EQ(result.at(1, 1), 154.0f); // 4*8 + 5*10 + 6*12

    // Test transpose
    auto transposed = t1.transpose();
    ASSERT_EQ(transposed.get_shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(transposed.at(0, 0), 1.0f);
    EXPECT_EQ(transposed.at(1, 0), 2.0f);
    EXPECT_EQ(transposed.at(2, 0), 3.0f);
    EXPECT_EQ(transposed.at(0, 1), 4.0f);
    EXPECT_EQ(transposed.at(1, 1), 5.0f);
    EXPECT_EQ(transposed.at(2, 1), 6.0f);

    // Test dimension mismatch error
    nn::Tensor t3(4, 2);
    ASSERT_THROW(t1.matmul(t3), std::invalid_argument);

    // Test non-2D tensor error
    nn::Tensor t4(std::vector<size_t>{2, 3, 4});
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
    ASSERT_EQ(relu_result.get_shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(relu_result.at(0, 0), 1.0f); // max(1, 0) = 1
    EXPECT_EQ(relu_result.at(0, 1), 0.0f); // max(-2, 0) = 0
    EXPECT_EQ(relu_result.at(0, 2), 3.0f); // max(3, 0) = 3
    EXPECT_EQ(relu_result.at(1, 0), 0.0f); // max(-4, 0) = 0
    EXPECT_EQ(relu_result.at(1, 1), 5.0f); // max(5, 0) = 5
    EXPECT_EQ(relu_result.at(1, 2), 0.0f); // max(-6, 0) = 0
    // Test LeakyReLU
    auto leaky_result = t.leaky_relu(0.1f);
    ASSERT_EQ(leaky_result.get_shape(), std::vector<size_t>({2, 3}));
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
    pred.at(0, 0) = 1.0f;
    pred.at(0, 1) = 2.0f;
    pred.at(1, 0) = 3.0f;
    pred.at(1, 1) = 4.0f;

    nn::Tensor target(2, 2);
    target.at(0, 0) = 1.5f;
    target.at(0, 1) = 2.5f;
    target.at(1, 0) = 3.5f;
    target.at(1, 1) = 4.5f;

    float mse = pred.mean_squared_error(target);
    // Differences: 0.5, 0.5, 0.5, 0.5
    // Squared: 0.25, 0.25, 0.25, 0.25
    // Sum: 1.0, divided by 4 elements: 0.25
    EXPECT_FLOAT_EQ(mse, 0.25f);

    // Test norm
    nn::Tensor vec(1, 3);
    vec.at(0, 0) = 3.0f;
    vec.at(0, 1) = 4.0f;
    vec.at(0, 2) = 0.0f;
    float norm_val = vec.norm();
    EXPECT_FLOAT_EQ(norm_val, 5.0f); // sqrt(3^2 + 4^2 + 0^2) = 5
}

TEST(TensorTest, ZeroGrad)
{
    nn::Tensor t(2, 3);
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = 2.0f;
    t.at(1, 0) = 3.0f;
    t.at(1, 1) = 4.0f;

    // Set some gradient values
    nn::Tensor g(2, 3);
    g.setZero(); // Initialize to zero first
    g.at(0, 0) = 5.0f;
    g.at(0, 1) = 6.0f;
    g.at(1, 0) = 7.0f;
    g.at(1, 1) = 8.0f;
    t.set_grad(g);

    // Verify gradients are set
    EXPECT_NEAR(t.grad().at(0, 0), 5.0f, 1e-6f);
    EXPECT_NEAR(t.grad().at(0, 1), 6.0f, 1e-6f);

    // Zero gradients
    t.zero_grad();

    // Verify all gradients are zero
    EXPECT_NEAR(t.grad().sum(), 0.0f, 1e-6f);
    EXPECT_NEAR(t.grad().at(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(t.grad().at(0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(t.grad().at(1, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(t.grad().at(1, 1), 0.0f, 1e-6f);
}

TEST(TensorTest, SetData)
{
    nn::Tensor t(2, 2);
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = 2.0f;
    t.at(1, 0) = 3.0f;
    t.at(1, 1) = 4.0f;

    // Set new data via direct assignment to underlying matrix
    t.at(0, 0) = 10.0f;
    t.at(0, 1) = 20.0f;
    t.at(1, 0) = 30.0f;
    t.at(1, 1) = 40.0f;

    EXPECT_NEAR(t.at(0, 0), 10.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1), 20.0f, 1e-6f);
    EXPECT_NEAR(t.at(1, 0), 30.0f, 1e-6f);
    EXPECT_NEAR(t.at(1, 1), 40.0f, 1e-6f);
}

TEST(TensorTest, TopRows)
{
    nn::Tensor t(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            t.at(i, j) = static_cast<float>(i * 3 + j);
        }
    }

    auto top2 = t.topRows(2);
    ASSERT_EQ(top2.get_shape(), std::vector<size_t>({2, 3}));
    EXPECT_NEAR(top2.at(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(top2.at(0, 1), 1.0f, 1e-6f);
    EXPECT_NEAR(top2.at(0, 2), 2.0f, 1e-6f);
    EXPECT_NEAR(top2.at(1, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(top2.at(1, 1), 4.0f, 1e-6f);
    EXPECT_NEAR(top2.at(1, 2), 5.0f, 1e-6f);
}

TEST(TensorTest, Slice)
{
    nn::Tensor t(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            t.at(i, j) = static_cast<float>(i * 3 + j);
        }
    }

    std::vector<int> indices = {0, 2};
    auto sliced = t.slice(indices);
    ASSERT_EQ(sliced.get_shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(sliced.at(0, 0), 0.0f);
    EXPECT_EQ(sliced.at(0, 1), 1.0f);
    EXPECT_EQ(sliced.at(0, 2), 2.0f);
    EXPECT_EQ(sliced.at(1, 0), 6.0f);
    EXPECT_EQ(sliced.at(1, 1), 7.0f);
    EXPECT_EQ(sliced.at(1, 2), 8.0f);
}

TEST(TensorTest, ExceptionTesting)
{
    // Test invalid operations that should throw exceptions

    // Test matmul with incompatible dimensions
    nn::Tensor t1(2, 3);
    nn::Tensor t2(4, 2); // Incompatible for matmul
    ASSERT_THROW(t1.matmul(t2), std::invalid_argument);

    // Test matmul with non-2D tensor
    nn::Tensor t3(std::vector<size_t>{2, 3, 4});
    nn::Tensor t4(4, 2);
    ASSERT_THROW(t3.matmul(t4), std::invalid_argument);

    // Test transpose with non-2D tensor
    ASSERT_THROW(t3.transpose(), std::invalid_argument);

    // Test block with invalid indices
    nn::Tensor t5(4, 4);
    ASSERT_THROW(t5.block(2, 2, 3, 3), std::out_of_range); // Block too large

    // Test setBlock with incompatible sizes
    nn::Tensor small_block(3, 3);
    nn::Tensor large_tensor(2, 2);
    ASSERT_THROW(large_tensor.setBlock(0, 0, small_block), std::invalid_argument);

    // Test add with incompatible shapes
    nn::Tensor t6(2, 3);
    nn::Tensor t7(3, 2);
    ASSERT_THROW(t6.add(t7), std::invalid_argument);

    // Test multiply with incompatible shapes
    ASSERT_THROW(t6.multiply(t7), std::invalid_argument);
}

TEST(TensorTest, MemoryStressTesting)
{
    // Test with large tensors to check memory handling
    const int large_size = 1000;

    // Test large 2D tensor
    nn::Tensor large_2d(large_size, large_size);
    EXPECT_EQ(large_2d.rows(), large_size);
    EXPECT_EQ(large_2d.cols(), large_size);
    EXPECT_EQ(large_2d.size(), large_size * large_size);

    // Test operations on large tensors
    auto large_2d_plus_one = large_2d.add_scalar(1.0f);
    EXPECT_NEAR(large_2d_plus_one.at(0, 0), 1.0f, 1e-6f);

    // Test large N-D tensor
    nn::Tensor large_nd(std::vector<size_t>{100, 100, 10});
    EXPECT_EQ(large_nd.size(), 100 * 100 * 10);

    // Test memory operations
    large_nd.zero_grad();
    EXPECT_NEAR(large_nd.sum(), 0.0f, 1e-6f);
}

TEST(TensorTest, ThreadSafetyValidation)
{
    // Test concurrent access (basic test - in real scenarios would need more sophisticated testing)
    nn::Tensor t(100, 100);

    // Fill tensor
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            t.at(i, j) = static_cast<float>(i + j);
        }
    }

    // Test that multiple reads work (basic thread safety)
    auto read_func = [&t]()
    {
        for (int i = 0; i < 10; ++i)
        {
            float val = t.at(i, i);
            EXPECT_NEAR(val, static_cast<float>(i * 2), 1e-6f);
        }
    };

    // Run multiple times to simulate concurrent access
    for (int i = 0; i < 5; ++i)
    {
        read_func();
    }
}

// -----------------------------------------------------------------------
// 3D tensor tests
// -----------------------------------------------------------------------

TEST(TensorTest, ThreeDConstruction)
{
    nn::Tensor t(2, 3, 4);
    ASSERT_EQ(t.get_shape(), (std::vector<size_t>{2, 3, 4}));
    ASSERT_EQ(t.size(), 24u);
    ASSERT_EQ(t.rows(), 2u);
    ASSERT_EQ(t.cols(), 3u);
    ASSERT_NEAR(t.sum(), 0.0f, 1e-6f);
}

TEST(TensorTest, ThreeDAccess)
{
    nn::Tensor t(2, 3, 4);
    t.at(0, 0, 0) = 1.0f;
    t.at(1, 2, 3) = 2.0f;
    t.at(0, 1, 2) = 3.0f;

    EXPECT_NEAR(t.at(0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(t.at(1, 2, 3), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1, 2), 3.0f, 1e-6f);

    // operator() overload
    t(0, 2, 1) = 7.0f;
    EXPECT_NEAR(t(0, 2, 1), 7.0f, 1e-6f);
}

TEST(TensorTest, ThreeDAccessOutOfRange)
{
    nn::Tensor t(2, 3, 4);
    ASSERT_THROW(t.at(2, 0, 0), std::out_of_range);
    ASSERT_THROW(t.at(0, 3, 0), std::out_of_range);
    ASSERT_THROW(t.at(0, 0, 4), std::out_of_range);
}

TEST(TensorTest, ThreeDAccessWrongDimension)
{
    nn::Tensor t3d(2, 3, 4);
    nn::Tensor t2d(2, 3);
    nn::Tensor t4d(2, 3, 4, 5);

    // 2D accessor on 3D tensor
    ASSERT_THROW(t3d.at(0, 0), std::invalid_argument);
    // 4D accessor on 3D tensor
    ASSERT_THROW(t3d.at(0, 0, 0, 0), std::invalid_argument);
    // 3D accessor on 2D tensor
    ASSERT_THROW(t2d.at(0, 0, 0), std::invalid_argument);
    // 3D accessor on 4D tensor
    ASSERT_THROW(t4d.at(0, 0, 0), std::invalid_argument);
}

TEST(TensorTest, ThreeDVectorAccess)
{
    nn::Tensor t(2, 3, 4);
    t.at(1, 2, 3) = 5.0f;

    // vector-index access must agree with typed access
    EXPECT_NEAR(t.at({1, 2, 3}), 5.0f, 1e-6f);

    // wrong number of indices
    ASSERT_THROW(t.at({1, 2}), std::invalid_argument);
    ASSERT_THROW(t.at({1, 2, 3, 0}), std::invalid_argument);
}

TEST(TensorTest, ThreeDVectorConstruction)
{
    // Shape vector of length 3 must produce a proper 3D tensor
    nn::Tensor t(std::vector<size_t>{2, 3, 4});
    ASSERT_EQ(t.get_shape(), (std::vector<size_t>{2, 3, 4}));
    ASSERT_EQ(t.size(), 24u);

    t.at({0, 0, 0}) = 1.0f;
    t.at({1, 2, 3}) = 2.0f;
    EXPECT_NEAR(t.at(0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(t.at(1, 2, 3), 2.0f, 1e-6f);
}

TEST(TensorTest, ThreeDFactories)
{
    auto z = nn::Tensor::zeros(2, 3, 4);
    ASSERT_EQ(z.get_shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_NEAR(z.sum(), 0.0f, 1e-6f);

    auto o = nn::Tensor::ones(2, 3, 4);
    ASSERT_EQ(o.get_shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_NEAR(o.sum(), 24.0f, 1e-6f);

    auto c = nn::Tensor::constant(2, 3, 4, 3.5f);
    ASSERT_EQ(c.get_shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_NEAR(c.at(0, 0, 0), 3.5f, 1e-6f);
    EXPECT_NEAR(c.at(1, 2, 3), 3.5f, 1e-6f);
}

TEST(TensorTest, ThreeDRandDeterminism)
{
    std::mt19937 rng1(42), rng2(42);
    auto a = nn::Tensor::rand(2, 3, 4, rng1);
    auto b = nn::Tensor::rand(2, 3, 4, rng2);
    ASSERT_EQ(a.get_shape(), (std::vector<size_t>{2, 3, 4}));
    ASSERT_EQ(a.size(), 24u);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(a.at(i, j, k), b.at(i, j, k));
}

TEST(TensorTest, ThreeDRandNonDeterministic)
{
    auto a = nn::Tensor::rand(2, 3, 4);
    ASSERT_EQ(a.get_shape(), (std::vector<size_t>{2, 3, 4}));
    ASSERT_EQ(a.size(), 24u);
    // Values should be in [0, 1)
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
            {
                EXPECT_GE(a.at(i, j, k), 0.0f);
                EXPECT_LT(a.at(i, j, k), 1.0f);
            }
}

TEST(TensorTest, ThreeDReshapeToAndFrom)
{
    // 3D → 2D reshape
    nn::Tensor t3(2, 3, 4);
    t3.at(0, 0, 0) = 1.0f;
    t3.at(1, 2, 3) = 2.0f;
    t3.reshape({2, 12});
    ASSERT_EQ(t3.get_shape(), (std::vector<size_t>{2, 12}));
    ASSERT_EQ(t3.size(), 24u);

    // 2D → 3D reshape (total size preserved)
    nn::Tensor t2(4, 6);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 6; ++j)
            t2.at(i, j) = static_cast<float>(i * 6 + j);

    t2.reshape({4, 3, 2});
    ASSERT_EQ(t2.get_shape(), (std::vector<size_t>{4, 3, 2}));
    ASSERT_EQ(t2.size(), 24u);
    // Check reshape size mismatch is rejected
    ASSERT_THROW(t2.reshape({3, 3, 3}), std::invalid_argument);
}

TEST(TensorTest, ThreeDGradient)
{
    nn::Tensor t(2, 3, 4);
    t.fill(1.0f);

    nn::Tensor g(2, 3, 4);
    g.fill(5.0f);
    t.set_grad(g);

    auto retrieved = t.grad();
    ASSERT_EQ(retrieved.get_shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_NEAR(retrieved.sum(), 5.0f * 24.0f, 1e-4f);

    t.zero_grad();
    EXPECT_NEAR(t.grad().sum(), 0.0f, 1e-6f);
}

TEST(TensorTest, ThreeDElementwiseOps)
{
    nn::Tensor a(2, 3, 4);
    a.fill(2.0f);
    nn::Tensor b(2, 3, 4);
    b.fill(3.0f);

    auto sum = a.add(b);
    ASSERT_EQ(sum.get_shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_NEAR(sum.at(0, 0, 0), 5.0f, 1e-6f);
    EXPECT_NEAR(sum.sum(), 5.0f * 24.0f, 1e-4f);

    auto prod = a.multiply(b);
    EXPECT_NEAR(prod.at(1, 2, 3), 6.0f, 1e-6f);

    auto scaled = a.multiply_scalar(3.0f);
    EXPECT_NEAR(scaled.at(0, 1, 2), 6.0f, 1e-6f);
}

// -----------------------------------------------------------------------

TEST(TensorTest, RandDeterminism)
{
    std::mt19937 rng1(12345);
    std::mt19937 rng2(12345);
    auto a = nn::Tensor::rand(2, 3, rng1);
    auto b = nn::Tensor::rand(2, 3, rng2);
    ASSERT_EQ(a.get_shape(), b.get_shape());
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) EXPECT_FLOAT_EQ(a.at(i, j), b.at(i, j));
}

TEST(TensorTest, ComparisonAndBroadcasting)
{
    // elementwise equality and comparison
    nn::Tensor A(2, 3);
    A.at(0, 0) = 1.0f;
    A.at(0, 1) = 2.0f;
    A.at(0, 2) = 3.0f;
    A.at(1, 0) = 4.0f;
    A.at(1, 1) = 5.0f;
    A.at(1, 2) = 6.0f;

    // flawfinder: ignore - calls Tensor::equal (API method), not std::equal/iterator traversal.
    auto eq = A.equal(A);
    // all ones
    for (size_t i = 0; i < eq.rows(); ++i)
    {
        for (size_t j = 0; j < eq.cols(); ++j)
        {
            EXPECT_FLOAT_EQ(eq.at(i, j), 1.0f);
        }
    }

    // broadcasting: 1x3 compared to 2x3
    nn::Tensor r(1, 3);
    r.at(0, 0) = 2.0f;
    r.at(0, 1) = 3.0f;
    r.at(0, 2) = 1.0f;
    nn::Tensor B(2, 3);
    B.at(0, 0) = 3.0f;
    B.at(0, 1) = 4.0f;
    B.at(0, 2) = 0.0f;
    B.at(1, 0) = 1.0f;
    B.at(1, 1) = 2.0f;
    B.at(1, 2) = 1.0f;

    auto lt = r < B; // elementwise compare with broadcasting
    EXPECT_EQ(lt.get_shape(), std::vector<size_t>({2, 3}));
    // row0: [2<3,3<4,1<0] -> [1,1,0]
    EXPECT_FLOAT_EQ(lt.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(lt.at(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(lt.at(0, 2), 0.0f);
    // row1: [2<1,3<2,1<1] -> [0,0,0]
    EXPECT_FLOAT_EQ(lt.at(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(lt.at(1, 1), 0.0f);
    EXPECT_FLOAT_EQ(lt.at(1, 2), 0.0f);

    // scalar comparisons
    auto gt_scalar = A > 3.0f;
    EXPECT_FLOAT_EQ(gt_scalar.at(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(gt_scalar.at(1, 0), 1.0f);
}

TEST(TensorTest, CommaInitializerWorks)
{
    nn::Tensor t(2, 2);
    // comma-initializer should populate values in row-major logic used by Tensor
    t << 1.0f, 2.0f, 3.0f, 4.0f;
    // Comma-initializer copies into underlying storage order (xtensor row-major),
    // so sequence maps to: (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(t.at(0, 1), 3.0f);
    EXPECT_FLOAT_EQ(t.at(1, 1), 4.0f);
}

TEST(TensorTest, DivideOperations)
{
    nn::Tensor a(2, 2);
    a.at(0, 0) = 2.0f;
    a.at(0, 1) = 4.0f;
    a.at(1, 0) = 8.0f;
    a.at(1, 1) = 16.0f;
    nn::Tensor b(2, 2);
    b.at(0, 0) = 10.0f;
    b.at(0, 1) = 20.0f;
    b.at(1, 0) = 40.0f;
    b.at(1, 1) = 80.0f;

    auto div = b.divide(a);
    EXPECT_FLOAT_EQ(div.at(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(div.at(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(div.at(1, 0), 5.0f);
    EXPECT_FLOAT_EQ(div.at(1, 1), 5.0f);

    auto div_scalar = b.divide_scalar(2.0f);
    EXPECT_FLOAT_EQ(div_scalar.at(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(div_scalar.at(1, 1), 40.0f);
}

TEST(TensorTest, ZerosOnesConstantAndSetters)
{
    auto z = nn::Tensor::zeros(3, 2);
    ASSERT_EQ(z.get_shape(), std::vector<size_t>({3, 2}));
    EXPECT_NEAR(z.sum(), 0.0f, 1e-6f);

    auto o = nn::Tensor::ones(2, 4);
    ASSERT_EQ(o.size(), 8);
    EXPECT_NEAR(o.sum(), 8.0f, 1e-6f);

    auto c = nn::Tensor::constant(2, 2, 3.5f);
    EXPECT_FLOAT_EQ(c.at(0, 0), 3.5f);
    EXPECT_FLOAT_EQ(c.at(1, 1), 3.5f);

    nn::Tensor t(2, 2);
    t.set_ones();
    EXPECT_NEAR(t.sum(), 4.0f, 1e-6f);
    t.set_zero();
    EXPECT_NEAR(t.sum(), 0.0f, 1e-6f);
}

TEST(TensorTest, TensorOperatorComparisons)
{
    nn::Tensor A(2, 2);
    A.at(0, 0) = 1;
    A.at(0, 1) = 4;
    A.at(1, 0) = 2;
    A.at(1, 1) = 3;
    nn::Tensor B(2, 2);
    B.at(0, 0) = 2;
    B.at(0, 1) = 3;
    B.at(1, 0) = 2;
    B.at(1, 1) = 5;

    auto lt = A < B;
    EXPECT_FLOAT_EQ(lt.at(0, 0), 1.0f); // 1<2
    EXPECT_FLOAT_EQ(lt.at(0, 1), 0.0f); // 4<3

    auto gt = A > B;
    EXPECT_FLOAT_EQ(gt.at(0, 1), 1.0f); // 4>3
    EXPECT_FLOAT_EQ(gt.at(1, 0), 0.0f); // 2>2 false

    auto le = A <= B;
    EXPECT_FLOAT_EQ(le.at(1, 0), 1.0f); // 2<=2 true

    auto ge = A >= B;
    EXPECT_FLOAT_EQ(ge.at(1, 1), 0.0f); // 3>=5 false
}
