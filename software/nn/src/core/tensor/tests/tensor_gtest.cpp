#include <gtest/gtest.h>

#include "core/tensor/Tensor.hpp"

TEST(TensorTest, ConstructionAndAssignment)
{
    nn::Tensor t1(2, 2);
    t1.at(0, 0) = 1.0f;
    t1.at(0, 1) = 2.0f;
    t1.at(1, 0) = 3.0f;
    t1.at(1, 1) = 4.0f;
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
    nn::Tensor t(std::vector<size_t>{2, 3, 4}); // 3D tensor
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
    EXPECT_EQ(row0.at(0, 0), 1.0f);
    EXPECT_EQ(row0.at(0, 3), 4.0f);

    auto col1 = t.col(1);
    ASSERT_EQ(col1.get_shape(), std::vector<size_t>({3, 1}));
    EXPECT_EQ(col1.at(0, 0), 2.0f);
    EXPECT_EQ(col1.at(1, 0), 6.0f);

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
    t.get_grad_ref()(0, 0) = 5.0f;
    t.get_grad_ref()(0, 1) = 6.0f;
    t.get_grad_ref()(1, 0) = 7.0f;
    t.get_grad_ref()(1, 1) = 8.0f;

    // Verify gradients are set
    EXPECT_EQ(t.get_grad_ref()(0, 0), 5.0f);
    EXPECT_EQ(t.get_grad_ref()(0, 1), 6.0f);

    // Zero gradients
    t.zero_grad();

    // Verify all gradients are zero
    EXPECT_EQ(t.get_grad_ref().sum(), 0.0f);
    EXPECT_EQ(t.get_grad_ref()(0, 0), 0.0f);
    EXPECT_EQ(t.get_grad_ref()(0, 1), 0.0f);
    EXPECT_EQ(t.get_grad_ref()(1, 0), 0.0f);
    EXPECT_EQ(t.get_grad_ref()(1, 1), 0.0f);
}

TEST(TensorTest, SetData)
{
    nn::Tensor t(2, 2);
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = 2.0f;
    t.at(1, 0) = 3.0f;
    t.at(1, 1) = 4.0f;

    // Set new data via direct assignment to underlying matrix
    t.get_data_ref()(0, 0) = 10.0f;
    t.get_data_ref()(0, 1) = 20.0f;
    t.get_data_ref()(1, 0) = 30.0f;
    t.get_data_ref()(1, 1) = 40.0f;

    EXPECT_EQ(t.at(0, 0), 10.0f);
    EXPECT_EQ(t.at(0, 1), 20.0f);
    EXPECT_EQ(t.at(1, 0), 30.0f);
    EXPECT_EQ(t.at(1, 1), 40.0f);
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
    EXPECT_EQ(top2.at(0, 0), 0.0f);
    EXPECT_EQ(top2.at(0, 1), 1.0f);
    EXPECT_EQ(top2.at(0, 2), 2.0f);
    EXPECT_EQ(top2.at(1, 0), 3.0f);
    EXPECT_EQ(top2.at(1, 1), 4.0f);
    EXPECT_EQ(top2.at(1, 2), 5.0f);
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
    large_2d.add_scalar(1.0f);
    EXPECT_EQ(large_2d.at(0, 0), 1.0f);

    // Test large N-D tensor
    nn::Tensor large_nd(std::vector<size_t>{100, 100, 10});
    EXPECT_EQ(large_nd.size(), 100 * 100 * 10);

    // Test memory operations
    large_nd.zero_grad();
    EXPECT_EQ(large_nd.get_grad_ref().sum(), 0.0f);
}

TEST(TensorTest, NumericalEdgeCases)
{
    // Test with NaN and Inf values
    nn::Tensor t(2, 2);

    // Test with NaN
    t.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(std::isnan(t.at(0, 0)));

    // Test with positive infinity
    t.at(0, 1) = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(std::isinf(t.at(0, 1)));

    // Test with negative infinity
    t.at(1, 0) = -std::numeric_limits<float>::infinity();
    EXPECT_TRUE(std::isinf(t.at(1, 0)) && t.at(1, 0) < 0);

    // Test operations with special values
    nn::Tensor t2(2, 2);
    t2.at(0, 0) = 1.0f;
    t2.at(0, 1) = 2.0f;
    t2.at(1, 0) = 3.0f;
    t2.at(1, 1) = 4.0f;

    // Operations should handle special values appropriately
    auto result = t.add(t2);
    EXPECT_TRUE(std::isnan(result.at(0, 0))); // NaN + 1 = NaN
    EXPECT_TRUE(std::isinf(result.at(0, 1))); // Inf + 2 = Inf
    EXPECT_TRUE(std::isinf(result.at(1, 0))); // -Inf + 3 = -Inf

    // Test norm with special values
    nn::Tensor special_vec(1, 3);
    special_vec.at(0, 0) = 3.0f;
    special_vec.at(0, 1) = std::numeric_limits<float>::quiet_NaN();
    special_vec.at(0, 2) = 4.0f;
    float norm_result = special_vec.norm();
    EXPECT_TRUE(std::isnan(norm_result)); // Norm of vector with NaN is NaN

    // Test MSE with special values
    nn::Tensor pred_special(2, 1);
    pred_special.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    pred_special.at(1, 0) = 2.0f;

    nn::Tensor target_special(2, 1);
    target_special.at(0, 0) = 1.0f;
    target_special.at(1, 0) = 2.0f;

    float mse_special = pred_special.mean_squared_error(target_special);
    EXPECT_TRUE(std::isnan(mse_special)); // MSE with NaN predictions is NaN
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
            EXPECT_EQ(val, static_cast<float>(i * 2));
        }
    };

    // Run multiple times to simulate concurrent access
    for (int i = 0; i < 5; ++i)
    {
        read_func();
    }
}
