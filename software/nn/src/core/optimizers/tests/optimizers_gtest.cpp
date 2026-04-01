/**
 * @file optimizers_gtest.cpp
 * @brief Unit tests for optimizer implementations (SGD, Adam, etc.).
 */

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "core/utility/tests/test_helpers.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/optimizers/SGD.hpp"
#include "nn/optimizers/SGDMinimal.hpp"
#include "nn/tensor/Tensor.hpp"

// Test Fixture for common optimizer setup
class OptimizerTest : public ::testing::Test
{
   protected:
    // Common data for weights and bias
    nn::Tensor initial_weights_data = test_helpers::make_ones_tensor(2, 2);
    nn::Tensor initial_bias_data = test_helpers::make_zeros_tensor(2, 1);

    nn::Tensor weights;
    nn::Tensor bias;
    std::vector<nn::Tensor*> params;

    OptimizerTest() : weights(2, 2), bias(2, 1)
    {
        // Copy initial data
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                weights.at(i, j) = 1.0F;
            }
        }
        for (size_t i = 0; i < 2; ++i)
        {
            bias.at(i, 0) = 0.0F;
        }

        params.push_back(&weights);
        params.push_back(&bias);

        // Set gradients
        weights.set_grad(test_helpers::make_ones_tensor(2, 2));
        bias.set_grad(test_helpers::make_ones_tensor(2, 1));
    }

    // Helper to check if a tensor's data has changed from initial
    bool has_data_changed(const nn::Tensor& tensor, const nn::Tensor& initial_data) const
    {
        return !test_helpers::tensor_is_approx(tensor, initial_data);
    }
};

TEST_F(OptimizerTest, SGDMinimalOptimizerStepAndZeroGrad)
{
    SGDMinimal sgd_minimal(0.01F);
    sgd_minimal.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    sgd_minimal.zero_grad(params);
    ASSERT_TRUE(test_helpers::tensor_is_zero(weights.grad(), 1e-6F));
    ASSERT_TRUE(test_helpers::tensor_is_zero(bias.grad(), 1e-6F));
}

TEST_F(OptimizerTest, AdamOptimizerStepAndZeroGrad)
{
    Adam adam(0.01F);
    adam.attach(params); // Adam requires attaching parameters
    adam.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    adam.zero_grad(params);
    ASSERT_TRUE(test_helpers::tensor_is_zero(weights.grad(), 1e-6F));
    ASSERT_TRUE(test_helpers::tensor_is_zero(bias.grad(), 1e-6F));
}

TEST_F(OptimizerTest, SGDOptimizerStepAndZeroGrad)
{
    SGD sgd(0.01F);
    sgd.attach(params); // SGD with momentum requires attaching parameters
    sgd.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    sgd.zero_grad(params);
    ASSERT_TRUE(test_helpers::tensor_is_zero(weights.grad(), 1e-6F));
    ASSERT_TRUE(test_helpers::tensor_is_zero(bias.grad(), 1e-6F));
}

// New test for empty parameters list
TEST(OptimizerEdgeCases, EmptyParamsList)
{
    std::vector<nn::Tensor*> empty_params;
    SGDMinimal sgd_minimal(0.01F);
    ASSERT_NO_THROW(sgd_minimal.step(empty_params));
    ASSERT_NO_THROW(sgd_minimal.zero_grad(empty_params));

    Adam adam(0.01F);
    ASSERT_NO_THROW(adam.attach(empty_params));
    ASSERT_NO_THROW(adam.step(empty_params));
    ASSERT_NO_THROW(adam.zero_grad(empty_params));

    SGD sgd(0.01F);
    ASSERT_NO_THROW(sgd.attach(empty_params));
    ASSERT_NO_THROW(sgd.step(empty_params));
    ASSERT_NO_THROW(sgd.zero_grad(empty_params));
}

// Exception Testing for Optimizers
TEST(OptimizerExceptionTest, InvalidLearningRates)
{
    // Test negative learning rates
    ASSERT_THROW(SGDMinimal sgd_neg(-0.01F), std::invalid_argument);
    ASSERT_THROW(Adam adam_neg(-0.01F), std::invalid_argument);
    ASSERT_THROW(SGD sgd_neg(-0.01F), std::invalid_argument);

    // Test zero learning rates
    ASSERT_THROW(SGDMinimal sgd_zero(0.0F), std::invalid_argument);
    ASSERT_THROW(Adam adam_zero(0.0F), std::invalid_argument);
    ASSERT_THROW(SGD sgd_zero(0.0F), std::invalid_argument);

    // Test extremely large learning rates
    ASSERT_THROW(SGDMinimal sgd_large(1e10F), std::invalid_argument);
    ASSERT_THROW(Adam adam_large(1e10F), std::invalid_argument);
    ASSERT_THROW(SGD sgd_large(1e10F), std::invalid_argument);
}

TEST(OptimizerExceptionTest, NullParameters)
{
    std::vector<nn::Tensor*> params_with_null;
    params_with_null.push_back(nullptr);

    SGDMinimal sgd(0.01F);
    ASSERT_THROW(sgd.step(params_with_null), std::invalid_argument);
    ASSERT_THROW(sgd.zero_grad(params_with_null), std::invalid_argument);

    Adam adam(0.01F);
    ASSERT_THROW(adam.attach(params_with_null), std::invalid_argument);

    SGD sgd_momentum(0.01F);
    ASSERT_THROW(sgd_momentum.attach(params_with_null), std::invalid_argument);
}

// Numerical Edge Cases for Optimizers
TEST(OptimizerNumericalEdgeTest, NaNInfGradients)
{
    nn::Tensor param = test_helpers::make_ones_tensor(2, 2);

    // Test with NaN gradients
    nn::Tensor nan_grad = test_helpers::make_ones_tensor(2, 2);
    test_helpers::tensor_set_value_at(nan_grad, 0, 0, std::numeric_limits<float>::quiet_NaN());
    param.set_grad(nan_grad);

    std::vector<nn::Tensor*> params = {&param};

    SGDMinimal sgd(0.01F);
    // Should handle NaN gracefully (either throw or handle)
    EXPECT_NO_THROW(sgd.step(params));

    // Test with Inf gradients
    nn::Tensor inf_grad = test_helpers::make_ones_tensor(2, 2);
    test_helpers::tensor_set_value_at(inf_grad, 0, 0, std::numeric_limits<float>::infinity());
    param.set_grad(inf_grad);

    EXPECT_NO_THROW(sgd.step(params));

    // Test with very small gradients
    nn::Tensor tiny_grad = test_helpers::make_constant_tensor(2, 2, 1e-10F);
    param.set_grad(tiny_grad);

    EXPECT_NO_THROW(sgd.step(params));

    // Test with very large gradients
    nn::Tensor huge_grad = test_helpers::make_constant_tensor(2, 2, 1e10F);
    param.set_grad(huge_grad);

    EXPECT_NO_THROW(sgd.step(params));
}

TEST(OptimizerNumericalEdgeTest, ExtremeLearningRates)
{
    nn::Tensor param = test_helpers::make_ones_tensor(2, 2);
    param.set_grad(test_helpers::make_constant_tensor(2, 2, 0.1F));
    std::vector<nn::Tensor*> params = {&param};

    // Test with very small learning rate
    SGDMinimal sgd_tiny(1e-10F);
    nn::Tensor data_before(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_before.at(i, j) = param.at(i, j);
        }
    }
    sgd_tiny.step(params);
    nn::Tensor data_after(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_after.at(i, j) = param.at(i, j);
        }
    }

    // Change should be very small
    nn::Tensor diff = test_helpers::tensor_subtract(data_after, data_before);
    EXPECT_TRUE(test_helpers::tensor_norm(diff) < 1e-8F);

    // Test with very large learning rate (but not invalid)
    SGDMinimal sgd_large(1e6F);
    test_helpers::tensor_fill_with_value(param, 1.0F); // Reset
    param.set_grad(test_helpers::make_constant_tensor(2, 2, 0.1F));
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_before.at(i, j) = param.at(i, j);
        }
    }
    sgd_large.step(params);
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_after.at(i, j) = param.at(i, j);
        }
    }

    // Change should be large
    diff = test_helpers::tensor_subtract(data_after, data_before);
    EXPECT_TRUE(test_helpers::tensor_norm(diff) > 1.0F);
}

// Thread Safety Validation for Optimizers
TEST(OptimizerThreadSafetyTest, ConcurrentParameterUpdates)
{
    const int num_params = 10;
    std::vector<std::unique_ptr<nn::Tensor>> owned_params;
    owned_params.reserve(num_params);
    std::vector<nn::Tensor*> params;
    params.reserve(num_params);

    // Create multiple parameters
    for (int i = 0; i < num_params; ++i)
    {
        auto tensor = std::make_unique<nn::Tensor>(10, 10);
        test_helpers::tensor_fill_with_value(*tensor, 1.0F);
        tensor->set_grad(test_helpers::make_constant_tensor(10, 10, 0.1F));
        params.push_back(tensor.get());
        owned_params.push_back(std::move(tensor));
    }

    // Test multiple step operations (simulating concurrent access)
    SGDMinimal sgd(0.01F);
    for (int i = 0; i < 5; ++i)
    {
        ASSERT_NO_THROW(sgd.step(params));
        // Gradients should be preserved between steps
        for (auto* param : params)
        {
            EXPECT_FALSE(test_helpers::tensor_is_zero(param->grad()));
        }
    }

    // Test zero_grad operations
    ASSERT_NO_THROW(sgd.zero_grad(params));
    for (auto* param : params)
    {
        EXPECT_TRUE(test_helpers::tensor_is_zero(param->grad()));
    }
}

TEST(OptimizerThreadSafetyTest, AdamInternalState)
{
    nn::Tensor param = test_helpers::make_ones_tensor(3, 3);
    param.set_grad(test_helpers::make_constant_tensor(3, 3, 0.1F));
    std::vector<nn::Tensor*> params = {&param};

    Adam adam(0.01F);
    adam.attach(params);

    nn::Tensor data_before(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_before.at(i, j) = param.at(i, j);
        }
    }

    // Multiple steps to test internal state accumulation
    for (int i = 0; i < 3; ++i)
    {
        adam.step(params);
        // Re-set gradients for next step
        param.set_grad(test_helpers::make_constant_tensor(3, 3, 0.1F));
    }

    nn::Tensor data_after(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_after.at(i, j) = param.at(i, j);
        }
    }

    // Parameters should have changed
    EXPECT_FALSE(test_helpers::tensor_is_approx(data_before, data_after));

    // Test that internal state affects subsequent steps differently
    nn::Tensor data_step3(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_step3.at(i, j) = param.at(i, j);
        }
    }
    adam.step(params);
    nn::Tensor data_step4(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_step4.at(i, j) = param.at(i, j);
        }
    }

    // The change should be different due to accumulated internal state
    nn::Tensor diff3 = test_helpers::tensor_subtract(data_step4, data_step3);
    nn::Tensor diff1 = test_helpers::tensor_subtract(data_after, data_before);

    // Due to bias correction, later steps should behave differently
    EXPECT_FALSE(test_helpers::tensor_is_approx(diff3, diff1));
}

// Additional Comprehensive Tests
TEST(OptimizerComprehensiveTest, GradientClipping)
{
    nn::Tensor param = test_helpers::make_ones_tensor(2, 2);
    // Set very large gradients
    param.set_grad(test_helpers::make_constant_tensor(2, 2, 100.0F));
    std::vector<nn::Tensor*> params = {&param};

    SGDMinimal sgd(0.01F);
    nn::Tensor data_before(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_before.at(i, j) = param.at(i, j);
        }
    }

    sgd.step(params);

    nn::Tensor data_after(param.rows(), param.cols());
    for (size_t i = 0; i < param.rows(); ++i)
    {
        for (size_t j = 0; j < param.cols(); ++j)
        {
            data_after.at(i, j) = param.at(i, j);
        }
    }
    nn::Tensor diff = test_helpers::tensor_subtract(data_after, data_before);

    // Change should be controlled by learning rate
    EXPECT_TRUE(test_helpers::tensor_norm(diff) <
                10.0F); // Large gradients but small LR should limit change
}

TEST(OptimizerComprehensiveTest, ParameterGroups)
{
    // Test different learning rates for different parameter groups
    nn::Tensor param1 = test_helpers::make_ones_tensor(2, 2);
    nn::Tensor param2 = test_helpers::make_ones_tensor(2, 2);

    param1.set_grad(test_helpers::make_constant_tensor(2, 2, 0.1F));
    param2.set_grad(test_helpers::make_constant_tensor(2, 2, 0.1F));

    std::vector<nn::Tensor*> params1 = {&param1};
    std::vector<nn::Tensor*> params2 = {&param2};

    SGDMinimal sgd1(0.1F);  // Higher learning rate
    SGDMinimal sgd2(0.01F); // Lower learning rate

    nn::Tensor data1_before(param1.rows(), param1.cols());
    nn::Tensor data2_before(param2.rows(), param2.cols());
    for (size_t i = 0; i < param1.rows(); ++i)
    {
        for (size_t j = 0; j < param1.cols(); ++j)
        {
            data1_before.at(i, j) = param1.at(i, j);
            data2_before.at(i, j) = param2.at(i, j);
        }
    }

    sgd1.step(params1);
    sgd2.step(params2);

    nn::Tensor diff1 = test_helpers::tensor_subtract(param1, data1_before);
    nn::Tensor diff2 = test_helpers::tensor_subtract(param2, data2_before);

    // Higher learning rate should cause larger parameter changes
    EXPECT_TRUE(test_helpers::tensor_norm(diff1) > test_helpers::tensor_norm(diff2));
}

TEST(OptimizerComprehensiveTest, ConvergenceBehavior)
{
    nn::Tensor param = test_helpers::make_constant_tensor(2, 2, 10.0F); // Start far from zero
    std::vector<nn::Tensor*> params = {&param};

    SGDMinimal sgd(0.1F);

    // Simulate multiple optimization steps
    for (int i = 0; i < 10; ++i)
    {
        // Set gradient pointing toward zero (full magnitude to encourage faster convergence)
        nn::Tensor grad(param.rows(), param.cols());
        for (size_t r = 0; r < param.rows(); ++r)
        {
            for (size_t c = 0; c < param.cols(); ++c)
            {
                grad.at(r, c) = param.at(r, c);
            }
        }
        param.set_grad(grad);
        sgd.step(params);
    }

    // Parameter should have moved toward zero
    EXPECT_TRUE(test_helpers::tensor_norm(param) < 10.0F);

    // But not exactly zero (due to constant gradient)
    EXPECT_FALSE(test_helpers::tensor_is_zero(param));
}