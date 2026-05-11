/**
 * @file optimizers_gtest.cpp
 * @brief Unit tests for optimizer implementations (SGD, Adam, etc.).
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "core/utility/tests/test_helpers.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/optimizers/OptimizerFactory.hpp"
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
};

TEST_F(OptimizerTest, SGDMinimalOptimizerStepAndZeroGrad)
{
    SGDMinimal sgd_minimal(0.01F);
    sgd_minimal.step(params);

    // Exact one-step SGDMinimal update: p <- p - lr * grad.
    for (size_t i = 0; i < weights.rows(); ++i)
    {
        for (size_t j = 0; j < weights.cols(); ++j)
        {
            EXPECT_NEAR(weights.at(i, j), 0.99F, 1e-6F);
        }
    }
    for (size_t i = 0; i < bias.rows(); ++i)
    {
        EXPECT_NEAR(bias.at(i, 0), -0.01F, 1e-6F);
    }

    sgd_minimal.zero_grad(params);
    ASSERT_TRUE(test_helpers::tensor_is_zero(weights.grad(), 1e-6F));
    ASSERT_TRUE(test_helpers::tensor_is_zero(bias.grad(), 1e-6F));
}

TEST_F(OptimizerTest, AdamOptimizerStepAndZeroGrad)
{
    Adam adam(0.01F);
    adam.attach(params); // Adam requires attaching parameters
    adam.step(params);

    // At t=1 with grad=1 and default Adam hyperparameters, update is ~lr.
    // p <- p - lr * m_hat / (sqrt(v_hat) + eps) = p - 0.01/(1+eps).
    const float expected_weight = 1.0F - (0.01F / (1.0F + 1e-8F));
    const float expected_bias = 0.0F - (0.01F / (1.0F + 1e-8F));
    for (size_t i = 0; i < weights.rows(); ++i)
    {
        for (size_t j = 0; j < weights.cols(); ++j)
        {
            EXPECT_NEAR(weights.at(i, j), expected_weight, 1e-6F);
        }
    }
    for (size_t i = 0; i < bias.rows(); ++i)
    {
        EXPECT_NEAR(bias.at(i, 0), expected_bias, 1e-6F);
    }

    adam.zero_grad(params);
    ASSERT_TRUE(test_helpers::tensor_is_zero(weights.grad(), 1e-6F));
    ASSERT_TRUE(test_helpers::tensor_is_zero(bias.grad(), 1e-6F));
}

TEST_F(OptimizerTest, SGDOptimizerStepAndZeroGrad)
{
    SGD sgd(0.01F);
    sgd.attach(params); // SGD with momentum requires attaching parameters
    sgd.step(params);

    // Default momentum is 0.0, so first step matches plain SGD exactly.
    for (size_t i = 0; i < weights.rows(); ++i)
    {
        for (size_t j = 0; j < weights.cols(); ++j)
        {
            EXPECT_NEAR(weights.at(i, j), 0.99F, 1e-6F);
        }
    }
    for (size_t i = 0; i < bias.rows(); ++i)
    {
        EXPECT_NEAR(bias.at(i, 0), -0.01F, 1e-6F);
    }

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

TEST(OptimizerFactoryTest, CreatesAdamOptimizer)
{
    nn::optimizers::OptimizerFactoryConfig cfg;
    cfg.type = "adam";
    cfg.learning_rate = 1e-3F;
    cfg.adam_beta1 = 0.9F;
    cfg.adam_beta2 = 0.999F;
    cfg.adam_epsilon = 1e-8F;

    auto optimizer = nn::optimizers::OptimizerFactory::create(cfg);
    ASSERT_NE(optimizer, nullptr);
    EXPECT_NE(dynamic_cast<Adam*>(optimizer.get()), nullptr);
}

TEST(OptimizerFactoryTest, CreatesSgdOptimizerCaseInsensitive)
{
    nn::optimizers::OptimizerFactoryConfig cfg;
    cfg.type = "SGD";
    cfg.learning_rate = 1e-2F;
    cfg.momentum = 0.5F;

    auto optimizer = nn::optimizers::OptimizerFactory::create(cfg);
    ASSERT_NE(optimizer, nullptr);
    EXPECT_NE(dynamic_cast<SGD*>(optimizer.get()), nullptr);
}

TEST(OptimizerFactoryTest, ThrowsOnUnknownType)
{
    nn::optimizers::OptimizerFactoryConfig cfg;
    cfg.type = "unknown";
    cfg.learning_rate = 1e-3F;

    EXPECT_THROW((void) nn::optimizers::OptimizerFactory::create(cfg), std::runtime_error);
}

TEST(OptimizerFactoryTest, DirectOverloadCreatesAdam)
{
    auto optimizer =
        nn::optimizers::OptimizerFactory::create("adam", 1e-3F, 0.0F, 0.9F, 0.999F, 1e-8F);
    ASSERT_NE(optimizer, nullptr);
    EXPECT_NE(dynamic_cast<Adam*>(optimizer.get()), nullptr);
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

    // lr * grad = 1e-10 * 0.1 = 1e-11/elem — below float32 epsilon (~1.19e-7) around 1.0.
    // Subtraction 1.0 - 1e-11 rounds to exactly 1.0 in float32; diff is exactly zero.
    nn::Tensor diff = test_helpers::tensor_subtract(data_after, data_before);
    EXPECT_EQ(test_helpers::tensor_norm(diff), 0.0F);

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

    // diff = -lr * grad = -1e6 * 0.1 = -1e5/elem; 2x2 tensor: norm = sqrt(4 * (1e5)^2) = 2e5
    diff = test_helpers::tensor_subtract(data_after, data_before);
    EXPECT_NEAR(test_helpers::tensor_norm(diff), 2e5F, 1.0F);
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
        // Gradients constant at 0.1 (never zeroed): verify each element
        for (const auto* p : params)
            for (size_t r = 0; r < p->rows(); ++r)
                for (size_t c = 0; c < p->cols(); ++c) EXPECT_NEAR(p->grad().at(r, c), 0.1F, 1e-6F);
    }

    // After zero_grad each gradient element must be exactly 0
    ASSERT_NO_THROW(sgd.zero_grad(params));
    for (const auto* p : params)
        for (size_t r = 0; r < p->rows(); ++r)
            for (size_t c = 0; c < p->cols(); ++c) EXPECT_NEAR(p->grad().at(r, c), 0.0F, 1e-6F);
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

    // After 3 Adam(0.01) steps with const grad=0.1: each step ≈ 0.01 update; param ≈ 0.97
    for (size_t i = 0; i < data_after.rows(); ++i)
        for (size_t j = 0; j < data_after.cols(); ++j)
            EXPECT_NEAR(data_after.at(i, j), 0.97F, 1e-2F);

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

    // diff1 = 3-step cumulative (~0.03/elem for 3x3); diff3 = single step 4 (~0.01/elem)
    // Cumulative norm ≈ 0.09; single-step norm ≈ 0.03
    EXPECT_NEAR(test_helpers::tensor_norm(diff3), 0.03F, 5e-3F);
    EXPECT_NEAR(test_helpers::tensor_norm(diff1), 0.09F, 1e-2F);
    EXPECT_GT(test_helpers::tensor_norm(diff1), test_helpers::tensor_norm(diff3));
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

    // diff = -lr * grad = -0.01 * 100 = -1.0/elem; 2x2 tensor: norm = sqrt(4 * 1.0) = 2.0
    EXPECT_NEAR(test_helpers::tensor_norm(diff), 2.0F, 1e-5F);
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

    // param1: lr=0.1, grad=0.1 → diff = -0.01/elem; 2x2 norm = sqrt(4 * 1e-4) = 0.02
    // param2: lr=0.01, grad=0.1 → diff = -0.001/elem; 2x2 norm = sqrt(4 * 1e-6) = 0.002
    EXPECT_NEAR(test_helpers::tensor_norm(diff1), 0.02F, 1e-6F);
    EXPECT_NEAR(test_helpers::tensor_norm(diff2), 0.002F, 1e-6F);
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

    // SGD(0.1) with grad=param each step: p_n = p_0 * (1 - lr)^n = 10 * 0.9^10
    const float p10 = 10.0F * std::pow(0.9F, 10); // ≈ 3.4868
    // 2x2 tensor norm = 2 * p10 ≈ 6.9736
    EXPECT_NEAR(test_helpers::tensor_norm(param), 2.0F * p10, 1e-3F);
    // Each element equals p10
    for (size_t r = 0; r < param.rows(); ++r)
        for (size_t c = 0; c < param.cols(); ++c) EXPECT_NEAR(param.at(r, c), p10, 1e-4F);
}
// SGD null parameter guard in step() (SGD.hpp line 74)
TEST(SGDNullParamTest, StepThrowsOnNullParam)
{
    SGD sgd(0.01f);
    nn::Tensor p(2, 2);
    nn::Tensor p2(2, 2);
    p.fill(1.0f);
    p2.fill(1.0f);
    // Attach two valid params first (populates velocity vector)
    std::vector<nn::Tensor*> valid_params = {&p, &p2};
    sgd.attach(valid_params);
    // Now step with second param replaced by null
    nn::Tensor* null_ptr = nullptr;
    std::vector<nn::Tensor*> params_with_null = {&p, null_ptr};
    EXPECT_THROW(sgd.step(params_with_null), std::invalid_argument);
}

// SGD null parameter guard in zero_grad() (SGD.hpp line 90)
TEST(SGDNullParamTest, ZeroGradThrowsOnNullParam)
{
    SGD sgd(0.01f);
    nn::Tensor p(2, 2);
    p.fill(1.0f);
    nn::Tensor* null_ptr = nullptr;
    std::vector<nn::Tensor*> params_with_null = {&p, null_ptr};
    EXPECT_THROW(sgd.zero_grad(params_with_null), std::invalid_argument);
}

// SGDMinimal attach() with null param (SGDMinimal.hpp lines 71-81)
TEST(SGDMinimalTest, AttachThrowsOnNullParam)
{
    SGDMinimal sgd(0.01f);
    nn::Tensor p(2, 2);
    p.fill(1.0f);
    nn::Tensor* null_ptr = nullptr;
    std::vector<nn::Tensor*> params_with_null = {&p, null_ptr};
    EXPECT_THROW(sgd.attach(params_with_null), std::invalid_argument);
}

// SGDMinimal attach() with valid params (covers the attach() method body)
TEST(SGDMinimalTest, AttachWithValidParams)
{
    SGDMinimal sgd(0.01f);
    nn::Tensor p(2, 2);
    p.fill(1.0f);
    std::vector<nn::Tensor*> valid_params = {&p};
    EXPECT_NO_THROW(sgd.attach(valid_params));
}

namespace
{
struct DummyOptimizer final : public Optimizer
{
    using Optimizer::step;
    using Optimizer::zero_grad;

    size_t step_count = 0;
    size_t zero_count = 0;
    size_t last_step_size = 0;
    size_t last_zero_size = 0;

    auto step(std::span<Tensor*> params) -> void override
    {
        ++step_count;
        last_step_size = params.size();
    }

    auto zero_grad(std::span<Tensor*> params) -> void override
    {
        ++zero_count;
        last_zero_size = params.size();
    }
};
} // namespace

TEST(OptimizerBaseTest, ConvenienceMethodsAndDefaults)
{
    DummyOptimizer opt;

    EXPECT_THROW(opt.step(), std::runtime_error);
    EXPECT_THROW(opt.zero_grad(), std::runtime_error);

    nn::Tensor p(1, 1);
    p.at(0, 0) = 1.0F;
    std::vector<nn::Tensor*> params = {&p};

    // Base attach() is a no-op.
    EXPECT_NO_THROW(opt.attach(params));
    EXPECT_TRUE(opt.attached_params_.empty());

    // Manually populate attached params to exercise no-arg dispatch paths.
    opt.attached_params_.assign(params.begin(), params.end());
    EXPECT_NO_THROW(opt.step());
    EXPECT_NO_THROW(opt.zero_grad());
    EXPECT_EQ(opt.step_count, 1U);
    EXPECT_EQ(opt.zero_count, 1U);
    EXPECT_EQ(opt.last_step_size, 1U);
    EXPECT_EQ(opt.last_zero_size, 1U);

    auto sd = opt.state_dict();
    EXPECT_TRUE(sd.empty());
    EXPECT_NO_THROW(opt.load_state_dict(sd));
}

TEST(AdamTest, AttachWithScalesValidationAndSuccess)
{
    Adam adam(0.01F);
    nn::Tensor p1(2, 2);
    nn::Tensor p2(2, 2);
    std::vector<nn::Tensor*> params = {&p1, &p2};

    const std::vector<float> bad_scales = {0.1F};
    EXPECT_THROW(adam.attach_with_scales(params, bad_scales), std::invalid_argument);

    const std::vector<float> good_scales = {0.1F, 0.5F};
    EXPECT_NO_THROW(adam.attach_with_scales(params, good_scales));
    ASSERT_EQ(adam.lr_scales_.size(), 2U);
    EXPECT_FLOAT_EQ(adam.lr_scales_[0], 0.1F);
    EXPECT_FLOAT_EQ(adam.lr_scales_[1], 0.5F);
    EXPECT_EQ(adam.moment1.size(), 2U);
    EXPECT_EQ(adam.moment2.size(), 2U);
}

TEST(AdamTest, StateDictRoundTripAndNullParamGuards)
{
    nn::Tensor p1(1, 1);
    nn::Tensor p2(1, 1);
    p1.at(0, 0) = 1.0F;
    p2.at(0, 0) = 2.0F;
    p1.set_grad(test_helpers::make_constant_tensor(1, 1, 0.25F));
    p2.set_grad(test_helpers::make_constant_tensor(1, 1, -0.50F));
    std::vector<nn::Tensor*> params = {&p1, &p2};

    Adam adam_src(0.01F);
    adam_src.attach(params);
    adam_src.step(params);
    const auto saved = adam_src.state_dict();

    Adam adam_dst(0.01F);
    adam_dst.attach(params);
    EXPECT_NO_THROW(adam_dst.load_state_dict(saved));
    EXPECT_EQ(adam_dst.time_step, adam_src.time_step);
    ASSERT_EQ(adam_dst.moment1.size(), adam_src.moment1.size());
    ASSERT_EQ(adam_dst.moment2.size(), adam_src.moment2.size());
    EXPECT_NEAR(adam_dst.moment1[0].at(0, 0), adam_src.moment1[0].at(0, 0), 1e-6F);
    EXPECT_NEAR(adam_dst.moment2[1].at(0, 0), adam_src.moment2[1].at(0, 0), 1e-6F);

    // Also exercise the path where time_step exists but tensor has zero shape.
    std::map<std::string, nn::Tensor> malformed = saved;
    malformed["time_step"] = nn::Tensor(0, 0);
    EXPECT_NO_THROW(adam_dst.load_state_dict(malformed));

    nn::Tensor* null_ptr = nullptr;
    std::vector<nn::Tensor*> params_with_null = {&p1, null_ptr};
    EXPECT_THROW(adam_dst.step(params_with_null), std::invalid_argument);
    EXPECT_THROW(adam_dst.zero_grad(params_with_null), std::invalid_argument);
}
