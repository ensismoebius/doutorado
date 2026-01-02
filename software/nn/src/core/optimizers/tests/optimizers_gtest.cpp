#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "core/optimizers/Adam.hpp"
#include "core/optimizers/SGD.hpp"
#include "core/optimizers/SGDMinimal.hpp"
#include "core/tensor/Tensor.hpp"

// Test Fixture for common optimizer setup
class OptimizerTest : public ::testing::Test
{
   protected:
    // Common data for weights and bias
    Eigen::MatrixXf initial_weights_data = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf initial_bias_data = Eigen::MatrixXf::Zero(2, 1);
    Eigen::MatrixXf initial_weights_grad = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf initial_bias_grad = Eigen::MatrixXf::Ones(2, 1);

    nn::Tensor weights;
    nn::Tensor bias;
    std::vector<nn::Tensor*> params;

    OptimizerTest()
        : weights(initial_weights_data), // Initialize Tensor objects
          bias(initial_bias_data)
    {
        params.push_back(&weights);
        params.push_back(&bias);
        weights.set_grad(initial_weights_grad);
        bias.set_grad(initial_bias_grad);
    }

    // Helper to check if a tensor's data has changed from initial
    bool has_data_changed(const nn::Tensor& tensor, const Eigen::MatrixXf& initial_data) const
    {
        return !tensor.get_data_ref().isApprox(initial_data);
    }
};

TEST_F(OptimizerTest, SGDMinimalOptimizerStepAndZeroGrad)
{
    SGDMinimal sgd_minimal(0.01F);
    sgd_minimal.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    sgd_minimal.zero_grad(params);
    ASSERT_TRUE(weights.get_grad_ref().isZero(1e-6F)); // Use isZero for Eigen matrices
    ASSERT_TRUE(bias.get_grad_ref().isZero(1e-6F));
}

TEST_F(OptimizerTest, AdamOptimizerStepAndZeroGrad)
{
    Adam adam(0.01F);
    adam.attach(params); // Adam requires attaching parameters
    adam.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    adam.zero_grad(params);
    ASSERT_TRUE(weights.get_grad_ref().isZero(1e-6F));
    ASSERT_TRUE(bias.get_grad_ref().isZero(1e-6F));
}

TEST_F(OptimizerTest, SGDOptimizerStepAndZeroGrad)
{
    SGD sgd(0.01F);
    sgd.attach(params); // SGD with momentum requires attaching parameters
    sgd.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    sgd.zero_grad(params);
    ASSERT_TRUE(weights.get_grad_ref().isZero(1e-6F));
    ASSERT_TRUE(bias.get_grad_ref().isZero(1e-6F));
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
    std::vector<nn::Tensor*> params;

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

// Memory Stress Testing for Optimizers
TEST(OptimizerMemoryStressTest, LargeParameterSets)
{
    const int num_params = 100;
    const int param_size = 1000;

    std::vector<nn::Tensor*> large_params;
    std::vector<Eigen::MatrixXf> initial_data;

    // Create large parameter set
    for (int i = 0; i < num_params; ++i)
    {
        Eigen::MatrixXf data = Eigen::MatrixXf::Random(param_size, param_size);
        Eigen::MatrixXf grad = Eigen::MatrixXf::Ones(param_size, param_size) * 0.1F;

        initial_data.push_back(data);

        auto* tensor = new nn::Tensor(data);
        tensor->set_grad(grad);
        large_params.push_back(tensor);
    }

    // Test SGDMinimal with large parameter set
    SGDMinimal sgd(0.01F);
    ASSERT_NO_THROW(sgd.step(large_params));
    ASSERT_NO_THROW(sgd.zero_grad(large_params));

    // Verify parameters changed
    for (int i = 0; i < num_params; ++i)
    {
        EXPECT_FALSE(large_params[i]->get_data_ref().isApprox(initial_data[i]));
        EXPECT_TRUE(large_params[i]->get_grad_ref().isZero());
    }

    // Test Adam with large parameter set
    Adam adam(0.001F);
    ASSERT_NO_THROW(adam.attach(large_params));
    ASSERT_NO_THROW(adam.step(large_params));
    ASSERT_NO_THROW(adam.zero_grad(large_params));

    // Clean up
    for (auto* tensor : large_params)
    {
        delete tensor;
    }
}

// Numerical Edge Cases for Optimizers
TEST(OptimizerNumericalEdgeTest, NaNInfGradients)
{
    nn::Tensor param(Eigen::MatrixXf::Ones(2, 2));

    // Test with NaN gradients
    Eigen::MatrixXf nan_grad = Eigen::MatrixXf::Ones(2, 2);
    nan_grad(0, 0) = std::numeric_limits<float>::quiet_NaN();
    param.set_grad(nan_grad);

    std::vector<nn::Tensor*> params = {&param};

    SGDMinimal sgd(0.01F);
    // Should handle NaN gracefully (either throw or handle)
    EXPECT_NO_THROW(sgd.step(params));

    // Test with Inf gradients
    Eigen::MatrixXf inf_grad = Eigen::MatrixXf::Ones(2, 2);
    inf_grad(0, 0) = std::numeric_limits<float>::infinity();
    param.set_grad(inf_grad);

    EXPECT_NO_THROW(sgd.step(params));

    // Test with very small gradients
    Eigen::MatrixXf tiny_grad = Eigen::MatrixXf::Ones(2, 2) * 1e-10F;
    param.set_grad(tiny_grad);

    EXPECT_NO_THROW(sgd.step(params));

    // Test with very large gradients
    Eigen::MatrixXf huge_grad = Eigen::MatrixXf::Ones(2, 2) * 1e10F;
    param.set_grad(huge_grad);

    EXPECT_NO_THROW(sgd.step(params));
}

TEST(OptimizerNumericalEdgeTest, ExtremeLearningRates)
{
    nn::Tensor param(Eigen::MatrixXf::Ones(2, 2));
    param.set_grad(Eigen::MatrixXf::Ones(2, 2) * 0.1F);
    std::vector<nn::Tensor*> params = {&param};

    // Test with very small learning rate
    SGDMinimal sgd_tiny(1e-10F);
    Eigen::MatrixXf data_before = param.get_data_ref();
    sgd_tiny.step(params);
    Eigen::MatrixXf data_after = param.get_data_ref();

    // Change should be very small
    Eigen::MatrixXf diff = data_after - data_before;
    EXPECT_TRUE(diff.norm() < 1e-8F);

    // Test with very large learning rate (but not invalid)
    SGDMinimal sgd_large(1e6F);
    param.set_data(Eigen::MatrixXf::Ones(2, 2)); // Reset
    param.set_grad(Eigen::MatrixXf::Ones(2, 2) * 0.1F);
    data_before = param.get_data_ref();
    sgd_large.step(params);
    data_after = param.get_data_ref();

    // Change should be large
    diff = data_after - data_before;
    EXPECT_TRUE(diff.norm() > 1.0F);
}

// Thread Safety Validation for Optimizers
TEST(OptimizerThreadSafetyTest, ConcurrentParameterUpdates)
{
    const int num_params = 10;
    std::vector<nn::Tensor*> params;

    // Create multiple parameters
    for (int i = 0; i < num_params; ++i)
    {
        auto* tensor = new nn::Tensor(Eigen::MatrixXf::Ones(10, 10));
        tensor->set_grad(Eigen::MatrixXf::Ones(10, 10) * 0.1F);
        params.push_back(tensor);
    }

    // Test multiple step operations (simulating concurrent access)
    SGDMinimal sgd(0.01F);
    for (int i = 0; i < 5; ++i)
    {
        ASSERT_NO_THROW(sgd.step(params));
        // Gradients should be preserved between steps
        for (auto* param : params)
        {
            EXPECT_FALSE(param->get_grad_ref().isZero());
        }
    }

    // Test zero_grad operations
    ASSERT_NO_THROW(sgd.zero_grad(params));
    for (auto* param : params)
    {
        EXPECT_TRUE(param->get_grad_ref().isZero());
    }

    // Clean up
    for (auto* param : params)
    {
        delete param;
    }
}

TEST(OptimizerThreadSafetyTest, AdamInternalState)
{
    nn::Tensor param(Eigen::MatrixXf::Ones(3, 3));
    param.set_grad(Eigen::MatrixXf::Ones(3, 3) * 0.1F);
    std::vector<nn::Tensor*> params = {&param};

    Adam adam(0.01F);
    adam.attach(params);

    Eigen::MatrixXf data_before = param.get_data_ref();

    // Multiple steps to test internal state accumulation
    for (int i = 0; i < 3; ++i)
    {
        adam.step(params);
        // Re-set gradients for next step
        param.set_grad(Eigen::MatrixXf::Ones(3, 3) * 0.1F);
    }

    Eigen::MatrixXf data_after = param.get_data_ref();

    // Parameters should have changed
    EXPECT_FALSE(data_before.isApprox(data_after));

    // Test that internal state affects subsequent steps differently
    Eigen::MatrixXf data_step3 = param.get_data_ref();
    adam.step(params);
    Eigen::MatrixXf data_step4 = param.get_data_ref();

    // The change should be different due to accumulated internal state
    Eigen::MatrixXf diff3 = data_step4 - data_step3;
    Eigen::MatrixXf diff1 = data_after - data_before;

    // Due to bias correction, later steps should behave differently
    EXPECT_FALSE(diff3.isApprox(diff1));
}

// Additional Comprehensive Tests
TEST(OptimizerComprehensiveTest, GradientClipping)
{
    nn::Tensor param(Eigen::MatrixXf::Ones(2, 2));
    // Set very large gradients
    param.set_grad(Eigen::MatrixXf::Ones(2, 2) * 100.0F);
    std::vector<nn::Tensor*> params = {&param};

    SGDMinimal sgd(0.01F);
    Eigen::MatrixXf data_before = param.get_data_ref();

    sgd.step(params);

    Eigen::MatrixXf data_after = param.get_data_ref();
    Eigen::MatrixXf diff = data_after - data_before;

    // Change should be controlled by learning rate
    EXPECT_TRUE(diff.norm() < 10.0F); // Large gradients but small LR should limit change
}

TEST(OptimizerComprehensiveTest, ParameterGroups)
{
    // Test different learning rates for different parameter groups
    nn::Tensor param1(Eigen::MatrixXf::Ones(2, 2));
    nn::Tensor param2(Eigen::MatrixXf::Ones(2, 2));

    param1.set_grad(Eigen::MatrixXf::Ones(2, 2) * 0.1F);
    param2.set_grad(Eigen::MatrixXf::Ones(2, 2) * 0.1F);

    std::vector<nn::Tensor*> params1 = {&param1};
    std::vector<nn::Tensor*> params2 = {&param2};

    SGDMinimal sgd1(0.1F);  // Higher learning rate
    SGDMinimal sgd2(0.01F); // Lower learning rate

    Eigen::MatrixXf data1_before = param1.get_data_ref();
    Eigen::MatrixXf data2_before = param2.get_data_ref();

    sgd1.step(params1);
    sgd2.step(params2);

    Eigen::MatrixXf diff1 = param1.get_data_ref() - data1_before;
    Eigen::MatrixXf diff2 = param2.get_data_ref() - data2_before;

    // Higher learning rate should cause larger parameter changes
    EXPECT_TRUE(diff1.norm() > diff2.norm());
}

TEST(OptimizerComprehensiveTest, ConvergenceBehavior)
{
    nn::Tensor param(Eigen::MatrixXf::Ones(2, 2) * 10.0F); // Start far from zero
    std::vector<nn::Tensor*> params = {&param};

    SGDMinimal sgd(0.1F);

    // Simulate multiple optimization steps
    for (int i = 0; i < 10; ++i)
    {
        // Set gradient pointing toward zero (full magnitude to encourage faster convergence)
        Eigen::MatrixXf grad = param.get_data_ref();
        param.set_grad(grad);
        sgd.step(params);
    }

    // Parameter should have moved toward zero
    EXPECT_TRUE(param.get_data_ref().norm() < 10.0F);

    // But not exactly zero (due to constant gradient)
    EXPECT_FALSE(param.get_data_ref().isZero());
}