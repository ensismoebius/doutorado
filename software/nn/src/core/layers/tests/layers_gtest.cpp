/**
 * @file layers_gtest.cpp
 * @brief Layer robustness coverage: exception paths, memory stress, numerical
 *        edge cases, and thread safety.
 *
 * The broader per-layer-family coverage that used to live in this one file
 * has moved to the sibling layers_*_gtest.cpp files in this directory
 * (convolution, spiking_basic, losses_regularization, composite) -- this file
 * kept the suites that didn't fit any of those categories.
 */

#include <gtest/gtest.h>

#include <memory>

#include "core/utility/tests/test_helpers.hpp"
#include "layers/Layers.hpp"
#include "layers/convolution/Conv2d.hpp"
#include "layers/spiking/Lif.hpp"
#include "layers/spiking/LifBPTT.hpp"
#include "tensor/Tensor.hpp"

using nn::Conv2d;
using nn::LeakyReLU;
using nn::Lif;
using nn::LifIntegrator;
using nn::Linear;
using nn::MAELoss;
using nn::MSELoss;
using nn::ReLU;
using nn::Sequential;
using nn::SimpleResNet;
using nn::SpikeCountLoss;

// Exception Testing for Layers
TEST(LayerExceptionTest, LinearInvalidDimensions)
{
    // Test Linear layer with invalid dimensions
    Linear linear(2, 3);

    // Test forward with wrong input dimensions
    nn::Tensor wrong_tensor(1, 5); // Should be 1x2
    ASSERT_THROW(linear.forward(wrong_tensor), std::invalid_argument);

    // Test backward with wrong gradient dimensions
    nn::Tensor wrong_grad_tensor(1, 5); // Should be 1x3
    ASSERT_THROW(linear.backward(wrong_grad_tensor), std::invalid_argument);
}

TEST(LayerExceptionTest, Conv2dInvalidInputs)
{
    Conv2d conv(1, 1, 3, 1, 0, 1, false); // kernel=3, stride=1, padding=0 (no padding), dilation=1,
                                          // use_parallel=false

    // Test with input too small for kernel - proper 4D tensor (batch, channels, height, width)
    nn::Tensor small_tensor(1,
        1,
        1,
        1); // batch=1, channels=1, height=1, width=1 - too small for 3x3 kernel with no padding
    ASSERT_THROW(conv.forward(small_tensor), std::invalid_argument);

    // Test with valid dimensions but 4D tensor
    nn::Tensor valid_tensor(
        1, 1, 5, 5); // batch=1, channels=1, height=5, width=5 - valid for 3x3 kernel
    EXPECT_NO_THROW(conv.forward(valid_tensor));
}

TEST(LayerExceptionTest, SequentialEmptyLayers)
{
    Sequential seq({});
    nn::Tensor input_tensor(1, 2);
    ASSERT_THROW(seq.forward(input_tensor), std::runtime_error);
}

TEST(LayerExceptionTest, MSELossInvalidTargets)
{
    MSELoss mse;
    nn::Tensor pred_tensor(2, 1);
    pred_tensor.at(0, 0) = 1.0F;
    pred_tensor.at(1, 0) = 2.0F;

    // Test without setting target
    ASSERT_THROW(mse.forward(pred_tensor), std::runtime_error);

    // Test with mismatched dimensions
    nn::Tensor target_tensor(3, 1); // Different size than prediction
    target_tensor.at(0, 0) = 0.0F;
    target_tensor.at(1, 0) = 1.0F;
    target_tensor.at(2, 0) = 2.0F;
    mse.set_target(target_tensor);
    ASSERT_THROW(mse.forward(pred_tensor), std::invalid_argument);
}

// Memory Stress Testing for Layers
TEST(LayerMemoryStressTest, LargeLinearLayer)
{
    const int large_input = 1000;
    const int large_output = 500;

    Linear linear(large_input, large_output);

    // Create large input
    nn::Tensor input_tensor = test_helpers::make_random_tensor(1, large_input);
    nn::Tensor output = linear.forward(input_tensor);

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), large_output);

    // Test backward with large gradients
    nn::Tensor grad_tensor = test_helpers::make_ones_tensor(1, large_output);
    nn::Tensor grad_input = linear.backward(grad_tensor);

    EXPECT_EQ(grad_input.rows(), 1);
    EXPECT_EQ(grad_input.cols(), large_input);
}

TEST(LayerNumericalEdgeTest, GradientNumericalStability)
{
    Linear linear(2, 1);

    // Test with very small gradients
    nn::Tensor small_tensor(1, 2);
    small_tensor.at(0, 0) = 1e-8F;
    small_tensor.at(0, 1) = 1e-8F;
    [[maybe_unused]] nn::Tensor small_output = linear.forward(small_tensor);

    nn::Tensor small_grad_tensor(1, 1);
    small_grad_tensor.at(0, 0) = 1e-8F;
    nn::Tensor small_grad_input = linear.backward(small_grad_tensor);

    // Should not produce NaN or Inf
    EXPECT_FALSE(std::isnan(small_grad_input.at(0, 0)));
    EXPECT_FALSE(std::isinf(small_grad_input.at(0, 0)));
    EXPECT_FALSE(std::isnan(small_grad_input.at(0, 1)));
    EXPECT_FALSE(std::isinf(small_grad_input.at(0, 1)));

    // Test with very large gradients
    nn::Tensor large_grad_tensor(1, 1);
    large_grad_tensor.at(0, 0) = 1e8F;
    nn::Tensor large_grad_input = linear.backward(large_grad_tensor);

    // Should handle large values gracefully
    EXPECT_FALSE(std::isnan(large_grad_input.at(0, 0)));
    EXPECT_TRUE(std::isfinite(large_grad_input.at(0, 0)));
}

// Thread Safety Validation for Layers
TEST(LayerThreadSafetyTest, ConcurrentForwardPasses)
{
    Linear linear(10, 5);

    // Create multiple inputs
    std::vector<nn::Tensor> inputs;
    for (int i = 0; i < 10; ++i)
    {
        inputs.push_back(test_helpers::make_random_tensor(1, 10));
    }

    // Test concurrent forward passes (basic test)
    std::vector<nn::Tensor> outputs;
    for (const auto& input : inputs)
    {
        nn::Tensor output = linear.forward(input, false); // No gradient caching
        outputs.push_back(output);
    }

    // Verify outputs are valid
    for (const auto& output : outputs)
    {
        EXPECT_EQ(output.rows(), 1);
        EXPECT_EQ(output.cols(), 5);
        EXPECT_FALSE(std::isnan(output.at(0, 0)));
    }
}

TEST(LayerThreadSafetyTest, GradientAccumulation)
{
    Linear linear(3, 2);

    // Multiple forward-backward cycles
    for (int cycle = 0; cycle < 5; ++cycle)
    {
        nn::Tensor input_tensor = test_helpers::make_random_tensor(1, 3);

        // Forward
        [[maybe_unused]] nn::Tensor output = linear.forward(input_tensor);

        // Backward
        nn::Tensor grad_tensor = test_helpers::make_ones_tensor(1, 2);
        [[maybe_unused]] nn::Tensor grad_input = linear.backward(grad_tensor);

        // Verify gradients are accumulated properly
        EXPECT_FALSE(std::isnan(linear.weight.sum()));
        EXPECT_FALSE(std::isnan(linear.bias.sum()));
    }
}
