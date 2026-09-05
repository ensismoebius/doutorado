/**
 * @file layers_convolution_gtest.cpp
 * @brief Conv1d/Conv2d layer unit tests (forward, backward, edge cases, caching).
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

// Deterministic random fills for tests use test_helpers::rand_fill(..., nn::testing::kSeed)

TEST(Conv2dTest, ForwardAndBackward)
{
    // Layer parameters
    const int in_channels = 1;
    const int out_channels = 1;
    const int kernel_size = 2;
    [[maybe_unused]] const int batch_size = 1;
    [[maybe_unused]] const int input_height = 3;
    [[maybe_unused]] const int input_width = 3;

    // Create layer
    Conv2d conv(in_channels, out_channels, kernel_size);

    // Initialize weights and bias
    nn::Tensor weights(std::vector<size_t>{
        static_cast<size_t>(kernel_size * kernel_size * in_channels), out_channels});
    weights.at(0, 0) = 1.0F;
    weights.at(1, 0) = 2.0F;
    weights.at(2, 0) = 3.0F;
    weights.at(3, 0) = 4.0F;
    conv.set_weights(weights);

    nn::Tensor bias(std::vector<size_t>{1, out_channels});
    bias.at(0, 0) = 0.5F;
    conv.set_bias(bias);

    // Input tensor
    nn::Tensor input(std::vector<size_t>{1, 1, 3, 3});
    input.at(0, 0, 0, 0) = 1;
    input.at(0, 0, 0, 1) = 2;
    input.at(0, 0, 0, 2) = 3;
    input.at(0, 0, 1, 0) = 4;
    input.at(0, 0, 1, 1) = 5;
    input.at(0, 0, 1, 2) = 6;
    input.at(0, 0, 2, 0) = 7;
    input.at(0, 0, 2, 1) = 8;
    input.at(0, 0, 2, 2) = 9;

    // Forward pass
    nn::Tensor output = conv.forward(input);

    // Check output shape
    const auto& out_shape = output.get_shape();
    ASSERT_EQ(out_shape.size(), 4);
    ASSERT_EQ(out_shape[0], batch_size);
    ASSERT_EQ(out_shape[1], out_channels);
    ASSERT_EQ(out_shape[2], 2); // 3 - 2 + 1
    ASSERT_EQ(out_shape[3], 2); // 3 - 2 + 1

    // Check output values (manual calculation)
    // Output[0,0,0,0] = (1*1 + 2*2 + 4*3 + 5*4) + 0.5 = 1+4+12+20 + 0.5 = 37.5
    // Output[0,0,0,1] = (2*1 + 3*2 + 5*3 + 6*4) + 0.5 = 2+6+15+24 + 0.5 = 47.5
    // Output[0,0,1,0] = (4*1 + 5*2 + 7*3 + 8*4) + 0.5 = 4+10+21+32 + 0.5 = 67.5
    // Output[0,0,1,1] = (5*1 + 6*2 + 8*3 + 9*4) + 0.5 = 5+12+24+36 + 0.5 = 77.5
    ASSERT_NEAR(output.at(0, 0, 0, 0), 37.5, 1e-5);
    ASSERT_NEAR(output.at(0, 0, 0, 1), 47.5, 1e-5);
    ASSERT_NEAR(output.at(0, 0, 1, 0), 67.5, 1e-5);
    ASSERT_NEAR(output.at(0, 0, 1, 1), 77.5, 1e-5);

    nn::Tensor grad_output(std::vector<size_t>{
        static_cast<size_t>(batch_size), static_cast<size_t>(out_channels), 2, 2});
    grad_output.set_ones(); // Gradient of 1 for all output elements

    nn::Tensor grad_input = conv.backward(grad_output);

    // Check bias gradient
    ASSERT_NEAR(conv.get_bias().grad().at(0, 0), 4.0, 1e-5); // sum of grad_output = 1+1+1+1

    // Check weights gradient
    // grad_w[0] = 1*1 + 2*1 + 4*1 + 5*1 = 12
    // grad_w[1] = 2*1 + 3*1 + 5*1 + 6*1 = 16
    // grad_w[2] = 4*1 + 5*1 + 7*1 + 8*1 = 24
    // grad_w[3] = 5*1 + 6*1 + 8*1 + 9*1 = 28
    // Weights are (patch_rows, out_channels) = (4, 1)
    ASSERT_NEAR(conv.get_weights().grad().at(0, 0), 12, 1e-5);
    ASSERT_NEAR(conv.get_weights().grad().at(1, 0), 16, 1e-5);
    ASSERT_NEAR(conv.get_weights().grad().at(2, 0), 24, 1e-5);
    ASSERT_NEAR(conv.get_weights().grad().at(3, 0), 28, 1e-5);

    // Check input gradient
    ASSERT_NEAR(grad_input.at(0, 0, 0, 0), 1.0, 1e-5);
    ASSERT_NEAR(grad_input.at(0, 0, 0, 1), 3.0, 1e-5);
    ASSERT_NEAR(grad_input.at(0, 0, 1, 1), 10.0, 1e-5);
    ASSERT_NEAR(grad_input.at(0, 0, 2, 2), 4.0, 1e-5);
}

// Test: Multiple batches processing
TEST(Conv2dTest, MultipleBatches)
{
    const int in_channels = 1;
    const int out_channels = 1;
    const int kernel_size = 2;
    const int batch_size = 4;
    const int input_height = 3;
    const int input_width = 3;

    Conv2d conv(in_channels, out_channels, kernel_size);

    // Simple weights and bias
    nn::Tensor weights(kernel_size * kernel_size * in_channels, out_channels);
    weights.set_ones();
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.at(0, 0) = 0.0F;
    conv.set_bias(bias);

    // Create input with different values for each batch
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    for (int b = 0; b < batch_size; ++b)
    {
        for (int i = 0; i < input_height; ++i)
        {
            for (int j = 0; j < input_width; ++j)
            {
                input.at(b, 0, i, j) = static_cast<float>((b * 10) + (i * 3) + j);
            }
        }
    }

    nn::Tensor output = conv.forward(input);

    // Verify output shape
    const auto& out_shape = output.get_shape();
    ASSERT_EQ(out_shape[0], batch_size);
    ASSERT_EQ(out_shape[1], out_channels);
    ASSERT_EQ(out_shape[2], 2);
    ASSERT_EQ(out_shape[3], 2);

    // Verify output values for first batch
    // input[0]: [[0,1,2],[3,4,5],[6,7,8]]
    // output[0,0,0,0] = 0+1+3+4 = 8
    ASSERT_NEAR(output.at(0, 0, 0, 0), 8.0, 1e-5);

    // Verify output for last batch
    // input[3]: [[30,31,32],[33,34,35],[36,37,38]]
    // output[3,0,0,0] = 30+31+33+34 = 128
    ASSERT_NEAR(output.at(3, 0, 0, 0), 128.0, 1e-5);
}

// Test: Multi-channel convolution
TEST(Conv2dTest, MultiChannelForward)
{
    const int in_channels = 3;
    const int out_channels = 2;
    const int kernel_size = 3;
    const int batch_size = 1;
    const int input_height = 5;
    const int input_width = 5;

    Conv2d conv(in_channels, out_channels, kernel_size);

    // Initialize weights with known pattern
    nn::Tensor weights(kernel_size * kernel_size * in_channels, out_channels);
    weights.setZero();

    // First output channel: use only first input channel (all 1s)
    for (int i = 0; i < kernel_size * kernel_size; ++i)
    {
        weights.at(i, 0) = 1.0F;
    }

    // Second output channel: use only second input channel (all 2s)
    for (int i = kernel_size * kernel_size; i < 2 * kernel_size * kernel_size; ++i)
    {
        weights.at(i, 1) = 2.0F;
    }

    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.at(0, 0) = 0.5F;
    bias.at(0, 1) = 1.0F;
    conv.set_bias(bias);

    // Create input
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    input.setZero();

    // Set first channel to all 1s
    for (int i = 0; i < input_height; ++i)
    {
        for (int j = 0; j < input_width; ++j)
        {
            input.at(0, 0, i, j) = 1.0F;
        }
    }

    // Set second channel to all 2s
    for (int i = 0; i < input_height; ++i)
    {
        for (int j = 0; j < input_width; ++j)
        {
            input.at(0, 1, i, j) = 2.0F;
        }
    }

    nn::Tensor output = conv.forward(input);

    // Verify shape
    const auto& out_shape = output.get_shape();
    ASSERT_EQ(out_shape[0], batch_size);
    ASSERT_EQ(out_shape[1], out_channels);
    ASSERT_EQ(out_shape[2], 3);
    ASSERT_EQ(out_shape[3], 3);

    // First output channel: sum of (first channel weights * 1) + bias
    // Should be positive value with bias 0.5 added
    ASSERT_GT(output.at(0, 0, 0, 0), 0.0F);

    // Second output channel: sum of (second channel weights * 2) + bias
    // Should be larger due to multiplier
    ASSERT_GT(output.at(0, 1, 0, 0), output.at(0, 0, 0, 0));
}

// Test: Different kernel sizes
TEST(Conv2dTest, DifferentKernelSizes)
{
    const int batch_size = 1;
    const int in_channels = 1;
    const int out_channels = 1;
    const int input_height = 8;
    const int input_width = 8;

    // Test 1x1 kernel
    {
        Conv2d conv(in_channels, out_channels, 1);
        nn::Tensor weights(1, out_channels);
        weights.at(0, 0) = 2.0F;
        conv.set_weights(weights);

        nn::Tensor bias(1, out_channels);
        bias.at(0, 0) = 0.0F;
        conv.set_bias(bias);

        nn::Tensor input(batch_size, in_channels, input_height, input_width);
        input.set_ones();

        nn::Tensor output = conv.forward(input);

        // 1x1 kernel should preserve spatial dimensions
        const auto& out_shape = output.get_shape();
        ASSERT_EQ(out_shape[2], input_height);
        ASSERT_EQ(out_shape[3], input_width);
        // Output should be 1 * 2 = 2
        ASSERT_NEAR(output.at(0, 0, 0, 0), 2.0F, 1e-5);
    }

    // Test 3x3 kernel
    {
        Conv2d conv(in_channels, out_channels, 3);
        nn::Tensor weights(9, out_channels);
        weights.set_ones();
        conv.set_weights(weights);

        nn::Tensor bias(1, out_channels);
        bias.at(0, 0) = 0.0F;
        conv.set_bias(bias);

        nn::Tensor input(batch_size, in_channels, input_height, input_width);
        input.set_ones();

        nn::Tensor output = conv.forward(input);

        // 3x3 kernel on all-ones produces 9 (sum of 9 ones)
        const auto& out_shape = output.get_shape();
        ASSERT_EQ(out_shape[2], 6); // 8 - 3 + 1
        ASSERT_EQ(out_shape[3], 6);
        ASSERT_NEAR(output.at(0, 0, 0, 0), 9.0F, 1e-5);
    }

    // Test 5x5 kernel
    {
        Conv2d conv(in_channels, out_channels, 5);
        nn::Tensor weights(25, out_channels);
        weights.set_ones();
        conv.set_weights(weights);

        nn::Tensor bias(1, out_channels);
        bias.at(0, 0) = 0.0F;
        conv.set_bias(bias);

        nn::Tensor input(batch_size, in_channels, input_height, input_width);
        input.set_ones();

        nn::Tensor output = conv.forward(input);

        // 5x5 kernel on all-ones produces 25
        const auto& out_shape = output.get_shape();
        ASSERT_EQ(out_shape[2], 4); // 8 - 5 + 1
        ASSERT_EQ(out_shape[3], 4);
        ASSERT_NEAR(output.at(0, 0, 0, 0), 25.0F, 1e-5);
    }
}

// Test: Parallel flag control and execution
TEST(Conv2dTest, ParallelExecution)
{
    const int in_channels = 2;
    const int out_channels = 3;
    const int kernel_size = 3;
    const int batch_size = 4;
    const int input_height = 16;
    const int input_width = 16;

    // Test parallel enabled
    Conv2d conv_parallel(in_channels, out_channels, kernel_size, batch_size, true);
    conv_parallel.set_parallel_enabled(true);
    ASSERT_TRUE(conv_parallel.is_parallel_enabled());

    // Test parallel disabled
    Conv2d conv_sequential(in_channels, out_channels, kernel_size, batch_size, false);
    conv_sequential.set_parallel_enabled(false);
    ASSERT_FALSE(conv_sequential.is_parallel_enabled());

    // Initialize with same weights and bias
    nn::Tensor weights(kernel_size * kernel_size * in_channels, out_channels);
    weights.set_ones();
    conv_parallel.set_weights(weights);
    conv_sequential.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.setZero();
    conv_parallel.set_bias(bias);
    conv_sequential.set_bias(bias);

    // Create input
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    input.set_ones();

    // Forward passes - just verify they execute without crashing
    nn::Tensor output_parallel = conv_parallel.forward(input);
    nn::Tensor output_sequential = conv_sequential.forward(input);

    // Verify shapes are correct
    const auto& shape_p = output_parallel.get_shape();
    const auto& shape_s = output_sequential.get_shape();
    ASSERT_EQ(shape_p[0], batch_size);
    ASSERT_EQ(shape_s[0], batch_size);
    ASSERT_EQ(shape_p[1], out_channels);
    ASSERT_EQ(shape_s[1], out_channels);
    ASSERT_EQ(shape_p[2], shape_s[2]);
    ASSERT_EQ(shape_p[3], shape_s[3]);

    // Backward passes
    nn::Tensor grad_output(batch_size, out_channels, shape_p[2], shape_p[3]);
    grad_output.set_ones();

    nn::Tensor grad_input_p = conv_parallel.backward(grad_output);
    nn::Tensor grad_input_s = conv_sequential.backward(grad_output);

    // Verify gradient shapes
    const auto& grad_shape_p = grad_input_p.get_shape();
    const auto& grad_shape_s = grad_input_s.get_shape();
    ASSERT_EQ(grad_shape_p[0], batch_size);
    ASSERT_EQ(grad_shape_s[0], batch_size);
}

// Test: Gradient computation with exact deterministic reference
TEST(Conv2dTest, GradientComputationExact)
{
    const int in_channels = 1;
    const int out_channels = 1;
    const int kernel_size = 2;
    const int batch_size = 1;
    const int input_height = 4;
    const int input_width = 4;

    Conv2d conv(in_channels, out_channels, kernel_size);

    // Simple weights
    nn::Tensor weights(kernel_size * kernel_size * in_channels, out_channels);
    weights.at(0, 0) = 0.5F;
    weights.at(1, 0) = 1.0F;
    weights.at(2, 0) = 1.5F;
    weights.at(3, 0) = 2.0F;
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.at(0, 0) = 0.1F;
    conv.set_bias(bias);

    // Input: deterministic 1..16 grid
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    float value = 1.0F;
    for (int r = 0; r < input_height; ++r)
    {
        for (int c = 0; c < input_width; ++c)
        {
            input.at(0, 0, r, c) = value;
            value += 1.0F;
        }
    }

    // Forward and backward
    conv.forward(input);
    nn::Tensor grad_output(batch_size, out_channels, 3, 3);
    grad_output.set_ones();
    conv.backward(grad_output);

    // Verify exact gradients for 2x2 kernel on 4x4 input with grad_output=ones(3x3).
    auto weight_grad = conv.get_weights().grad();
    auto bias_grad = conv.get_bias().grad();

    // Kernel gradient layout is row-major [k00, k01, k10, k11].
    ASSERT_NEAR(weight_grad.at(0), 54.0F, 1e-5F);
    ASSERT_NEAR(weight_grad.at(1), 63.0F, 1e-5F);
    ASSERT_NEAR(weight_grad.at(2), 90.0F, 1e-5F);
    ASSERT_NEAR(weight_grad.at(3), 99.0F, 1e-5F);

    // Bias gradient should be sum of grad_output
    float expected_bias_grad = grad_output.sum();
    ASSERT_NEAR(bias_grad.at(0), expected_bias_grad, 1e-4F);
}

// Test: Edge case - small input
TEST(Conv2dTest, SmallInputSize)
{
    const int in_channels = 1;
    const int out_channels = 1;
    const int kernel_size = 3;
    const int batch_size = 1;
    const int input_height = 3;
    const int input_width = 3;

    Conv2d conv(in_channels, out_channels, kernel_size);

    nn::Tensor weights(kernel_size * kernel_size * in_channels, out_channels);
    weights.set_ones();
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.at(0, 0) = 0.0F;
    conv.set_bias(bias);

    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    input.set_ones();

    nn::Tensor output = conv.forward(input);

    // 3x3 input with 3x3 kernel = 1x1 output
    const auto& out_shape = output.get_shape();
    ASSERT_EQ(out_shape[2], 1);
    ASSERT_EQ(out_shape[3], 1);
    ASSERT_NEAR(output.at(0, 0, 0, 0), 9.0F, 1e-5);
}

// Test: Parallel flag getter/setter
TEST(Conv2dTest, ParallelFlagControl)
{
    Conv2d conv(1, 1, 2);

    // Default should be true
    ASSERT_TRUE(conv.is_parallel_enabled());

    // Disable
    conv.set_parallel_enabled(false);
    ASSERT_FALSE(conv.is_parallel_enabled());

    // Enable
    conv.set_parallel_enabled(true);
    ASSERT_TRUE(conv.is_parallel_enabled());
}

// Test: Large batch processing
TEST(Conv2dTest, LargeBatchSize)
{
    const int in_channels = 2;
    const int out_channels = 2;
    const int kernel_size = 3;
    const int batch_size = 32;
    const int input_height = 8;
    const int input_width = 8;

    Conv2d conv(in_channels, out_channels, kernel_size, batch_size);

    nn::Tensor weights(kernel_size * kernel_size * in_channels, out_channels);
    test_helpers::rand_fill(weights, -0.5F, 0.5F, 42U);
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    test_helpers::rand_fill(bias, -0.5F, 0.5F, 42U);
    conv.set_bias(bias);

    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    test_helpers::rand_fill(input, -0.5F, 0.5F, 42U);

    nn::Tensor output = conv.forward(input);

    // Verify shape
    const auto& out_shape = output.get_shape();
    ASSERT_EQ(out_shape[0], batch_size);
    ASSERT_EQ(out_shape[1], out_channels);
    ASSERT_EQ(out_shape[2], 6); // 8 - 3 + 1
    ASSERT_EQ(out_shape[3], 6);

    // Backward pass
    nn::Tensor grad_output(batch_size, out_channels, 6, 6);
    test_helpers::rand_fill(grad_output, -0.5F, 0.5F, 42U);

    nn::Tensor grad_input = conv.backward(grad_output);

    // Verify gradient shape
    const auto& grad_shape = grad_input.get_shape();
    ASSERT_EQ(grad_shape[0], batch_size);
    ASSERT_EQ(grad_shape[1], in_channels);
    ASSERT_EQ(grad_shape[2], input_height);
    ASSERT_EQ(grad_shape[3], input_width);
}

// Ensure bias tensors shaped as (out_channels,1) and (1,out_channels) are both accepted
TEST(Conv2dTest, BiasShapeVariants)
{
    const int in_channels = 1;
    const int out_channels = 2;
    const int kernel_size = 1;
    const int batch_size = 1;
    const int input_height = 2;
    const int input_width = 2;

    Conv2d conv_col(in_channels, out_channels, kernel_size);
    Conv2d conv_row(in_channels, out_channels, kernel_size);

    // Weights: simple ones
    nn::Tensor weights(1, out_channels);
    weights.set_ones();
    conv_col.set_weights(weights);
    conv_row.set_weights(weights);

    // Bias as column (out_channels, 1)
    nn::Tensor bias_col(out_channels, 1);
    bias_col.at(0, 0) = 0.5F;
    bias_col.at(1, 0) = 1.0F;
    conv_col.set_bias(bias_col);

    // Bias as row (1, out_channels)
    nn::Tensor bias_row(1, out_channels);
    bias_row.at(0, 0) = 0.5F;
    bias_row.at(0, 1) = 1.0F;
    conv_row.set_bias(bias_row);

    // Input: all ones
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    input.set_ones();

    nn::Tensor out_col = conv_col.forward(input);
    nn::Tensor out_row = conv_row.forward(input);

    // Shapes should match and data should be identical
    ASSERT_EQ(out_col.get_shape(), out_row.get_shape());
    ASSERT_TRUE(test_helpers::tensor_is_approx(out_col, out_row));
}

TEST(Conv2dTest, ForwardThrowsOnNon4DInput)
{
    Conv2d conv(1, 1, 3);
    // 2D tensor (shape size != 4) — must throw invalid_argument
    nn::Tensor input_2d(std::vector<size_t>{4, 4});
    EXPECT_THROW(conv.forward(input_2d), std::invalid_argument);

    // 3D tensor (shape size != 4) — must also throw
    nn::Tensor input_3d(std::vector<size_t>{1, 4, 4});
    EXPECT_THROW(conv.forward(input_3d), std::invalid_argument);
}

TEST(Conv2dTest, ForwardThrowsOnWrongInputChannels)
{
    Conv2d conv(1, 1, 3); // expects 1 input channel
    // 4D tensor with 2 input channels — must throw; shape {batch, channels, H, W}
    nn::Tensor input_wrong_ch(std::vector<size_t>{1, 2, 8, 8});
    EXPECT_THROW(conv.forward(input_wrong_ch), std::invalid_argument);
}

TEST(Conv2dTest, ReusesCachedIndicesAndBuffersOnRepeatedShape)
{
    Conv2d conv(1, 1, 2);

    nn::Tensor input(std::vector<size_t>{1, 1, 3, 3});
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            input.at(0, 0, y, x) = static_cast<float>((y * 3) + x + 1);
        }
    }

    nn::Tensor grad_out(std::vector<size_t>{1, 1, 2, 2});
    grad_out.set_ones();

    auto out1 = conv.forward(input);
    auto in_grad1 = conv.backward(grad_out);
    auto out2 = conv.forward(input);
    auto in_grad2 = conv.backward(grad_out);

    ASSERT_EQ(out1.get_shape(), out2.get_shape());
    ASSERT_EQ(in_grad1.get_shape(), in_grad2.get_shape());
    for (size_t y = 0; y < 2; ++y)
    {
        for (size_t x = 0; x < 2; ++x)
        {
            EXPECT_NEAR(out2.at(0, 0, y, x), out1.at(0, 0, y, x), 1e-7F);
            EXPECT_NEAR(in_grad2.at(0, 0, y, x), in_grad1.at(0, 0, y, x), 1e-7F);
        }
    }
}

TEST(Conv2dTest, ConvenienceCtorAndConstGetters)
{
    Conv2d conv(1, 1, 3, 1, 0, 1);

    const auto& const_conv = std::as_const(conv);
    const auto& weights = const_conv.get_weights();
    const auto& bias = const_conv.get_bias();

    EXPECT_EQ(weights.get_shape().size(), 2U);
    EXPECT_EQ(bias.get_shape().size(), 2U);

    nn::Tensor input(std::vector<size_t>{1, 1, 5, 5});
    input.set_ones();
    auto output = conv.forward(input);
    EXPECT_EQ(output.get_shape(), std::vector<size_t>({1, 1, 3, 3}));
}

// Conv1d basic forward and backward (Conv1d_impl.cpp validation/getters)
TEST(Conv1dTest, BasicForwardBackward)
{
    nn::Conv1d conv(1,
        1,
        2,
        1,
        0,
        1); // in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, dilation=1
    nn::Tensor input(std::vector<size_t>{1, 1, 4}); // batch=1, channels=1, length=4
    input.at(0, 0, 0) = 1.0F;
    input.at(0, 0, 1) = 2.0F;
    input.at(0, 0, 2) = 3.0F;
    input.at(0, 0, 3) = 4.0F;

    auto output = conv.forward(input);
    ASSERT_EQ(output.get_shape().size(), 3); // Should be 3D (B, C_out, L_out)

    nn::Tensor grad_out(output.get_shape());
    grad_out.set_ones();
    auto input_grad = conv.backward(grad_out);
    ASSERT_EQ(input_grad.get_shape(), input.get_shape());
}

// Conv1d forward with wrong input channels (Conv1d_impl.cpp line 70)
TEST(Conv1dTest, ForwardThrowsOnWrongInputChannels)
{
    nn::Conv1d conv(2, 1, 2, 1, 0, 1);              // Expects 2 input channels
    nn::Tensor input(std::vector<size_t>{1, 1, 4}); // Only 1 input channel
    EXPECT_THROW(conv.forward(input), std::invalid_argument);
}

// Conv1d forward with wrong input dimensions (Conv1d_impl.cpp line 63)
TEST(Conv1dTest, ForwardThrowsOnNot3DInput)
{
    nn::Conv1d conv(1, 1, 2, 1, 0, 1);
    nn::Tensor input(std::vector<size_t>{1, 1, 4, 4}); // 4D instead of 3D
    EXPECT_THROW(conv.forward(input), std::invalid_argument);
}

// Conv1d with invalid output length (Conv1d_impl.cpp line 74)
TEST(Conv1dTest, ForwardThrowsOnInvalidOutputLength)
{
    nn::Conv1d conv(1, 1, 5, 1, 0, 1);              // kernel=5, stride=1, padding=0, dilation=1
    nn::Tensor input(std::vector<size_t>{1, 1, 2}); // Only 2 time steps → output would be negative
    EXPECT_THROW(conv.forward(input), std::invalid_argument);
}

// Conv1d getters (Conv1d_impl.cpp getter methods)
TEST(Conv1dTest, GettersReturnCorrectTypes)
{
    nn::Conv1d conv(2, 3, 2, 1, 0, 1);  // in=2, out=3, kernel=2, stride=1, padding=0, dilation=1
    auto& weights = conv.get_weights(); // Calls mutable getter
    const auto& const_weights = std::as_const(conv).get_weights(); // Calls const getter
    EXPECT_EQ(weights.get_shape(), const_weights.get_shape());

    auto& bias = conv.get_bias();                            // Calls mutable bias getter
    const auto& const_bias = std::as_const(conv).get_bias(); // Calls const bias getter
    EXPECT_EQ(bias.get_shape(), const_bias.get_shape());
}
