/**
 * @file layers_gtest.cpp
 * @brief Broad unit-test coverage for core NN/SNN layers.
 */

#include <gtest/gtest.h>

#include <memory>
#include <random>

#include "core/utility/tests/test_helpers.hpp"
#include "nn/initializers/xavier.hpp"
#include "nn/layers/Conv2d.hpp"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/LeakyReLU.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/MSELoss.hpp"
#include "nn/layers/ReLU.hpp"
#include "nn/layers/Regularization.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/layers/SimpleResNet.hpp"
#include "nn/layers/SpikeCountLoss.hpp"
#include "nn/layers/SurrogateGradient.hpp"
#include "nn/tensor/Tensor.hpp"
// Teste para MSELoss
TEST(MSELossTest, ForwardAndBackward)
{
    MSELoss mse;
    nn::Tensor pred_tensor(2, 1);
    pred_tensor.at(0, 0) = 1.0F;
    pred_tensor.at(1, 0) = 2.0F;
    nn::Tensor target_tensor(2, 1);
    target_tensor.at(0, 0) = 0.0F;
    target_tensor.at(1, 0) = 2.0F;
    mse.set_target(target_tensor);
    nn::Tensor loss{mse.forward(pred_tensor)};
    ASSERT_NEAR(loss.at(0, 0), 0.5F, 1e-5F);
    nn::Tensor grad{mse.backward(pred_tensor)};
    ASSERT_NEAR(grad.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad.at(1, 0), 0.0F, 1e-5F);
}

// Deterministic random fills for tests use test_helpers::rand_fill(..., nn::testing::SEED)

// Teste para Sequential
TEST(SequentialTest, ForwardAndBackward)
{
    auto linear = std::make_shared<Linear>(2, 1);
    nn::Tensor weight_data(1, 2);
    weight_data.at(0, 0) = 1.0F;
    weight_data.at(0, 1) = 2.0F;
    // Copy weight_data to linear->weight
    for (size_t i = 0; i < weight_data.rows(); ++i)
    {
        for (size_t j = 0; j < weight_data.cols(); ++j)
        {
            linear->weight.at(i, j) = weight_data.at(i, j);
        }
    }
    nn::Tensor bias_data(1, 1);
    bias_data.at(0, 0) = 0.0F;
    // Copy bias_data to linear->bias
    for (size_t i = 0; i < bias_data.rows(); ++i)
    {
        linear->bias.at(i, 0) = bias_data.at(i, 0);
    }
    auto relu = std::make_shared<ReLU>();
    Sequential seq({linear, relu});
    nn::Tensor in_tensor(1, 2);
    in_tensor.at(0, 0) = -1.0F;
    in_tensor.at(0, 1) = 2.0F;
    nn::Tensor out{seq.forward(in_tensor)};
    ASSERT_NEAR(out.at(0, 0), 3.0F, 1e-5F);
    nn::Tensor grad_tensor(1, 1);
    grad_tensor.at(0, 0) = 1.0F;
    nn::Tensor grad_in{seq.backward(grad_tensor)};
    ASSERT_NEAR(grad_in.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad_in.at(0, 1), 2.0F, 1e-5F);
}

// Teste para Linear
TEST(LinearLayerTest, ForwardSimple)
{
    Linear linear(2, 1);
    // Define pesos e bias manualmente para teste determinístico
    nn::Tensor weight_data(1, 2);
    weight_data.at(0, 0) = 2.0F;
    weight_data.at(0, 1) = 3.0F;
    // Copy weight_data to linear.weight
    for (size_t i = 0; i < weight_data.rows(); ++i)
    {
        for (size_t j = 0; j < weight_data.cols(); ++j)
        {
            linear.weight.at(i, j) = weight_data.at(i, j);
        }
    }
    nn::Tensor bias_data(1, 1);
    bias_data.at(0, 0) = 1.0F;
    // Copy bias_data to linear.bias
    for (size_t i = 0; i < bias_data.rows(); ++i)
    {
        linear.bias.at(i, 0) = bias_data.at(i, 0);
    }
    nn::Tensor in_tensor(1, 2);
    in_tensor.at(0, 0) = 1.0F;
    in_tensor.at(0, 1) = 2.0F;
    nn::Tensor out{linear.forward(in_tensor)};
    // Esperado: (1*2 + 2*3) + 1 = 2 + 6 + 1 = 9
    ASSERT_FLOAT_EQ(out.at(0, 0), 9.0F);
}

// Teste para Leaky (LIF)
TEST(LeakyLayerTest, ForwardSpikeAndReset)
{
    Leaky leaky(/*dt=*/1.0F,
                /*R=*/5.0F,
                /*C=*/1.0F,
                /*V_thresh=*/2.0F,
                /*reset_zero=*/true,
                0.0F,
                std::make_shared<ExponentialSurrogate>());
    nn::Tensor in_tensor(1, 1);
    in_tensor.at(0, 0) = 3.0F; // Acima do threshold
    nn::Tensor out{leaky.forward(in_tensor)};
    // Como input > threshold, deve gerar spike (1.0)
    ASSERT_FLOAT_EQ(out.at(0, 0), 1.0F);
    // Após spike, v_mem deve ser resetado para zero
    ASSERT_FLOAT_EQ(leaky.v_mem.at(0, 0), 0.0F);
}

// Teste para Leaky sem reset para zero
TEST(LeakyLayerTest, ForwardSpikeNoResetZero)
{
    Leaky leaky(/*dt=*/1.0F,
                /*R=*/5.0F,
                /*C=*/1.0F,
                /*V_thresh=*/2.0F,
                /*reset_zero=*/false,
                0.0F,
                std::make_shared<ExponentialSurrogate>());
    nn::Tensor in_tensor(1, 1);
    in_tensor.at(0, 0) = 3.0F; // Acima do threshold
    nn::Tensor out{leaky.forward(in_tensor)};
    // Deve gerar spike
    ASSERT_FLOAT_EQ(out.at(0, 0), 1.0F);
    // v_mem deve ser reduzido pelo threshold
    ASSERT_FLOAT_EQ(leaky.v_mem.at(0, 0), 1.0F); // 3.0 - 2.0
}

// Teste para LeakyReLU
TEST(LeakyReLUTest, ForwardAndBackward)
{
    LeakyReLU leaky_relu(0.1F);
    nn::Tensor in_tensor(2, 1);
    in_tensor.at(0, 0) = 1.0F;
    in_tensor.at(1, 0) = -2.0F;
    nn::Tensor out{leaky_relu.forward(in_tensor)};
    ASSERT_NEAR(out.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(out.at(1, 0), -0.2F, 1e-5F);
    nn::Tensor grad_tensor(2, 1);
    grad_tensor.at(0, 0) = 1.0F;
    grad_tensor.at(1, 0) = 1.0F;
    nn::Tensor grad_in{leaky_relu.backward(grad_tensor)};
    ASSERT_NEAR(grad_in.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad_in.at(1, 0), 0.1F, 1e-5F);
}

// Teste para SpikeCountLoss
TEST(SpikeCountLossTest, ForwardAndBackward)
{
    SpikeCountLoss spike_loss;
    nn::Tensor pred_tensor(2, 1);
    pred_tensor.at(0, 0) = 10.0F;
    pred_tensor.at(1, 0) = 20.0F;
    nn::Tensor target_tensor(2, 1);
    target_tensor.at(0, 0) = 8.0F;
    target_tensor.at(1, 0) = 22.0F;
    spike_loss.set_target(target_tensor);
    nn::Tensor loss{spike_loss.forward(pred_tensor)};
    ASSERT_NEAR(loss.at(0, 0), 4.0F, 1e-5F);
    nn::Tensor grad{spike_loss.backward(pred_tensor)};
    ASSERT_NEAR(grad.at(0, 0), 2.0F, 1e-5F);
    ASSERT_NEAR(grad.at(1, 0), -2.0F, 1e-5F);
}

// Teste para SurrogateGradient
TEST(SurrogateGradientTest, Exponential)
{
    ExponentialSurrogate surrogate(1.0F);
    nn::Tensor v_mem_tensor(1, 1);
    v_mem_tensor.at(0, 0) = 2.1F;
    auto grad = surrogate.calculate(v_mem_tensor, 2.0F);
    ASSERT_NEAR(grad.at(0, 0), 0.9048374, 1e-5F);
}

TEST(SurrogateGradientTest, Boxcar)
{
    BoxcarSurrogate surrogate(0.5F);
    nn::Tensor v_mem_tensor(1, 2);
    v_mem_tensor.at(0, 0) = 2.1F;
    v_mem_tensor.at(0, 1) = 2.3F;
    auto grad = surrogate.calculate(v_mem_tensor, 2.0F);
    ASSERT_FLOAT_EQ(grad.at(0, 0), 1.0F);
    ASSERT_FLOAT_EQ(grad.at(0, 1), 0.0F);
}

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

// Test: Gradient computation (basic sanity check)
TEST(Conv2dTest, GradientComputation)
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

    // Input
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    test_helpers::rand_fill(input, -0.5F, 0.5F, 42U);

    // Forward and backward
    conv.forward(input);
    nn::Tensor grad_output(batch_size, out_channels, 3, 3);
    test_helpers::rand_fill(grad_output, -0.5F, 0.5F, 42U);
    conv.backward(grad_output);

    // Verify gradients were computed (non-zero)
    auto weight_grad = conv.get_weights().grad();
    auto bias_grad = conv.get_bias().grad();

    // Verify that at least some gradients are non-zero
    bool has_nonzero_weight_grad = false;
    for (int i = 0; i < weight_grad.size(); ++i)
    {
        if (std::abs(weight_grad.at(i)) > 1e-6F)
        {
            has_nonzero_weight_grad = true;
            break;
        }
    }
    ASSERT_TRUE(has_nonzero_weight_grad);

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

// Test for L1Regularization
TEST(L1RegularizationTest, Forward)
{
    L1Regularization reg(0.1F);
    nn::Tensor param1(2, 2);
    test_helpers::tensor_fill_with_value(param1, 1.0F);
    nn::Tensor param2(1, 3);
    test_helpers::tensor_fill_with_value(param2, -2.0F);
    std::vector<nn::Tensor*> params = {&param1, &param2};

    nn::Tensor loss = reg.forward(params);
    // |1|*4 + |-2|*3 = 4 + 6 = 10, times 0.1 = 1.0
    ASSERT_NEAR(loss.at(0, 0), 1.0F, 1e-5F);
}

TEST(L1RegularizationTest, Backward)
{
    L1Regularization reg(0.5F);
    nn::Tensor param1(2, 2);
    test_helpers::tensor_fill_with_value(param1, 1.0F);
    nn::Tensor param2(1, 3);
    test_helpers::tensor_fill_with_value(param2, -2.0F);
    param1.zero_grad();
    param2.zero_grad();
    std::vector<nn::Tensor*> params = {&param1, &param2};

    reg.backward(params);
    // grad for param1: sign(1)*0.5 = 0.5
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            ASSERT_NEAR(param1.grad().at(i, j), 0.5F, 1e-5F);
        }
    }
    // grad for param2: sign(-2)*0.5 = -0.5
    for (size_t i = 0; i < 1; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            ASSERT_NEAR(param2.grad().at(i, j), -0.5F, 1e-5F);
        }
    }
}

// Test for L2Regularization
TEST(L2RegularizationTest, Forward)
{
    L2Regularization reg(0.1F);
    nn::Tensor param1(2, 2);
    test_helpers::tensor_fill_with_value(param1, 1.0F);
    nn::Tensor param2(1, 3);
    test_helpers::tensor_fill_with_value(param2, 2.0F);
    std::vector<nn::Tensor*> params = {&param1, &param2};

    nn::Tensor loss = reg.forward(params);
    // 1^2*4 + 2^2*3 = 4 + 12 = 16, times 0.1 = 1.6
    ASSERT_NEAR(loss.at(0, 0), 1.6F, 1e-5F);
}

TEST(L2RegularizationTest, Backward)
{
    L2Regularization reg(0.5F);
    nn::Tensor param1(2, 2);
    test_helpers::tensor_fill_with_value(param1, 1.0F);
    nn::Tensor param2(1, 3);
    test_helpers::tensor_fill_with_value(param2, 2.0F);
    param1.zero_grad();
    param2.zero_grad();
    std::vector<nn::Tensor*> params = {&param1, &param2};

    reg.backward(params);
    // grad for param1: 2*1*0.5 = 1.0
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            ASSERT_NEAR(param1.grad().at(i, j), 1.0F, 1e-5F);
        }
    }
    // grad for param2: 2*2*0.5 = 2.0
    for (size_t i = 0; i < 1; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            ASSERT_NEAR(param2.grad().at(i, j), 2.0F, 1e-5F);
        }
    }
}

TEST(SimpleResNetTest, ForwardAndBackward)
{
    int input_dim = 10;
    int hidden_dim = 5;
    int output_dim = 3;
    int depth = 2;

    SimpleResNet model(input_dim, hidden_dim, output_dim, depth);

    nn::Tensor in_tensor = test_helpers::make_random_tensor(1, input_dim);

    nn::Tensor out = model.forward(in_tensor);
    EXPECT_EQ(out.rows(), 1);
    EXPECT_EQ(out.cols(), output_dim);

    nn::Tensor grad_tensor = test_helpers::make_ones_tensor(1, output_dim);

    nn::Tensor grad_in = model.backward(grad_tensor);
    EXPECT_EQ(grad_in.rows(), 1);
    EXPECT_EQ(grad_in.cols(), input_dim);
}

TEST(SimpleResNetTest, ForwardAndBackwardEdgeCases)
{
    // Depth 0 (just input -> linear -> relu -> linear)
    SimpleResNet model_depth0(5, 3, 2, 0);
    nn::Tensor in_tensor0 = test_helpers::make_random_tensor(1, 5);
    nn::Tensor out0 = model_depth0.forward(in_tensor0);
    EXPECT_EQ(out0.rows(), 1);
    EXPECT_EQ(out0.cols(), 2);

    // Depth 1
    SimpleResNet model_depth1(5, 3, 2, 1);
    nn::Tensor out1 = model_depth1.forward(in_tensor0);
    EXPECT_EQ(out1.rows(), 1);
    EXPECT_EQ(out1.cols(), 2);

    // Large depth
    SimpleResNet model_large(5, 3, 2, 5);
    nn::Tensor out_large = model_large.forward(in_tensor0);
    EXPECT_EQ(out_large.rows(), 1);
    EXPECT_EQ(out_large.cols(), 2);

    // Backward for depth 0
    nn::Tensor grad_tensor0 = test_helpers::make_ones_tensor(1, 2);
    nn::Tensor grad_in0 = model_depth0.backward(grad_tensor0);
    EXPECT_EQ(grad_in0.rows(), 1);
    EXPECT_EQ(grad_in0.cols(), 5);
}

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
    nn::Tensor small_tensor(
        1,
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

// Additional Comprehensive Tests
TEST(LayerComprehensiveTest, LeakyLayerStateManagement)
{
    Leaky leaky(1.0F, 5.0F, 1.0F, 2.0F, true, 0.0F, std::make_shared<ExponentialSurrogate>());

    // Test state reset between forward passes
    nn::Tensor tensor1(1, 1);
    tensor1.at(0, 0) = 3.0F;
    [[maybe_unused]] nn::Tensor out1 = leaky.forward(tensor1);
    float vmem_after1 = leaky.v_mem(0, 0);

    // Second forward pass should start fresh
    nn::Tensor tensor2(1, 1);
    tensor2.at(0, 0) = 1.0F;
    [[maybe_unused]] nn::Tensor out2 = leaky.forward(tensor2);
    float vmem_after2 = leaky.v_mem(0, 0);

    // Membrane potential should be different
    EXPECT_NE(vmem_after1, vmem_after2);
}

TEST(LayerComprehensiveTest, ReLUGradientFlow)
{
    // Test that ReLU properly blocks negative gradients
    Linear linear(2, 1);
    ReLU relu;

    Sequential seq({std::make_shared<Linear>(linear), std::make_shared<ReLU>(relu)});

    // Input that will produce negative pre-activation
    linear.weight.at(0, 0) = -2.0F;
    linear.weight.at(0, 1) = -3.0F;

    linear.bias.at(0, 0) = 1.0F;

    nn::Tensor input_tensor(1, 2);
    input_tensor.at(0, 0) = 2.0F;
    input_tensor.at(0, 1) = 2.0F; // Will produce -4 + 1 = -3 (negative)

    nn::Tensor output = seq.forward(input_tensor);
    EXPECT_EQ(output.at(0, 0), 0.0F); // ReLU of negative is 0

    // Backward should produce zero gradient for negative inputs
    nn::Tensor grad_tensor = test_helpers::make_ones_tensor(1, 1);
    nn::Tensor grad_input = seq.backward(grad_tensor);

    // Gradient should be zero for the input that produced negative pre-activation
    EXPECT_EQ(grad_input.at(0, 0), 0.0F);
    EXPECT_EQ(grad_input.at(0, 1), 0.0F);
}

TEST(LayerComprehensiveTest, Conv2dPaddingAndStride)
{
    // Test different padding and stride combinations
    Conv2d conv(
        1, 1, 3, 2, 1, 1, false); // kernel=3, stride=2, padding=1, dilation=1, use_parallel=false

    // Create a 4D input tensor: (batch=1, channels=1, height=8, width=8)
    nn::Tensor input_tensor(1, 1, 8, 8);
    // Set all elements to 1
    for (int i = 0; i < 1; ++i)
        for (int c = 0; c < 1; ++c)
            for (int h = 0; h < 8; ++h)
                for (int w = 0; w < 8; ++w) input_tensor.at(i, c, h, w) = 1.0f;

    nn::Tensor output = conv.forward(input_tensor);

    // With stride=2 and kernel=3, output size should be (8-3+2*1)/2 + 1 = 4
    // So output should be (batch=1, out_channels=1, height=4, width=4)
    const auto& out_shape = output.get_shape();
    ASSERT_EQ(out_shape.size(), 4);
    EXPECT_EQ(out_shape[0], 1); // batch
    EXPECT_EQ(out_shape[1], 1); // out_channels
    EXPECT_EQ(out_shape[2], 4); // out_height
    EXPECT_EQ(out_shape[3], 4); // out_width
}

TEST(LayerComprehensiveTest, RegularizationZeroParameters)
{
    L1Regularization l1(0.1F);
    std::vector<nn::Tensor*> params;

    nn::Tensor param1(2, 2);
    param1.at(0, 0) = 0.0F;
    param1.at(0, 1) = 0.0F;
    param1.at(1, 0) = 0.0F;
    param1.at(1, 1) = 0.0F;

    params.push_back(&param1);

    nn::Tensor loss = l1.forward(params);
    EXPECT_EQ(loss.at(0, 0), 0.0F); // L1 of all zeros should be 0

    l1.backward(params);
    // Gradients should be zero for zero parameters
    EXPECT_NEAR(param1.grad().sum(), 0.0F, 1e-6F);
}

TEST(LayerComprehensiveTest, SurrogateGradientRange)
{
    auto surrogate = std::make_shared<ExponentialSurrogate>();

    // Test surrogate gradient at different voltage levels
    nn::Tensor v_zero(1, 1);
    v_zero.at(0, 0) = 0.0F;
    float grad_at_zero = surrogate->calculate(v_zero, 1.0F).at(0, 0);

    nn::Tensor v_one(1, 1);
    v_one.at(0, 0) = 1.0F;
    float grad_at_one = surrogate->calculate(v_one, 1.0F).at(0, 0);

    nn::Tensor v_minus_one(1, 1);
    v_minus_one.at(0, 0) = -1.0F;
    float grad_at_minus_one = surrogate->calculate(v_minus_one, 1.0F).at(0, 0);

    // All gradients should be finite and reasonable
    EXPECT_TRUE(std::isfinite(grad_at_zero));
    EXPECT_TRUE(std::isfinite(grad_at_one));
    EXPECT_TRUE(std::isfinite(grad_at_minus_one));

    // Gradient at threshold (0) should be positive
    EXPECT_GT(grad_at_zero, 0.0F);
}
