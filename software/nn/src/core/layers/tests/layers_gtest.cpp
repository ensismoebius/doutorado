#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>

#include "../Conv2d.hpp"
#include "../Leaky.hpp"
#include "../LeakyReLU.hpp"
#include "../Linear.hpp"
#include "../MSELoss.hpp"
#include "../ReLU.hpp"
#include "../Regularization.hpp"
#include "../Sequential.hpp"
#include "../SpikeCountLoss.hpp"
#include "../SurrogateGradient.hpp"
#include "core/initializers/xavier.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/tensor/Tensor.hpp"
// Teste para MSELoss
TEST(MSELossTest, ForwardAndBackward)
{
    MSELoss mse;
    Eigen::MatrixXf pred(2, 1);
    pred << 1.0F, 2.0F;
    Eigen::MatrixXf target(2, 1);
    target << 0.0F, 2.0F;
    nn::Tensor pred_tensor{pred};
    nn::Tensor target_tensor{target};
    mse.set_target(target_tensor);
    nn::Tensor loss{mse.forward(pred_tensor)};
    ASSERT_NEAR(loss.at(0, 0), 0.5F, 1e-5F);
    nn::Tensor grad{mse.backward(pred_tensor)};
    ASSERT_NEAR(grad.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad.at(1, 0), 0.0F, 1e-5F);
}

// Teste para Sequential
TEST(SequentialTest, ForwardAndBackward)
{
    auto linear = std::make_shared<Linear>(2, 1);
    linear->weight.set_data((Eigen::MatrixXf(1, 2) << 1.0F, 2.0F).finished());
    linear->bias.set_data((Eigen::MatrixXf(1, 1) << 0.0F).finished());
    auto relu = std::make_shared<ReLU>();
    Sequential seq({linear, relu});
    Eigen::MatrixXf input(1, 2);
    input << -1.0F, 2.0F;
    nn::Tensor in_tensor{input};
    nn::Tensor out{seq.forward(in_tensor)};
    ASSERT_NEAR(out.at(0, 0), 3.0F, 1e-5F);
    Eigen::MatrixXf grad_out(1, 1);
    grad_out << 1.0F;
    nn::Tensor grad_tensor{grad_out};
    nn::Tensor grad_in{seq.backward(grad_tensor)};
    ASSERT_NEAR(grad_in.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad_in.at(0, 1), 2.0F, 1e-5F);
}

// Teste para Linear
TEST(LinearLayerTest, ForwardSimple)
{
    Linear linear(2, 1);
    // Define pesos e bias manualmente para teste determinístico
    linear.weight.set_data((Eigen::MatrixXf(1, 2) << 2.0F, 3.0F).finished());
    linear.bias.set_data((Eigen::MatrixXf(1, 1) << 1.0F).finished());
    Eigen::MatrixXf input(1, 2);
    input << 1.0F, 2.0F;
    nn::Tensor in_tensor{input};
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
    Eigen::MatrixXf input(1, 1);
    input << 3.0F; // Acima do threshold
    nn::Tensor in_tensor{input};
    nn::Tensor out{leaky.forward(in_tensor)};
    // Como input > threshold, deve gerar spike (1.0)
    ASSERT_FLOAT_EQ(out.at(0, 0), 1.0F);
    // Após spike, v_mem deve ser resetado para zero
    ASSERT_FLOAT_EQ(leaky.v_mem(0, 0), 0.0F);
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
    Eigen::MatrixXf input(1, 1);
    input << 3.0F; // Acima do threshold
    nn::Tensor in_tensor{input};
    nn::Tensor out{leaky.forward(in_tensor)};
    // Deve gerar spike
    ASSERT_FLOAT_EQ(out.at(0, 0), 1.0F);
    // v_mem deve ser reduzido pelo threshold
    ASSERT_FLOAT_EQ(leaky.v_mem(0, 0), 1.0F); // 3.0 - 2.0
}

// Teste para LeakyReLU
TEST(LeakyReLUTest, ForwardAndBackward)
{
    LeakyReLU leaky_relu(0.1F);
    Eigen::MatrixXf input(2, 1);
    input << 1.0F, -2.0F;
    nn::Tensor in_tensor{input};
    nn::Tensor out{leaky_relu.forward(in_tensor)};
    ASSERT_NEAR(out.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(out.at(1, 0), -0.2F, 1e-5F);
    Eigen::MatrixXf grad_out(2, 1);
    grad_out << 1.0F, 1.0F;
    nn::Tensor grad_tensor{grad_out};
    nn::Tensor grad_in{leaky_relu.backward(grad_tensor)};
    ASSERT_NEAR(grad_in.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad_in.at(1, 0), 0.1F, 1e-5F);
}

// Teste para SpikeCountLoss
TEST(SpikeCountLossTest, ForwardAndBackward)
{
    SpikeCountLoss spike_loss;
    Eigen::MatrixXf pred(2, 1);
    pred << 10.0F, 20.0F;
    Eigen::MatrixXf target(2, 1);
    target << 8.0F, 22.0F;
    nn::Tensor pred_tensor{pred};
    nn::Tensor target_tensor{target};
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
    Eigen::MatrixXf v_mem(1, 1);
    v_mem << 2.1F;
    Eigen::MatrixXf grad = surrogate.calculate(v_mem, 2.0F);
    ASSERT_NEAR(grad(0, 0), 0.9048374, 1e-5F);
}

TEST(SurrogateGradientTest, Boxcar)
{
    BoxcarSurrogate surrogate(0.5F);
    Eigen::MatrixXf v_mem(1, 2);
    v_mem << 2.1F, 2.3F;
    Eigen::MatrixXf grad = surrogate.calculate(v_mem, 2.0F);
    ASSERT_FLOAT_EQ(grad(0, 0), 1.0F);
    ASSERT_FLOAT_EQ(grad(0, 1), 0.0F);
}

TEST(Conv2dTest, ForwardAndBackward)
{
    // Layer parameters
    const int in_channels = 1;
    const int out_channels = 1;
    const int kernel_size = 2;
    const int batch_size = 1;
    const int input_height = 3;
    const int input_width = 3;

    // Create layer
    Conv2d conv(in_channels, out_channels, kernel_size);

    // Initialize weights and bias
    nn::Tensor weights(static_cast<Eigen::Index>(kernel_size * kernel_size * in_channels),
                       out_channels);
    weights.at(0, 0) = 1.0F;
    weights.at(1, 0) = 2.0F;
    weights.at(2, 0) = 3.0F;
    weights.at(3, 0) = 4.0F;
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.at(0, 0) = 0.5F;
    conv.set_bias(bias);

    // Input tensor
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
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

    // Backward pass
    nn::Tensor grad_output(batch_size, out_channels, 2, 2);
    grad_output.get_data_ref().setOnes(); // Gradient of 1 for all output elements

    nn::Tensor grad_input = conv.backward(grad_output);

    // Check bias gradient
    ASSERT_NEAR(conv.get_bias().get_grad_ref()(0), 4.0, 1e-5); // sum of grad_output = 1+1+1+1

    // Check weights gradient
    // grad_w[0] = 1*1 + 2*1 + 4*1 + 5*1 = 12
    // grad_w[1] = 2*1 + 3*1 + 5*1 + 6*1 = 16
    // grad_w[2] = 4*1 + 5*1 + 7*1 + 8*1 = 24
    // grad_w[3] = 5*1 + 6*1 + 8*1 + 9*1 = 28
    ASSERT_NEAR(conv.get_weights().get_grad_ref()(0), 12, 1e-5);
    ASSERT_NEAR(conv.get_weights().get_grad_ref()(1), 16, 1e-5);
    ASSERT_NEAR(conv.get_weights().get_grad_ref()(2), 24, 1e-5);
    ASSERT_NEAR(conv.get_weights().get_grad_ref()(3), 28, 1e-5);

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
    nn::Tensor weights(static_cast<Eigen::Index>(kernel_size * kernel_size * in_channels),
                       out_channels);
    weights.get_data_ref().setOnes();
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
    nn::Tensor weights(static_cast<Eigen::Index>(kernel_size * kernel_size * in_channels),
                       out_channels);
    weights.get_data_ref().setZero();

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
    input.get_data_ref().setZero();

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
        input.get_data_ref().setOnes();

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
        weights.get_data_ref().setOnes();
        conv.set_weights(weights);

        nn::Tensor bias(1, out_channels);
        bias.at(0, 0) = 0.0F;
        conv.set_bias(bias);

        nn::Tensor input(batch_size, in_channels, input_height, input_width);
        input.get_data_ref().setOnes();

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
        weights.get_data_ref().setOnes();
        conv.set_weights(weights);

        nn::Tensor bias(1, out_channels);
        bias.at(0, 0) = 0.0F;
        conv.set_bias(bias);

        nn::Tensor input(batch_size, in_channels, input_height, input_width);
        input.get_data_ref().setOnes();

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
    nn::Tensor weights(static_cast<Eigen::Index>(kernel_size * kernel_size * in_channels),
                       out_channels);
    weights.get_data_ref().setOnes();
    conv_parallel.set_weights(weights);
    conv_sequential.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.get_data_ref().setZero();
    conv_parallel.set_bias(bias);
    conv_sequential.set_bias(bias);

    // Create input
    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    input.get_data_ref().setOnes();

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
    grad_output.get_data_ref().setOnes();

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
    nn::Tensor weights(static_cast<Eigen::Index>(kernel_size * kernel_size * in_channels),
                       out_channels);
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
    input.get_data_ref().setRandom();

    // Forward and backward
    conv.forward(input);
    nn::Tensor grad_output(batch_size, out_channels, 3, 3);
    grad_output.get_data_ref().setRandom();
    conv.backward(grad_output);

    // Verify gradients were computed (non-zero)
    auto& weight_grad = conv.get_weights().get_grad_ref();
    auto& bias_grad = conv.get_bias().get_grad_ref();

    // Verify that at least some gradients are non-zero
    bool has_nonzero_weight_grad = false;
    for (int i = 0; i < weight_grad.size(); ++i)
    {
        if (std::abs(weight_grad(i)) > 1e-6F)
        {
            has_nonzero_weight_grad = true;
            break;
        }
    }
    ASSERT_TRUE(has_nonzero_weight_grad);

    // Bias gradient should be sum of grad_output
    float expected_bias_grad = grad_output.get_data_ref().sum();
    ASSERT_NEAR(bias_grad(0), expected_bias_grad, 1e-4F);
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

    nn::Tensor weights(static_cast<Eigen::Index>(kernel_size * kernel_size * in_channels),
                       out_channels);
    weights.get_data_ref().setOnes();
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.at(0, 0) = 0.0F;
    conv.set_bias(bias);

    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    input.get_data_ref().setOnes();

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

    nn::Tensor weights(static_cast<Eigen::Index>(kernel_size * kernel_size * in_channels),
                       out_channels);
    weights.get_data_ref().setRandom();
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.get_data_ref().setRandom();
    conv.set_bias(bias);

    nn::Tensor input(batch_size, in_channels, input_height, input_width);
    input.get_data_ref().setRandom();

    nn::Tensor output = conv.forward(input);

    // Verify shape
    const auto& out_shape = output.get_shape();
    ASSERT_EQ(out_shape[0], batch_size);
    ASSERT_EQ(out_shape[1], out_channels);
    ASSERT_EQ(out_shape[2], 6); // 8 - 3 + 1
    ASSERT_EQ(out_shape[3], 6);

    // Backward pass
    nn::Tensor grad_output(batch_size, out_channels, 6, 6);
    grad_output.get_data_ref().setRandom();

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
    weights.get_data_ref().setOnes();
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
    input.get_data_ref().setOnes();

    nn::Tensor out_col = conv_col.forward(input);
    nn::Tensor out_row = conv_row.forward(input);

    // Shapes should match and data should be identical
    ASSERT_EQ(out_col.get_shape(), out_row.get_shape());
    ASSERT_TRUE(out_col.get_data_ref().isApprox(out_row.get_data_ref()));
}

// Test for L1Regularization
TEST(L1RegularizationTest, Forward)
{
    L1Regularization reg(0.1F);
    nn::Tensor param1{Eigen::MatrixXf::Constant(2, 2, 1.0F)};
    nn::Tensor param2{Eigen::MatrixXf::Constant(1, 3, -2.0F)};
    std::vector<nn::Tensor*> params = {&param1, &param2};

    nn::Tensor loss = reg.forward(params);
    // |1|*4 + |-2|*3 = 4 + 6 = 10, times 0.1 = 1.0
    ASSERT_NEAR(loss.at(0, 0), 1.0F, 1e-5F);
}

TEST(L1RegularizationTest, Backward)
{
    L1Regularization reg(0.5F);
    nn::Tensor param1{Eigen::MatrixXf::Constant(2, 2, 1.0F)};
    nn::Tensor param2{Eigen::MatrixXf::Constant(1, 3, -2.0F)};
    param1.zero_grad();
    param2.zero_grad();
    std::vector<nn::Tensor*> params = {&param1, &param2};

    reg.backward(params);
    // grad for param1: sign(1)*0.5 = 0.5
    ASSERT_TRUE(param1.get_grad_ref().isApprox(Eigen::MatrixXf::Constant(2, 2, 0.5F)));
    // grad for param2: sign(-2)*0.5 = -0.5
    ASSERT_TRUE(param2.get_grad_ref().isApprox(Eigen::MatrixXf::Constant(1, 3, -0.5F)));
}

// Test for L2Regularization
TEST(L2RegularizationTest, Forward)
{
    L2Regularization reg(0.1F);
    nn::Tensor param1{Eigen::MatrixXf::Constant(2, 2, 1.0F)};
    nn::Tensor param2{Eigen::MatrixXf::Constant(1, 3, 2.0F)};
    std::vector<nn::Tensor*> params = {&param1, &param2};

    nn::Tensor loss = reg.forward(params);
    // 1^2*4 + 2^2*3 = 4 + 12 = 16, times 0.1 = 1.6
    ASSERT_NEAR(loss.at(0, 0), 1.6F, 1e-5F);
}

TEST(L2RegularizationTest, Backward)
{
    L2Regularization reg(0.5F);
    nn::Tensor param1{Eigen::MatrixXf::Constant(2, 2, 1.0F)};
    nn::Tensor param2{Eigen::MatrixXf::Constant(1, 3, 2.0F)};
    param1.zero_grad();
    param2.zero_grad();
    std::vector<nn::Tensor*> params = {&param1, &param2};

    reg.backward(params);
    // grad for param1: 2*1*0.5 = 1.0
    ASSERT_TRUE(param1.get_grad_ref().isApprox(Eigen::MatrixXf::Constant(2, 2, 1.0F)));
    // grad for param2: 2*2*0.5 = 2.0
    ASSERT_TRUE(param2.get_grad_ref().isApprox(Eigen::MatrixXf::Constant(1, 3, 2.0F)));
}
