#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>

#include "core/initializers/xavier.hpp"
#include "../Conv2d.hpp"
#include "../Leaky.hpp"
#include "../LeakyReLU.hpp"
#include "../Linear.hpp"
#include "../MSELoss.hpp"
#include "../ReLU.hpp"
#include "../Sequential.hpp"
#include "../SpikeCountLoss.hpp"
#include "../SurrogateGradient.hpp"
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
    linear->weight.set_data( (Eigen::MatrixXf(1,2) << 1.0F, 2.0F).finished() );
    linear->bias.set_data( (Eigen::MatrixXf(1,1) << 0.0F).finished() );
    auto relu = std::make_shared<ReLU>();
    Sequential seq({linear, relu});
    Eigen::MatrixXf input(1, 2);
    input << -1.0F, 2.0F;
    nn::Tensor in_tensor{input};
    nn::Tensor out{seq.forward(in_tensor)};
    ASSERT_NEAR(out.at(0, 0), 3.0F, 1e-5F);
    Eigen::MatrixXf grad_out(1, 1);
            grad_out << 1.0F;
            nn::Tensor grad_tensor{grad_out};    nn::Tensor grad_in{seq.backward(grad_tensor)};
    ASSERT_NEAR(grad_in.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad_in.at(0, 1), 2.0F, 1e-5F);
}

// Teste para Linear
TEST(LinearLayerTest, ForwardSimple)
{
    Linear linear(2, 1);
    // Define pesos e bias manualmente para teste determinístico
    linear.weight.set_data( (Eigen::MatrixXf(1,2) << 2.0F, 3.0F).finished() );
    linear.bias.set_data( (Eigen::MatrixXf(1,1) << 1.0F).finished() );
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
    nn::Tensor weights(kernel_size * kernel_size * in_channels, out_channels);
    weights.at(0, 0) = 1.0f;
    weights.at(1, 0) = 2.0f;
    weights.at(2, 0) = 3.0f;
    weights.at(3, 0) = 4.0f;
    conv.set_weights(weights);

    nn::Tensor bias(1, out_channels);
    bias.at(0, 0) = 0.5f;
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
