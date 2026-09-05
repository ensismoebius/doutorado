/**
 * @file layers_composite_gtest.cpp
 * @brief Linear, Sequential, SimpleResNet, Module base-class defaults, and cross-layer
 * comprehensive scenarios.
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

namespace
{
class PassthroughModule final : public Module<nn::Backend>
{
   public:
    auto forward(const Tensor& input, bool /*requires_grad*/ = true) -> Tensor override
    {
        return input;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        return grad_output;
    }
};
} // namespace

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

TEST(LinearTest, OneDimensionalForwardBackwardPath)
{
    Linear layer(3, 2);
    layer.weight.fill(0.0F);
    layer.bias.fill(0.0F);
    layer.weight.at(0, 0) = 1.0F;
    layer.weight.at(0, 1) = 1.0F;
    layer.weight.at(0, 2) = 1.0F;
    layer.weight.at(1, 0) = 2.0F;
    layer.weight.at(1, 1) = 2.0F;
    layer.weight.at(1, 2) = 2.0F;

    nn::Tensor x(std::vector<nn::Index>{3});
    x.at(0) = 1.0F;
    x.at(1) = 2.0F;
    x.at(2) = 3.0F;

    nn::Tensor out = layer.forward(x, true);
    ASSERT_EQ(out.rows(), 1U);
    ASSERT_EQ(out.cols(), 2U);
    EXPECT_NEAR(out.at(0, 0), 6.0F, 1e-5F);
    EXPECT_NEAR(out.at(0, 1), 12.0F, 1e-5F);

    nn::Tensor grad(std::vector<nn::Index>{2});
    grad.at(0) = 1.0F;
    grad.at(1) = 1.0F;
    nn::Tensor dx = layer.backward(grad);
    EXPECT_EQ(dx.size(), 3U);
}

TEST(LinearTest, BackwardThrowsOnWrongGradFeatures)
{
    Linear layer(3, 2);
    nn::Tensor x(1, 3);
    x.fill(1.0F);
    (void) layer.forward(x, true);

    nn::Tensor wrong_grad(1, 3); // out_features is 2
    wrong_grad.fill(1.0F);
    EXPECT_THROW(layer.backward(wrong_grad), std::invalid_argument);
}

TEST(LinearTest, StateDictLoadStateDict)
{
    Linear src(3, 2);
    src.weight.fill(0.0F);
    src.bias.fill(0.0F);
    src.weight.at(0, 0) = 3.0F;
    src.weight.at(1, 2) = -1.5F;
    src.bias.at(0, 0) = 0.25F;
    src.bias.at(1, 0) = -0.75F;

    auto sd = src.state_dict();
    ASSERT_TRUE(sd.find("weight") != sd.end());
    ASSERT_TRUE(sd.find("bias") != sd.end());

    Linear dst(3, 2);
    dst.weight.fill(0.0F);
    dst.bias.fill(0.0F);
    dst.load_state_dict(sd);

    EXPECT_NEAR(dst.weight.at(0, 0), 3.0F, 1e-6F);
    EXPECT_NEAR(dst.weight.at(1, 2), -1.5F, 1e-6F);
    EXPECT_NEAR(dst.bias.at(0, 0), 0.25F, 1e-6F);
    EXPECT_NEAR(dst.bias.at(1, 0), -0.75F, 1e-6F);
}

TEST(ModuleBaseTest, DefaultMethodsAreSafe)
{
    PassthroughModule m;
    EXPECT_NO_THROW(m.train(false));
    EXPECT_NO_THROW(m.train());
    EXPECT_NO_THROW(m.eval());
    EXPECT_NO_THROW(m.reset_state());
    EXPECT_TRUE(m.params().empty());
    EXPECT_TRUE(m.state_dict().empty());

    std::map<std::string, nn::Tensor> empty_sd;
    EXPECT_NO_THROW(m.load_state_dict(empty_sd));

    nn::Device cpu = nn::Device::from_string("cpu");
    auto& returned = m.to(cpu);
    EXPECT_EQ(&returned, &m);
}

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

TEST(SequentialTest, StateDictLoadTrainAndParams)
{
    auto l0 = std::make_shared<Linear>(2, 2);
    auto l1 = std::make_shared<Linear>(2, 1);
    l0->weight.fill(0.0F);
    l0->bias.fill(0.0F);
    l1->weight.fill(0.0F);
    l1->bias.fill(0.0F);

    Sequential seq({l0, l1});

    auto p = seq.params();
    EXPECT_EQ(p.size(), 4U);

    auto sd = seq.state_dict();
    EXPECT_TRUE(sd.find("0.weight") != sd.end());
    EXPECT_TRUE(sd.find("0.bias") != sd.end());
    EXPECT_TRUE(sd.find("1.weight") != sd.end());
    EXPECT_TRUE(sd.find("1.bias") != sd.end());

    nn::Tensor replacement_w(2, 2);
    replacement_w.fill(0.0F);
    replacement_w.at(0, 1) = 4.0F;
    replacement_w.at(1, 0) = -2.0F;

    std::map<std::string, nn::Tensor> update;
    update["0.weight"] = replacement_w;
    update["badkey"] = replacement_w;   // no dot: ignored
    update["9.weight"] = replacement_w; // out-of-range index: ignored
    seq.load_state_dict(update);

    EXPECT_NEAR(l0->weight.at(0, 1), 4.0F, 1e-6F);
    EXPECT_NEAR(l0->weight.at(1, 0), -2.0F, 1e-6F);

    EXPECT_NO_THROW(seq.train(false));
    EXPECT_NO_THROW(seq.train(true));
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

// Additional Comprehensive Tests
TEST(LayerComprehensiveTest, LeakyLayerStateManagement)
{
    Lif leaky(1.0F, 5.0F, 1.0F, 2.0F, true, 0.0F, std::make_shared<ExponentialSurrogate>());

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

    // Step 1: v=0*exp(-1/5)+3=3 > v_th(2) -> hard reset to 0.
    // Step 2: v=0*exp(-1/5)+1=1 (no spike) -> state remains 1.
    EXPECT_NEAR(vmem_after1, 0.0F, 1e-6F);
    EXPECT_NEAR(vmem_after2, 1.0F, 1e-6F);
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

    // Exponential surrogate with sharpness=1:
    // g(v, th) = exp(-|v-th|)
    EXPECT_NEAR(grad_at_zero, std::exp(-1.0F), 1e-6F);
    EXPECT_NEAR(grad_at_one, 1.0F, 1e-6F);
    EXPECT_NEAR(grad_at_minus_one, std::exp(-2.0F), 1e-6F);
}
