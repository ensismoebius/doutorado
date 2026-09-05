/**
 * @file layers_losses_regularization_gtest.cpp
 * @brief MSE/MAE/SpikeCount loss and L1/L2 regularization unit tests.
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

TEST(MSELossTest, TrainToggleAndNonFiniteBackward)
{
    MSELoss mse;
    nn::Tensor pred(1, 1);
    pred.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    nn::Tensor target(1, 1);
    target.at(0, 0) = 0.0F;

    mse.train(false);
    EXPECT_THROW(mse.forward(pred), std::runtime_error);

    mse.set_target(target);
    mse.train(true);
    (void) mse.forward(pred, true);
    nn::Tensor grad = mse.backward(pred);
    EXPECT_FLOAT_EQ(grad.at(0, 0), 0.0F);
}

// Default: NO clipping — backward() returns the exact MSE gradient and matches
// torch.nn.functional.mse_loss. This used to clip unconditionally at norm 1.0, which made
// MSELoss silently not the MSE gradient and overrode TrainerConfig::grad_clip_norm=0 ("no
// clipping") one layer down. Since MSELossImpl is Trainer's default LossType, that quietly
// affected every trained autoencoder. Caught by micro_network_parity_gtest.
TEST(MSELossTest, GradientIsExactByDefault)
{
    MSELoss mse;
    nn::Tensor pred(1, 1);
    pred.at(0, 0) = 100.0F;
    nn::Tensor target(1, 1);
    target.at(0, 0) = 0.0F;

    mse.set_target(target);
    (void) mse.forward(pred, true);
    nn::Tensor grad = mse.backward(pred);
    // d/dpred mean((pred-target)^2) = 2*(pred-target)/n = 2*100/1 = 200 — NOT clipped to 1.
    EXPECT_NEAR(grad.at(0, 0), 200.0F, 1e-3F);
}

// The clip remains available as an explicit escape hatch; prefer TrainerConfig::grad_clip_norm.
TEST(MSELossTest, GradientIsClippedWhenExplicitlyEnabled)
{
    MSELoss mse;
    mse.max_gradient_norm = 1.0F; // opt in
    nn::Tensor pred(1, 1);
    pred.at(0, 0) = 100.0F;
    nn::Tensor target(1, 1);
    target.at(0, 0) = 0.0F;

    mse.set_target(target);
    (void) mse.forward(pred, true);
    nn::Tensor grad = mse.backward(pred);
    EXPECT_NEAR(std::fabs(grad.at(0, 0)), 1.0F, 1e-5F);
}

TEST(MAELossTest, TrainToggleAndForwardBackward)
{
    MAELoss mae;
    nn::Tensor pred(2, 1);
    pred.at(0, 0) = 1.0F;
    pred.at(1, 0) = -1.0F;
    nn::Tensor target(2, 1);
    target.at(0, 0) = 0.0F;
    target.at(1, 0) = 0.0F;

    EXPECT_THROW(mae.forward(pred), std::runtime_error);

    mae.set_target(target);
    mae.train(false);
    nn::Tensor loss_eval = mae.forward(pred, true);
    EXPECT_NEAR(loss_eval.at(0, 0), 1.0F, 1e-5F);

    mae.train(true);
    nn::Tensor loss_train = mae.forward(pred, true);
    EXPECT_NEAR(loss_train.at(0, 0), 1.0F, 1e-5F);

    nn::Tensor grad = mae.backward(pred);
    EXPECT_NEAR(grad.at(0, 0), 0.5F, 1e-6F);
    EXPECT_NEAR(grad.at(1, 0), -0.5F, 1e-6F);
}

TEST(MAELossTest, NonFiniteBackwardReturnsZeroGradient)
{
    MAELoss mae;
    nn::Tensor pred(1, 1);
    pred.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    nn::Tensor target(1, 1);
    target.at(0, 0) = 0.0F;

    mae.set_target(target);
    (void) mae.forward(pred, true);
    nn::Tensor grad = mae.backward(pred);
    EXPECT_FLOAT_EQ(grad.at(0, 0), 0.0F);
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

// SpikeCountLoss.train() — covers SpikeCountLoss.hpp lines 54, 56, 57
TEST(SpikeCountLossTest, TrainModeToggle)
{
    SpikeCountLoss loss;
    nn::Tensor pred(2, 1);
    pred.at(0, 0) = 5.0F;
    pred.at(1, 0) = 3.0F;
    nn::Tensor target(2, 1);
    target.at(0, 0) = 5.0F;
    target.at(1, 0) = 3.0F;
    loss.set_target(target);

    // Call train(false) — disables gradient caching
    loss.train(false);
    nn::Tensor out1 = loss.forward(pred, true);
    EXPECT_NEAR(out1.at(0, 0), 0.0F, 1e-5F);

    // Call train(true) — re-enables gradient caching
    loss.train(true);
    nn::Tensor out2 = loss.forward(pred, true);
    EXPECT_NEAR(out2.at(0, 0), 0.0F, 1e-5F);
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
