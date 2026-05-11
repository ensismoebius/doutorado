/**
 * @file trainer_gtest.cpp
 * @brief Tests for TrainerConfig and EpochResult.
 *
 * Note: The Trainer template has pre-existing bugs in nested lambdas that access
 * member variables incorrectly. This test only covers Config and EpochResult.
 */

#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "../EpochResult.hpp"
#include "../TrainerConfig.hpp"
#include "nn/Backend.hpp"
#include "nn/layers/losses/SpikeCountLoss.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/training/ITrainingCallback.hpp"
#include "nn/utility/GradClip.hpp"

namespace
{

TEST(TrainerConfigTest, DefaultValues)
{
    nn::training::TrainerConfig cfg;

    EXPECT_EQ(cfg.epochs, 10);
    EXPECT_FLOAT_EQ(cfg.learning_rate, 0.001F);
    EXPECT_FLOAT_EQ(cfg.adam_beta1, 0.9F);
    EXPECT_FLOAT_EQ(cfg.adam_beta2, 0.999F);
    EXPECT_FLOAT_EQ(cfg.adam_epsilon, 1e-8F);
    EXPECT_FLOAT_EQ(cfg.grad_clip_norm, 0.0F);
    EXPECT_EQ(cfg.batch_size, 1);
    EXPECT_EQ(cfg.sampler_shuffle_seed, 42U);
}

TEST(TrainerConfigTest, SetEpochs)
{
    nn::training::TrainerConfig cfg;
    cfg.epochs = 100;
    EXPECT_EQ(cfg.epochs, 100);
}

TEST(TrainerConfigTest, SetLearningRate)
{
    nn::training::TrainerConfig cfg;
    cfg.learning_rate = 0.01F;
    EXPECT_FLOAT_EQ(cfg.learning_rate, 0.01F);
}

TEST(TrainerConfigTest, SetBatchSize)
{
    nn::training::TrainerConfig cfg;
    cfg.batch_size = 32;
    EXPECT_EQ(cfg.batch_size, 32);
}

TEST(TrainerConfigTest, SetGradClipNorm)
{
    nn::training::TrainerConfig cfg;
    cfg.grad_clip_norm = 1.0F;
    EXPECT_FLOAT_EQ(cfg.grad_clip_norm, 1.0F);
}

TEST(TrainerConfigTest, SetAdamParameters)
{
    nn::training::TrainerConfig cfg;
    cfg.adam_beta1 = 0.85F;
    cfg.adam_beta2 = 0.995F;
    cfg.adam_epsilon = 1e-10F;

    EXPECT_FLOAT_EQ(cfg.adam_beta1, 0.85F);
    EXPECT_FLOAT_EQ(cfg.adam_beta2, 0.995F);
    EXPECT_FLOAT_EQ(cfg.adam_epsilon, 1e-10F);
}

TEST(EpochResultTest, DefaultValues)
{
    nn::training::EpochResult result;

    EXPECT_EQ(result.epoch, 0);
    EXPECT_FLOAT_EQ(result.train_loss, 0.0F);
    EXPECT_FLOAT_EQ(result.val_loss, 0.0F);
    EXPECT_FLOAT_EQ(result.epoch_ms, 0.0F);
}

TEST(EpochResultTest, Structure)
{
    nn::training::EpochResult result;
    result.epoch = 5;
    result.train_loss = 0.5F;
    result.val_loss = 0.6F;
    result.epoch_ms = 1000.0F;

    EXPECT_EQ(result.epoch, 5);
    EXPECT_FLOAT_EQ(result.train_loss, 0.5F);
    EXPECT_FLOAT_EQ(result.val_loss, 0.6F);
    EXPECT_FLOAT_EQ(result.epoch_ms, 1000.0F);
}

TEST(GradClipTest, ScalesGradientsWhenGlobalNormExceedsThreshold)
{
    nn::Tensor p1(1, 1);
    nn::Tensor p2(1, 1);
    nn::Tensor g1(1, 1);
    nn::Tensor g2(1, 1);

    g1.at(0, 0) = 6.0F;
    g2.at(0, 0) = 8.0F;
    p1.set_grad(g1);
    p2.set_grad(g2);

    std::array<nn::Tensor*, 2> params = {&p1, &p2};
    nn::utils::clip_grad_norm<nn::Tensor>(std::span<nn::Tensor*>(params), 5.0F);

    EXPECT_FLOAT_EQ(p1.grad().at(0, 0), 3.0F);
    EXPECT_FLOAT_EQ(p2.grad().at(0, 0), 4.0F);
}

TEST(GradClipTest, LeavesGradientsUnchangedWithinThreshold)
{
    nn::Tensor p1(1, 1);
    nn::Tensor p2(1, 1);
    nn::Tensor g1(1, 1);
    nn::Tensor g2(1, 1);

    g1.at(0, 0) = 3.0F;
    g2.at(0, 0) = 4.0F;
    p1.set_grad(g1);
    p2.set_grad(g2);

    std::array<nn::Tensor*, 2> params = {&p1, &p2};
    nn::utils::clip_grad_norm<nn::Tensor>(std::span<nn::Tensor*>(params), 5.0F);

    EXPECT_FLOAT_EQ(p1.grad().at(0, 0), 3.0F);
    EXPECT_FLOAT_EQ(p2.grad().at(0, 0), 4.0F);
}

TEST(SpikeCountLossTest, ForwardBackwardWithoutRateRegularization)
{
    using SpikeLoss = SpikeCountLossImpl<nn::Backend>;

    SpikeLoss loss;
    loss.rate_reg_lambda = 0.0F;

    nn::Tensor pred(2, 1);
    pred.at(0, 0) = 10.0F;
    pred.at(1, 0) = 20.0F;

    nn::Tensor target(2, 1);
    target.at(0, 0) = 8.0F;
    target.at(1, 0) = 22.0F;
    loss.set_target(target);

    nn::Tensor out = loss.forward(pred, true);
    EXPECT_NEAR(out.at(0, 0), 4.0F, 1e-6F);

    nn::Tensor grad_out(1, 1);
    grad_out.at(0, 0) = 1.0F;
    nn::Tensor grad = loss.backward(grad_out);
    EXPECT_NEAR(grad.at(0, 0), 2.0F, 1e-6F);
    EXPECT_NEAR(grad.at(1, 0), -2.0F, 1e-6F);
}

TEST(SpikeCountLossTest, AppliesRateRegularizationAndGradientOffset)
{
    using SpikeLoss = SpikeCountLossImpl<nn::Backend>;

    SpikeLoss loss;
    loss.min_rate = 0.05F;
    loss.max_rate = 0.80F;
    loss.rate_reg_lambda = 2.0F;

    nn::Tensor pred(2, 1);
    pred.at(0, 0) = 0.0F;
    pred.at(1, 0) = 0.0F;

    nn::Tensor target(2, 1);
    target.at(0, 0) = 0.0F;
    target.at(1, 0) = 0.0F;
    loss.set_target(target);

    nn::Tensor out = loss.forward(pred, true);
    EXPECT_NEAR(out.at(0, 0), 0.005F, 1e-6F);
    EXPECT_NEAR(loss.last_mean_rate(), 0.0F, 1e-6F);

    nn::Tensor grad_out(1, 1);
    grad_out.at(0, 0) = 1.0F;
    nn::Tensor grad = loss.backward(grad_out);
    EXPECT_NEAR(grad.at(0, 0), -0.1F, 1e-6F);
    EXPECT_NEAR(grad.at(1, 0), -0.1F, 1e-6F);
}

} // namespace

// ITrainingCallback default implementations test (covers ITrainingCallback.hpp line 26)
TEST(ITrainingCallbackTest, DefaultImplementationsAreSafe)
{
    nn::training::ITrainingCallback cb;
    nn::training::TrainingState state;
    nn::training::EpochResult result;

    cb.on_train_begin(10);
    cb.on_epoch_begin(state);
    cb.on_epoch_end(state, result);
    cb.on_batch_begin(state);
    cb.on_batch_progress(state);
    cb.on_batch_end(state);
    cb.on_train_end({});
    EXPECT_FALSE(cb.should_stop());
}
