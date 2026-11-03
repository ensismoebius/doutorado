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
#include <span>
#include <stdexcept>
#include <vector>

#include "../EpochResult.hpp"
#include "../TrainerConfig.hpp"
#include "Backend.hpp"
#include "layers/losses/SpikeCountLoss.hpp"
#include "tensor/Tensor.hpp"
#include "training/BestWeightsCallback.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ITrainingCallback.hpp"
#include "utility/GradClip.hpp"

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

namespace
{

// Feeds one monitored loss per epoch; returns the epoch after which should_stop() first
// became true, 0 if it never did.
int first_stop_epoch(nn::training::EarlyStoppingCallback& cb, const std::vector<float>& losses)
{
    nn::training::TrainingState state;
    for (std::size_t i = 0; i < losses.size(); ++i)
    {
        state.epoch = static_cast<int>(i) + 1;
        cb.on_epoch_end(state, nn::training::EpochResult{state.epoch, losses[i], losses[i], 1.0F});
        if (cb.should_stop()) return state.epoch;
    }
    return 0;
}

} // namespace

// The default (+inf) arms on epoch 1: the plain patience rule, unchanged for every caller
// that does not pass arm_below.
TEST(EarlyStoppingCallback, DefaultStopsAfterPatienceEpochsWithoutANewBest)
{
    nn::training::EarlyStoppingCallback cb(5);
    EXPECT_EQ(first_stop_epoch(cb, {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.5F}), 6);
    EXPECT_EQ(cb.armed_epoch(), 1);
}

// A run on the mean-predictor plateau is not stopped there: the 12 flat epochs count for
// nothing, patience starts at the epoch that crosses arm_below (13), and the run stops 5
// epochs after its last new best (14).
TEST(EarlyStoppingCallback, PlateauEpochsDoNotCountUntilArmed)
{
    nn::training::EarlyStoppingCallback cb(5, 1e-8F, 0.9F);
    std::vector<float> losses(12, 1.0F);
    for (const float v : {0.8F, 0.7F, 0.75F, 0.75F, 0.75F, 0.75F, 0.75F}) losses.push_back(v);
    EXPECT_EQ(first_stop_epoch(cb, losses), 19);
    EXPECT_EQ(cb.armed_epoch(), 13);
}

TEST(EarlyStoppingCallback, ARunThatNeverArmsTrainsEveryEpoch)
{
    nn::training::EarlyStoppingCallback cb(5, 1e-8F, 0.9F);
    EXPECT_EQ(first_stop_epoch(cb, std::vector<float>(30, 0.95F)), 0);
    EXPECT_EQ(cb.armed_epoch(), 0);
}

namespace
{

struct TwoTensorModel
{
    nn::Tensor w = nn::Tensor(2, 2);
    nn::Tensor b = nn::Tensor(1, 2);
    std::vector<nn::Tensor*> ptrs{&w, &b};
    auto params() -> std::span<nn::Tensor*>
    {
        return ptrs;
    }
    void fill(float v)
    {
        for (nn::Index k = 0; k < w.size(); ++k) w.at(k) = v;
        for (nn::Index k = 0; k < b.size(); ++k) b.at(k) = -v;
    }
};

} // namespace

// The network handed back is the best validation epoch's (2), not the last one (3).
TEST(BestWeightsCallback, RestoresTheEpochWithTheLowestValidationLoss)
{
    TwoTensorModel model;
    nn::training::BestWeightsCallback<TwoTensorModel> cb(model);
    nn::training::TrainingState state;
    const std::vector<std::pair<float, float>> epochs{{1.0F, 0.5F}, {2.0F, 0.3F}, {3.0F, 0.4F}};
    for (std::size_t i = 0; i < epochs.size(); ++i)
    {
        model.fill(epochs[i].first); // what the optimizer left after this epoch
        state.epoch = static_cast<int>(i) + 1;
        cb.on_epoch_end(
            state, nn::training::EpochResult{state.epoch, 9.0F, epochs[i].second, 1.0F});
    }
    cb.on_train_end({});
    EXPECT_EQ(cb.best_epoch(), 2);
    for (nn::Index k = 0; k < model.w.size(); ++k) EXPECT_FLOAT_EQ(model.w.at(k), 2.0F);
    for (nn::Index k = 0; k < model.b.size(); ++k) EXPECT_FLOAT_EQ(model.b.at(k), -2.0F);
}

TEST(BestWeightsCallback, RefusesARunWithoutValidationLoss)
{
    TwoTensorModel model;
    nn::training::BestWeightsCallback<TwoTensorModel> cb(model);
    nn::training::TrainingState state;
    state.epoch = 1;
    EXPECT_THROW(
        cb.on_epoch_end(state,
            nn::training::EpochResult{1, 1.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F}),
        std::logic_error);
    nn::training::BestWeightsCallback<TwoTensorModel> untouched(model);
    EXPECT_THROW(untouched.on_train_end({}), std::logic_error); // no epoch ever ended
}
