/**
 * @file trainer_gtest.cpp
 * @brief Tests for TrainerConfig and EpochResult.
 *
 * Note: The Trainer template has pre-existing bugs in nested lambdas that access
 * member variables incorrectly. This test only covers Config and EpochResult.
 */

#include <gtest/gtest.h>
#include <limits>

#include "Trainer.hpp"

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

} // namespace