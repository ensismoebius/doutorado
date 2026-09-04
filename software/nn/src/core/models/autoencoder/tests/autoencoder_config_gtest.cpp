/**
 * @file autoencoder_config_gtest.cpp
 * @brief Tests for AutoencoderConfig structure.
 *
 * This file used to test a parallel AutoencoderConfig that no translation
 * unit ever included. It now tests the one three experiments actually
 * build against -- which, until this change, had no direct tests at all.
 */

#include <gtest/gtest.h>

#include "models/autoencoder/AutoencoderConfig.hpp"

namespace
{

TEST(AutoencoderConfigTest, DefaultValues)
{
    AutoencoderConfig cfg;

    EXPECT_EQ(cfg.loss_type, "mse");
    EXPECT_EQ(cfg.input_features, 128);
    EXPECT_EQ(cfg.hidden_size, 64);
    EXPECT_EQ(cfg.latent_size, 32);
    EXPECT_EQ(cfg.depth, 1);
    EXPECT_TRUE(cfg.layer_sizes.empty());
    EXPECT_EQ(cfg.architecture, AutoencoderArchitecture::Auto);
    EXPECT_EQ(cfg.branch_hidden_size, 0);
    EXPECT_EQ(cfg.fusion_hidden_size, 0);
    EXPECT_EQ(cfg.residual_blocks, 1);
    EXPECT_EQ(cfg.eeg_features, 0);
    EXPECT_EQ(cfg.audio_features, 0);
    EXPECT_FLOAT_EQ(cfg.delta_t, 1.0F);
    EXPECT_FLOAT_EQ(cfg.resistance, 1.0F);
    EXPECT_FLOAT_EQ(cfg.capacitance, 1.0F);
    EXPECT_FALSE(cfg.initializer_seed.has_value());
}

TEST(AutoencoderConfigTest, SetHiddenSize)
{
    AutoencoderConfig cfg;
    cfg.hidden_size = 256;
    EXPECT_EQ(cfg.hidden_size, 256);
}

TEST(AutoencoderConfigTest, SetLatentSize)
{
    AutoencoderConfig cfg;
    cfg.latent_size = 16;
    EXPECT_EQ(cfg.latent_size, 16);
}

TEST(AutoencoderConfigTest, SetLayerSizes)
{
    AutoencoderConfig cfg;
    cfg.layer_sizes = {128, 64, 32};

    ASSERT_EQ(cfg.layer_sizes.size(), 3);
    EXPECT_EQ(cfg.layer_sizes[0], 128);
    EXPECT_EQ(cfg.layer_sizes[1], 64);
    EXPECT_EQ(cfg.layer_sizes[2], 32);
}

TEST(AutoencoderConfigTest, SetMultimodalFeatures)
{
    AutoencoderConfig cfg;
    cfg.eeg_features = 64;
    cfg.audio_features = 128;

    EXPECT_EQ(cfg.eeg_features, 64);
    EXPECT_EQ(cfg.audio_features, 128);
}

TEST(AutoencoderConfigTest, SetSnnParameters)
{
    AutoencoderConfig cfg;
    cfg.delta_t = 0.001F;
    cfg.resistance = 10.0F;
    cfg.capacitance = 0.1F;

    EXPECT_FLOAT_EQ(cfg.delta_t, 0.001F);
    EXPECT_FLOAT_EQ(cfg.resistance, 10.0F);
    EXPECT_FLOAT_EQ(cfg.capacitance, 0.1F);
}

TEST(AutoencoderConfigTest, SetArchitecture)
{
    AutoencoderConfig cfg;
    cfg.architecture = AutoencoderArchitecture::ResidualDense;
    EXPECT_EQ(cfg.architecture, AutoencoderArchitecture::ResidualDense);
}

TEST(AutoencoderConfigTest, SetInitializerSeed)
{
    AutoencoderConfig cfg;
    cfg.initializer_seed = 42U;

    ASSERT_TRUE(cfg.initializer_seed.has_value());
    EXPECT_EQ(cfg.initializer_seed.value(), 42U);
}

} // namespace
namespace
{

/// The SNN half of the config, which the previous (dead) version of this
/// file did not have and therefore never checked.
TEST(AutoencoderConfigTest, SpikingDefaults)
{
    AutoencoderConfig cfg;

    // 0 means UNSET, and the SNN builders raise on it. It is deliberately
    // NOT 1: a single unrolled step trains and reports a loss while giving
    // no temporal credit assignment at all, so the wrong default here is a
    // silent downgrade rather than a crash.
    EXPECT_EQ(cfg.time_steps, 0);

    // `delta_t` is the step SIZE, `time_steps` the step COUNT. One letter
    // apart in an earlier naming, and the source of enough confusion to be
    // worth pinning in a test.
    EXPECT_FLOAT_EQ(cfg.delta_t, 1.0F);

    EXPECT_FLOAT_EQ(cfg.voltage_threshold, 1.0F);
    EXPECT_FLOAT_EQ(cfg.firing_rate_reg_lambda, 0.0F); // 0 = regularization off
    EXPECT_FLOAT_EQ(cfg.firing_rate_min, 0.05F);
    EXPECT_FLOAT_EQ(cfg.firing_rate_max, 0.80F);
}

} // namespace
