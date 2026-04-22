/**
 * @file autoencoder_config_gtest.cpp
 * @brief Tests for AutoencoderConfig structure.
 */

#include <gtest/gtest.h>

#include "autoencoder/Config.hpp"

namespace
{

TEST(AutoencoderConfigTest, DefaultValues)
{
    nn::models::autoencoder::AutoencoderConfig cfg;

    EXPECT_EQ(cfg.loss_type, "mse");
    EXPECT_EQ(cfg.input_features, 128);
    EXPECT_EQ(cfg.hidden_size, 64);
    EXPECT_EQ(cfg.latent_size, 32);
    EXPECT_EQ(cfg.depth, 1);
    EXPECT_TRUE(cfg.layer_sizes.empty());
    EXPECT_EQ(cfg.architecture, nn::models::autoencoder::AutoencoderArchitecture::Auto);
    EXPECT_EQ(cfg.branch_hidden_size, 0);
    EXPECT_EQ(cfg.fusion_hidden_size, 0);
    EXPECT_EQ(cfg.residual_blocks, 1);
    EXPECT_EQ(cfg.eeg_features, 0);
    EXPECT_EQ(cfg.audio_features, 0);
    EXPECT_FLOAT_EQ(cfg.time_step, 1.0F);
    EXPECT_FLOAT_EQ(cfg.resistance, 1.0F);
    EXPECT_FLOAT_EQ(cfg.capacitance, 1.0F);
    EXPECT_FALSE(cfg.initializer_seed.has_value());
}

TEST(AutoencoderConfigTest, SetHiddenSize)
{
    nn::models::autoencoder::AutoencoderConfig cfg;
    cfg.hidden_size = 256;
    EXPECT_EQ(cfg.hidden_size, 256);
}

TEST(AutoencoderConfigTest, SetLatentSize)
{
    nn::models::autoencoder::AutoencoderConfig cfg;
    cfg.latent_size = 16;
    EXPECT_EQ(cfg.latent_size, 16);
}

TEST(AutoencoderConfigTest, SetLayerSizes)
{
    nn::models::autoencoder::AutoencoderConfig cfg;
    cfg.layer_sizes = {128, 64, 32};

    ASSERT_EQ(cfg.layer_sizes.size(), 3);
    EXPECT_EQ(cfg.layer_sizes[0], 128);
    EXPECT_EQ(cfg.layer_sizes[1], 64);
    EXPECT_EQ(cfg.layer_sizes[2], 32);
}

TEST(AutoencoderConfigTest, SetMultimodalFeatures)
{
    nn::models::autoencoder::AutoencoderConfig cfg;
    cfg.eeg_features = 64;
    cfg.audio_features = 128;

    EXPECT_EQ(cfg.eeg_features, 64);
    EXPECT_EQ(cfg.audio_features, 128);
}

TEST(AutoencoderConfigTest, SetSnnParameters)
{
    nn::models::autoencoder::AutoencoderConfig cfg;
    cfg.time_step = 0.001F;
    cfg.resistance = 10.0F;
    cfg.capacitance = 0.1F;

    EXPECT_FLOAT_EQ(cfg.time_step, 0.001F);
    EXPECT_FLOAT_EQ(cfg.resistance, 10.0F);
    EXPECT_FLOAT_EQ(cfg.capacitance, 0.1F);
}

TEST(AutoencoderConfigTest, SetArchitecture)
{
    nn::models::autoencoder::AutoencoderConfig cfg;
    cfg.architecture = nn::models::autoencoder::AutoencoderArchitecture::ResidualDense;
    EXPECT_EQ(cfg.architecture, nn::models::autoencoder::AutoencoderArchitecture::ResidualDense);
}

TEST(AutoencoderConfigTest, SetInitializerSeed)
{
    nn::models::autoencoder::AutoencoderConfig cfg;
    cfg.initializer_seed = 42U;

    ASSERT_TRUE(cfg.initializer_seed.has_value());
    EXPECT_EQ(cfg.initializer_seed.value(), 42U);
}

} // namespace