/**
 * @file lstm_autoencoder_gtest.cpp
 * @brief Tests for LSTMAutoencoderConfig.
 *
 * Note: The LSTMAutoencoder class implementation is in experiments, not core.
 * This test covers only the config structure.
 */

#include <gtest/gtest.h>

#include "lstm/LSTMAutoencoder.hpp"

namespace
{

TEST(LSTMAutoencoderConfigTest, DefaultValues)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;

    EXPECT_EQ(cfg.input_size, 128);
    EXPECT_EQ(cfg.hidden_size, 64);
    EXPECT_EQ(cfg.latent_size, 32);
    EXPECT_EQ(cfg.num_layers, 1);
    EXPECT_FLOAT_EQ(cfg.dropout, 0.0F);
    EXPECT_FALSE(cfg.bidirectional);
}

TEST(LSTMAutoencoderConfigTest, SetDimensions)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;
    cfg.input_size = 256;
    cfg.hidden_size = 128;
    cfg.latent_size = 64;
    cfg.num_layers = 2;

    EXPECT_EQ(cfg.input_size, 256);
    EXPECT_EQ(cfg.hidden_size, 128);
    EXPECT_EQ(cfg.latent_size, 64);
    EXPECT_EQ(cfg.num_layers, 2);
}

TEST(LSTMAutoencoderConfigTest, SetBidirectional)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;
    cfg.bidirectional = true;

    EXPECT_TRUE(cfg.bidirectional);
}

TEST(LSTMAutoencoderConfigTest, SetDropout)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;
    cfg.dropout = 0.5F;

    EXPECT_FLOAT_EQ(cfg.dropout, 0.5F);
}

} // namespace