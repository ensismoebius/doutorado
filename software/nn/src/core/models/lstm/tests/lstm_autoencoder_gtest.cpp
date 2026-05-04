/**
 * @file lstm_autoencoder_gtest.cpp
 * @brief Tests for the core nn::models::lstm::LSTMAutoencoder and LSTMLayer.
 */

#include <gtest/gtest.h>

#include "nn/layers/lstm/LSTMLayer.hpp"
#include "nn/models/lstm/LSTMAutoencoder.hpp"
#include "nn/tensor/Tensor.hpp"

namespace
{

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

TEST(LSTMAutoencoderConfigTest, DefaultValues)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;

    EXPECT_EQ(cfg.input_size, 64);
    EXPECT_EQ(cfg.hidden_size, 128);
    EXPECT_EQ(cfg.latent_size, 16);
    EXPECT_EQ(cfg.num_layers, 1);
    EXPECT_EQ(cfg.seq_len, 32);
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

// ---------------------------------------------------------------------------
// LSTMLayer
// ---------------------------------------------------------------------------

TEST(CoreLSTMLayerTest, ForwardShape)
{
    nn::models::lstm::LSTMLayer layer(8, 16);
    nn::Tensor input = nn::Tensor::rand(4, 8);
    nn::Tensor output = layer.forward(input, false);
    EXPECT_EQ(output.rows(), 4u);
    EXPECT_EQ(output.cols(), 16u);
}

TEST(CoreLSTMLayerTest, BackwardShape)
{
    nn::models::lstm::LSTMLayer layer(8, 16);
    nn::Tensor input = nn::Tensor::rand(4, 8);
    layer.forward(input, true);
    nn::Tensor grad_h = nn::Tensor::ones(4, 16);
    nn::Tensor grad_input = layer.backward(grad_h);
    EXPECT_EQ(grad_input.rows(), 4u);
    EXPECT_EQ(grad_input.cols(), 8u);
}

TEST(CoreLSTMLayerTest, ResetStateClearsCache)
{
    nn::models::lstm::LSTMLayer layer(8, 16);
    nn::Tensor input = nn::Tensor::rand(4, 8);
    layer.forward(input, true);
    EXPECT_EQ(layer.cache_.size(), 4u);
    layer.reset_state();
    EXPECT_EQ(layer.cache_.size(), 0u);
}

// ---------------------------------------------------------------------------
// LSTMAutoencoder
// ---------------------------------------------------------------------------

TEST(CoreLSTMAutoencoderTest, ForwardOutputShape)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;
    cfg.input_size = 8;
    cfg.seq_len = 6;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.num_layers = 1;

    nn::models::lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(6, 8);
    model.reset_state();
    nn::Tensor recon = model.forward(input, false);
    EXPECT_EQ(recon.rows(), 6u);
    EXPECT_EQ(recon.cols(), 8u);
}

TEST(CoreLSTMAutoencoderTest, BackwardShapeMatchesInput)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;
    cfg.input_size = 8;
    cfg.seq_len = 6;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.num_layers = 1;

    nn::models::lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(6, 8);
    model.reset_state();
    nn::Tensor recon = model.forward(input, true);
    nn::Tensor grad = nn::Tensor::ones(recon.rows(), recon.cols());
    nn::Tensor grad_in = model.backward(grad);
    EXPECT_EQ(grad_in.rows(), input.rows());
    EXPECT_EQ(grad_in.cols(), input.cols());
}

TEST(CoreLSTMAutoencoderTest, ParamCountExactSingleLayer)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;
    cfg.input_size = 4;
    cfg.seq_len = 4;
    cfg.hidden_size = 8;
    cfg.latent_size = 2;
    cfg.num_layers = 1;

    nn::models::lstm::LSTMAutoencoder model(cfg);
    // Single-layer topology has:
    //   enc_lstm (W,U,b)=3
    //   enc_proj (W,b)=2
    //   dec_expand (W,b)=2
    //   dec_lstm (W,U,b)=3
    //   out_proj (W,b)=2
    // total = 12 trainable tensors.
    EXPECT_EQ(model.params().size(), 12u);
}

TEST(CoreLSTMAutoencoderTest, StateDictRoundtrip)
{
    nn::models::lstm::LSTMAutoencoderConfig cfg;
    cfg.input_size = 4;
    cfg.seq_len = 4;
    cfg.hidden_size = 8;
    cfg.latent_size = 2;
    cfg.num_layers = 1;

    nn::models::lstm::LSTMAutoencoder model(cfg);
    const auto sd = model.state_dict();
    EXPECT_FALSE(sd.empty());

    nn::models::lstm::LSTMAutoencoder model2(cfg);
    model2.load_state_dict(sd);
    const auto sd2 = model2.state_dict();
    EXPECT_EQ(sd.size(), sd2.size());
}

} // namespace