/**
 * @file LSTMAutoencoder_gtest.cpp
 * @brief Unit tests for integrated Experiment04 LSTM autoencoder components.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../../core/training/Trainer.hpp"
#include "nn/layers/lstm/LSTMLayer.hpp"
#include "nn/models/lstm/LSTMAutoencoder.hpp"
#include "nn/tensor/Tensor.hpp"

namespace
{
namespace lstm = nn::models::lstm;
using nn::training::TrainerConfig;

TEST(LSTMLayerTest, ForwardOutputShape)
{
    lstm::LSTMLayer lstm(4, 8);
    nn::Tensor input = nn::Tensor::rand(5, 4);
    nn::Tensor out = lstm.forward(input, false);

    EXPECT_EQ(out.rows(), 5u);
    EXPECT_EQ(out.cols(), 8u);
}

TEST(LSTMLayerTest, BackwardInputGradShape)
{
    lstm::LSTMLayer lstm(4, 8);
    nn::Tensor input = nn::Tensor::rand(5, 4);
    lstm.forward(input, true);
    nn::Tensor grad_h = nn::Tensor::ones(5, 8);
    nn::Tensor grad_in = lstm.backward(grad_h);

    EXPECT_EQ(grad_in.rows(), 5u);
    EXPECT_EQ(grad_in.cols(), 4u);
}

TEST(LSTMLayerTest, WeightGradientsNonZeroAfterBackward)
{
    lstm::LSTMLayer lstm(3, 6);
    nn::Tensor input = nn::Tensor::rand(4, 3);
    lstm.forward(input, true);
    lstm.backward(nn::Tensor::ones(4, 6));

    EXPECT_GT(lstm.W_.grad().norm(), 0.0f);
    EXPECT_GT(lstm.U_.grad().norm(), 0.0f);
    EXPECT_GT(lstm.b_.grad().norm(), 0.0f);
}

TEST(LSTMLayerTest, ResetStateClearsCache)
{
    lstm::LSTMLayer lstm(4, 8);
    nn::Tensor input = nn::Tensor::rand(5, 4);
    lstm.forward(input, true);
    EXPECT_EQ(lstm.cache_.size(), 5u);
    lstm.reset_state();
    EXPECT_EQ(lstm.cache_.size(), 0u);
}

TEST(LSTMLayerTest, ParamsReturnsThreePointers)
{
    lstm::LSTMLayer lstm(4, 8);
    auto p = lstm.params();
    EXPECT_EQ(p.size(), 3u);
    for (nn::Tensor* ptr : p) EXPECT_NE(ptr, nullptr);
}

TEST(LSTMLayerTest, StateDictRoundTrip)
{
    lstm::LSTMLayer lstm(4, 8);
    lstm.W_.at(0, 0) = 99.0f;

    auto sd = lstm.state_dict();
    ASSERT_EQ(sd.count("W"), 1u);

    lstm::LSTMLayer lstm2(4, 8);
    lstm2.load_state_dict(sd);
    EXPECT_FLOAT_EQ(lstm2.W_.at(0, 0), 99.0f);
}

TEST(LSTMLayerTest, BackwardCalledBeforeForwardThrows)
{
    lstm::LSTMLayer lstm(4, 8);
    nn::Tensor grad = nn::Tensor::ones(5, 8);
    EXPECT_THROW(lstm.backward(grad), std::runtime_error);
}

auto small_cfg() -> lstm::LSTMAutoencoderConfig
{
    lstm::LSTMAutoencoderConfig cfg;
    cfg.input_size = 8;
    cfg.seq_len = 6;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.num_layers = 1;
    return cfg;
}

TEST(LSTMAutoencoderTest, EncodeOutputShape)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    nn::Tensor z = model.encode(input, false);
    EXPECT_EQ(z.rows(), 1u);
    EXPECT_EQ(z.cols(), static_cast<nn::Index>(cfg.latent_size));
}

TEST(LSTMAutoencoderTest, DecodeOutputShape)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    nn::Tensor z = nn::Tensor::rand(1, cfg.latent_size);
    nn::Tensor recon = model.decode(z, cfg.seq_len, false);
    EXPECT_EQ(recon.rows(), static_cast<nn::Index>(cfg.seq_len));
    EXPECT_EQ(recon.cols(), static_cast<nn::Index>(cfg.input_size));
}

TEST(LSTMAutoencoderTest, ForwardOutputShape)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    nn::Tensor recon = model.forward(input, false);
    EXPECT_EQ(recon.rows(), input.rows());
    EXPECT_EQ(recon.cols(), input.cols());
}

TEST(LSTMAutoencoderTest, BackwardInputGradShape)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    model.forward(input, true);
    nn::Tensor grad_recon = nn::Tensor::ones(cfg.seq_len, cfg.input_size);
    nn::Tensor grad_in = model.backward(grad_recon);
    EXPECT_EQ(grad_in.rows(), static_cast<nn::Index>(cfg.seq_len));
    EXPECT_EQ(grad_in.cols(), static_cast<nn::Index>(cfg.input_size));
}

TEST(LSTMAutoencoderTest, ParamsNonEmpty)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    EXPECT_GT(model.params().size(), 0u);
}

TEST(LSTMAutoencoderTest, ResetStateClearsLSTMCaches)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    model.forward(input, true);
    model.reset_state();
    for (const auto& lstm : model.enc_lstms_) EXPECT_EQ(lstm->cache_.size(), 0u);
    for (const auto& lstm : model.dec_lstms_) EXPECT_EQ(lstm->cache_.size(), 0u);
}

TEST(LSTMAutoencoderTest, LatentBoundedByTanh)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    model.forward(input, false);
    const nn::Tensor& z = model.latent_cache_;

    for (nn::Index i = 0; i < z.size(); ++i)
    {
        EXPECT_GT(z.at(i), -1.0f);
        EXPECT_LT(z.at(i), 1.0f);
    }
}

TEST(LSTMAutoencoderTest, StateDictRoundTrip)
{
    const auto cfg = small_cfg();
    lstm::LSTMAutoencoder model(cfg);
    model.out_proj_->weight.at(0, 0) = 77.0f;

    auto sd = model.state_dict();
    EXPECT_FALSE(sd.empty());

    lstm::LSTMAutoencoder model2(cfg);
    model2.load_state_dict(sd);
    EXPECT_FLOAT_EQ(model2.out_proj_->weight.at(0, 0), 77.0f);
}

TEST(LSTMAutoencoderTest, MultiLayerForwardShape)
{
    auto cfg = small_cfg();
    cfg.num_layers = 2;
    lstm::LSTMAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    nn::Tensor recon = model.forward(input, false);
    EXPECT_EQ(recon.rows(), input.rows());
    EXPECT_EQ(recon.cols(), input.cols());
}

TEST(TrainerTest, LossDecreasesOnSyntheticData)
{
    auto arch_cfg = small_cfg();
    arch_cfg.input_size = 4;
    arch_cfg.seq_len = 5;
    arch_cfg.hidden_size = 8;
    arch_cfg.latent_size = 2;

    TrainerConfig train_cfg;
    train_cfg.epochs = 20;
    train_cfg.learning_rate = 1e-3f;
    train_cfg.grad_clip_norm = 1.0f;
    train_cfg.sampler_shuffle_seed = 7u;
    train_cfg.batch_size = 1;

    nn::Tensor sample(arch_cfg.seq_len, arch_cfg.input_size);
    for (nn::Index i = 0; i < static_cast<nn::Index>(arch_cfg.seq_len); ++i)
        for (nn::Index j = 0; j < static_cast<nn::Index>(arch_cfg.input_size); ++j)
            sample.at(i, j) = 0.5f;

    std::vector<nn::Tensor> train_samples(10, sample);

    lstm::LSTMAutoencoder model(arch_cfg);
    nn::training::Trainer<lstm::LSTMAutoencoder> trainer(model, train_cfg);
    const auto history = trainer.fit_autoencoder(train_samples);

    ASSERT_EQ(static_cast<int>(history.size()), train_cfg.epochs);
    EXPECT_TRUE(std::isfinite(history.front().train_loss));
    EXPECT_TRUE(std::isfinite(history.back().train_loss));
    EXPECT_LT(history.back().train_loss, history.front().train_loss);
}

TEST(TrainerTest, ValidationLossFiniteWhenProvided)
{
    auto arch_cfg = small_cfg();
    arch_cfg.input_size = 4;
    arch_cfg.seq_len = 5;
    arch_cfg.hidden_size = 8;
    arch_cfg.latent_size = 2;

    TrainerConfig train_cfg;
    train_cfg.epochs = 3;
    train_cfg.learning_rate = 1e-3f;
    train_cfg.grad_clip_norm = 0.0f;
    train_cfg.batch_size = 1;

    nn::Tensor sample(arch_cfg.seq_len, arch_cfg.input_size);
    sample.fill(0.3f);
    std::vector<nn::Tensor> train_s(5, sample);
    std::vector<nn::Tensor> val_s(2, sample);

    lstm::LSTMAutoencoder model(arch_cfg);
    nn::training::Trainer<lstm::LSTMAutoencoder> trainer(model, train_cfg);
    const auto history = trainer.fit_autoencoder(train_s, val_s);

    for (const auto& r : history)
    {
        EXPECT_TRUE(std::isfinite(r.val_loss));
        EXPECT_GE(r.val_loss, 0.0f);
    }
}

} // namespace
