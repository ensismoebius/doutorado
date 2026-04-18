/**
 * @file LSTMAutoencoder_gtest.cpp
 * @brief Unit tests for experiment04 — LSTM Autoencoder.
 *
 * Coverage:
 *   - LSTMLayer: shape invariants, state reset, parameter pointer stability.
 *   - LSTMLayer: gradient non-zero after backward.
 *   - LSTMAutoencoder: encode/decode round-trip shapes.
 *   - LSTMAutoencoder: forward/backward shape consistency.
 *   - LSTMAutoencoder: state_dict round-trip.
 *   - Trainer: loss decreases over a small synthetic dataset.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "Experiment04Config.hpp"
#include "LSTMAutoencoder.hpp"
#include "LSTMLayer.hpp"
#include "Trainer.hpp"
#include "nn/tensor/Tensor.hpp"

namespace
{

// ============================================================
// LSTMLayer tests
// ============================================================

TEST(LSTMLayerTest, ForwardOutputShape)
{
    // LSTMLayer with input_size=4, hidden_size=8
    // Input [T=5 × D=4] → forward → [T=5 × H=8]
    experiment04::LSTMLayer lstm(4, 8);
    nn::Tensor input = nn::Tensor::rand(5, 4);
    nn::Tensor out = lstm.forward(input, /*requires_grad=*/false);

    EXPECT_EQ(out.rows(), 5u);
    EXPECT_EQ(out.cols(), 8u);
}

TEST(LSTMLayerTest, BackwardInputGradShape)
{
    // Backward must return [T × D]
    experiment04::LSTMLayer lstm(4, 8);
    nn::Tensor input = nn::Tensor::rand(5, 4);
    nn::Tensor all_h = lstm.forward(input, /*requires_grad=*/true);
    nn::Tensor grad_h = nn::Tensor::ones(5, 8);
    nn::Tensor grad_in = lstm.backward(grad_h);

    EXPECT_EQ(grad_in.rows(), 5u);
    EXPECT_EQ(grad_in.cols(), 4u);
}

TEST(LSTMLayerTest, WeightGradientsNonZeroAfterBackward)
{
    experiment04::LSTMLayer lstm(3, 6);
    nn::Tensor input = nn::Tensor::rand(4, 3);
    lstm.forward(input, true);
    lstm.backward(nn::Tensor::ones(4, 6));

    // W, U, b gradients should be non-zero
    float w_norm = lstm.W_.grad().norm();
    float u_norm = lstm.U_.grad().norm();
    float b_norm = lstm.b_.grad().norm();

    EXPECT_GT(w_norm, 0.0f);
    EXPECT_GT(u_norm, 0.0f);
    EXPECT_GT(b_norm, 0.0f);
}

TEST(LSTMLayerTest, ResetStateClearsCache)
{
    experiment04::LSTMLayer lstm(4, 8);
    nn::Tensor input = nn::Tensor::rand(5, 4);
    lstm.forward(input, true);

    // Cache should be populated
    EXPECT_EQ(lstm.cache_.size(), 5u);

    lstm.reset_state();
    EXPECT_EQ(lstm.cache_.size(), 0u);
}

TEST(LSTMLayerTest, ParamsReturnsThreePointers)
{
    // W, U, b
    experiment04::LSTMLayer lstm(4, 8);
    auto p = lstm.params();
    EXPECT_EQ(p.size(), 3u);
    for (nn::Tensor* ptr : p)
    {
        EXPECT_NE(ptr, nullptr);
    }
}

TEST(LSTMLayerTest, StateDictRoundTrip)
{
    experiment04::LSTMLayer lstm(4, 8);
    // Modify W so the loaded value can be verified
    lstm.W_.at(0, 0) = 99.0f;

    auto sd = lstm.state_dict();
    ASSERT_EQ(sd.count("W"), 1u);

    experiment04::LSTMLayer lstm2(4, 8);
    lstm2.load_state_dict(sd);
    EXPECT_FLOAT_EQ(lstm2.W_.at(0, 0), 99.0f);
}

TEST(LSTMLayerTest, BackwardCalledBeforeForwardThrows)
{
    experiment04::LSTMLayer lstm(4, 8);
    nn::Tensor grad = nn::Tensor::ones(5, 8);
    EXPECT_THROW(lstm.backward(grad), std::runtime_error);
}

// ============================================================
// LSTMAutoencoder shape tests
// ============================================================

namespace
{
experiment04::LSTMAutoencoderConfig small_cfg()
{
    experiment04::LSTMAutoencoderConfig cfg;
    cfg.input_size = 8;
    cfg.seq_len = 6;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.num_layers = 1;
    return cfg;
}
} // namespace

TEST(LSTMAutoencoderTest, EncodeOutputShape)
{
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    nn::Tensor z = model.encode(input, false);

    EXPECT_EQ(z.rows(), 1u);
    EXPECT_EQ(z.cols(), static_cast<nn::Index>(cfg.latent_size));
}

TEST(LSTMAutoencoderTest, DecodeOutputShape)
{
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);

    nn::Tensor z = nn::Tensor::rand(1, cfg.latent_size);
    nn::Tensor recon = model.decode(z, cfg.seq_len, false);

    EXPECT_EQ(recon.rows(), static_cast<nn::Index>(cfg.seq_len));
    EXPECT_EQ(recon.cols(), static_cast<nn::Index>(cfg.input_size));
}

TEST(LSTMAutoencoderTest, ForwardOutputShape)
{
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    nn::Tensor recon = model.forward(input, false);

    EXPECT_EQ(recon.rows(), input.rows());
    EXPECT_EQ(recon.cols(), input.cols());
}

TEST(LSTMAutoencoderTest, BackwardInputGradShape)
{
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    nn::Tensor recon = model.forward(input, true);
    nn::Tensor grad_recon = nn::Tensor::ones(cfg.seq_len, cfg.input_size);
    nn::Tensor grad_in = model.backward(grad_recon);

    EXPECT_EQ(grad_in.rows(), static_cast<nn::Index>(cfg.seq_len));
    EXPECT_EQ(grad_in.cols(), static_cast<nn::Index>(cfg.input_size));
}

TEST(LSTMAutoencoderTest, ParamsNonEmpty)
{
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);
    EXPECT_GT(model.params().size(), 0u);
}

TEST(LSTMAutoencoderTest, ResetStateClearsLSTMCaches)
{
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    model.forward(input, true);

    // After reset, no LSTM should hold cached steps
    model.reset_state();
    for (const auto& lstm : model.enc_lstms_)
    {
        EXPECT_EQ(lstm->cache_.size(), 0u);
    }
    for (const auto& lstm : model.dec_lstms_)
    {
        EXPECT_EQ(lstm->cache_.size(), 0u);
    }
}

TEST(LSTMAutoencoderTest, LatentBoundedByTanh)
{
    // tanh output is in (-1, 1); verify all latent elements satisfy this
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);

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
    auto cfg = small_cfg();
    experiment04::LSTMAutoencoder model(cfg);

    // Set a sentinel value in the output projection weight
    model.out_proj_->weight.at(0, 0) = 77.0f;

    auto sd = model.state_dict();
    EXPECT_FALSE(sd.empty());

    // Load into a fresh model
    experiment04::LSTMAutoencoder model2(cfg);
    model2.load_state_dict(sd);
    EXPECT_FLOAT_EQ(model2.out_proj_->weight.at(0, 0), 77.0f);
}

TEST(LSTMAutoencoderTest, MultiLayerForwardShape)
{
    auto cfg = small_cfg();
    cfg.num_layers = 2;
    experiment04::LSTMAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(cfg.seq_len, cfg.input_size);
    nn::Tensor recon = model.forward(input, false);

    EXPECT_EQ(recon.rows(), input.rows());
    EXPECT_EQ(recon.cols(), input.cols());
}

// ============================================================
// Trainer: loss-decreasing smoke test
// ============================================================

TEST(TrainerTest, LossDecreasesOnSyntheticData)
{
    // Build a tiny model and train for a few epochs on a single repeated sample.
    // Reconstruction loss should strictly decrease (or at least not diverge).
    auto arch_cfg = small_cfg();
    arch_cfg.input_size = 4;
    arch_cfg.seq_len = 5;
    arch_cfg.hidden_size = 8;
    arch_cfg.latent_size = 2;

    Experiment04Config train_cfg;
    train_cfg.input_size = arch_cfg.input_size;
    train_cfg.seq_len = arch_cfg.seq_len;
    train_cfg.hidden_size = arch_cfg.hidden_size;
    train_cfg.latent_size = arch_cfg.latent_size;
    train_cfg.num_layers = arch_cfg.num_layers;
    train_cfg.epochs = 20;
    train_cfg.learning_rate = 1e-3f;
    train_cfg.grad_clip_norm = 1.0f;
    train_cfg.sampler_shuffle_seed = 7u;

    // Training sample: constant signal (easy to reconstruct)
    nn::Tensor sample(arch_cfg.seq_len, arch_cfg.input_size);
    for (nn::Index i = 0; i < static_cast<nn::Index>(arch_cfg.seq_len); ++i)
        for (nn::Index j = 0; j < static_cast<nn::Index>(arch_cfg.input_size); ++j)
            sample.at(i, j) = 0.5f;

    std::vector<nn::Tensor> train_samples(10, sample);

    experiment04::LSTMAutoencoder model(arch_cfg);
    experiment04::Trainer trainer(model, train_cfg);
    auto history = trainer.fit(train_samples);

    ASSERT_EQ(static_cast<int>(history.size()), train_cfg.epochs);

    float first_loss = history.front().train_loss;
    float last_loss = history.back().train_loss;

    // Loss should be finite throughout
    EXPECT_TRUE(std::isfinite(first_loss));
    EXPECT_TRUE(std::isfinite(last_loss));

    // Loss should have decreased by at least some amount after 20 epochs
    EXPECT_LT(last_loss, first_loss);
}

TEST(TrainerTest, ValidationLossFiniteWhenProvided)
{
    auto arch_cfg = small_cfg();
    arch_cfg.input_size = 4;
    arch_cfg.seq_len = 5;
    arch_cfg.hidden_size = 8;
    arch_cfg.latent_size = 2;

    Experiment04Config train_cfg;
    train_cfg.input_size = arch_cfg.input_size;
    train_cfg.seq_len = arch_cfg.seq_len;
    train_cfg.hidden_size = arch_cfg.hidden_size;
    train_cfg.latent_size = arch_cfg.latent_size;
    train_cfg.num_layers = arch_cfg.num_layers;
    train_cfg.epochs = 3;
    train_cfg.learning_rate = 1e-3f;
    train_cfg.grad_clip_norm = 0.0f; // disabled

    nn::Tensor sample(arch_cfg.seq_len, arch_cfg.input_size);
    sample.fill(0.3f);
    std::vector<nn::Tensor> train_s(5, sample);
    std::vector<nn::Tensor> val_s(2, sample);

    experiment04::LSTMAutoencoder model(arch_cfg);
    experiment04::Trainer trainer(model, train_cfg);
    auto history = trainer.fit(train_s, val_s);

    for (const auto& r : history)
    {
        EXPECT_TRUE(std::isfinite(r.val_loss));
        EXPECT_GE(r.val_loss, 0.0f);
    }
}

} // anonymous namespace
