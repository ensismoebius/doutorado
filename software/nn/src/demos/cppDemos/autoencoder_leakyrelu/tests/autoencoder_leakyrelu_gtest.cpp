/**
 * @file autoencoder_leakyrelu_gtest.cpp
 * @brief GoogleTest unit tests for autoencoder_leakyrelu demos.
 *
 * Covers:
 * 1. Dense MSE autoencoder (Sequential of Linear+LeakyReLU) — reconstruction quality.
 * 2. Spike autoencoder (Sequential of Linear+LifBPTT) — output shape invariant.
 *
 * Synthetic data only.  No filesystem access.
 */

#include <cmath>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "initializers/kaiming_snn.hpp"
#include "layers/Layers.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include "testing.hpp"
#include "utility/synthetic_spike_data.hpp"

using nn::LeakyReLU;
using nn::Lif;
using nn::LifBPTT;
using nn::Linear;
using nn::MSELoss;
using nn::Sequential;

// ---- Helpers ----

static nn::Tensor make_random_input(size_t rows, size_t cols, unsigned int seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    nn::Tensor t(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            t.at(i, j) = dist(rng);
    return t;
}

// ============================================================
// Dense MSE Autoencoder tests
// ============================================================

class DenseAutoencoderTest : public ::testing::Test
{
   protected:
    static constexpr int kInputDim = 16;
    static constexpr int kHidden = 8;
    static constexpr int kLatent = 4;
    static constexpr int kBatch = 8;

    // Encoder: input → hidden → latent
    std::shared_ptr<Linear> enc1;
    std::shared_ptr<LeakyReLU> act1;
    std::shared_ptr<Linear> enc2;

    // Decoder: latent → hidden → output
    std::shared_ptr<LeakyReLU> act2;
    std::shared_ptr<Linear> dec1;
    std::shared_ptr<LeakyReLU> act3;
    std::shared_ptr<Linear> dec2;

    Sequential encoder;
    Sequential decoder;

    void SetUp() override
    {
        enc1 = std::make_shared<Linear>(kInputDim, kHidden);
        act1 = std::make_shared<LeakyReLU>();
        enc2 = std::make_shared<Linear>(kHidden, kLatent);
        act2 = std::make_shared<LeakyReLU>();
        dec1 = std::make_shared<Linear>(kLatent, kHidden);
        act3 = std::make_shared<LeakyReLU>();
        dec2 = std::make_shared<Linear>(kHidden, kInputDim);

        encoder = Sequential({enc1, act1, enc2});
        decoder = Sequential({act2, dec1, act3, dec2});

        kaimingSNNInitializer(enc1, nn::testing::kSeed);
        kaimingSNNInitializer(enc2, nn::testing::kSeed);
        kaimingSNNInitializer(dec1, nn::testing::kSeed);
        kaimingSNNInitializer(dec2, nn::testing::kSeed);
    }
};

TEST_F(DenseAutoencoderTest, ForwardOutputShape)
{
    nn::Tensor x = make_random_input(kBatch, kInputDim);
    nn::Tensor latent = encoder.forward(x, false);
    nn::Tensor recon = decoder.forward(latent, false);
    EXPECT_EQ(recon.rows(), static_cast<size_t>(kBatch));
    EXPECT_EQ(recon.cols(), static_cast<size_t>(kInputDim));
}

TEST_F(DenseAutoencoderTest, ForwardOutputIsFinite)
{
    nn::Tensor x = make_random_input(kBatch, kInputDim);
    nn::Tensor recon = decoder.forward(encoder.forward(x, false), false);
    for (size_t i = 0; i < recon.rows(); ++i)
        for (size_t j = 0; j < recon.cols(); ++j)
            EXPECT_TRUE(std::isfinite(recon.at(i, j)));
}

TEST_F(DenseAutoencoderTest, MSELoss_PositiveBeforeTraining)
{
    nn::Tensor x = make_random_input(kBatch, kInputDim);
    nn::Tensor recon = decoder.forward(encoder.forward(x), false);
    MSELoss loss;
    loss.set_target(x);
    nn::Tensor loss_t = loss.forward(recon);
    EXPECT_GT(loss_t.at(0, 0), 0.0F);
}

TEST_F(DenseAutoencoderTest, MSELoss_DecreasesOverTenEpochs)
{
    nn::Tensor x = make_random_input(kBatch, kInputDim, 7);

    // Collect all params
    auto ep = encoder.params();
    auto dp = decoder.params();
    std::vector<nn::Tensor*> all_params(ep.begin(), ep.end());
    all_params.insert(all_params.end(), dp.begin(), dp.end());

    Adam optimizer(0.01F);
    optimizer.attach(all_params);

    MSELoss loss_fn;
    float first_loss = 0.0F;
    float last_loss = 0.0F;

    for (int epoch = 0; epoch < 10; ++epoch)
    {
        loss_fn.set_target(x);
        nn::Tensor latent = encoder.forward(x);
        nn::Tensor recon = decoder.forward(latent);
        nn::Tensor loss_t = loss_fn.forward(recon);
        nn::Tensor grad = loss_fn.backward(recon);
        decoder.backward(grad);
        nn::Tensor latent_grad = encoder.backward(latent);  // unused but exercises backprop
        optimizer.step(all_params);

        if (epoch == 0) first_loss = loss_t.at(0, 0);
        last_loss = loss_t.at(0, 0);
    }

    EXPECT_LT(last_loss, first_loss) << "MSE loss did not decrease over 10 epochs";
}

TEST_F(DenseAutoencoderTest, LatentSpaceIsSmaller)
{
    nn::Tensor x = make_random_input(kBatch, kInputDim);
    nn::Tensor latent = encoder.forward(x, false);
    EXPECT_EQ(latent.cols(), static_cast<size_t>(kLatent));
    EXPECT_LT(latent.cols(), static_cast<size_t>(kInputDim));
}

// ============================================================
// Spike autoencoder output shape test
// ============================================================

class SpikeAutoencoderTest : public ::testing::Test
{
   protected:
    static constexpr int kInputDim = 8;
    static constexpr int kHidden = 4;
    static constexpr int kBatch = 2;
    static constexpr int kSteps = 5;

    std::shared_ptr<Linear> fc_enc;
    std::shared_ptr<LifBPTT> lif_enc;
    std::shared_ptr<Linear> fc_dec;
    std::shared_ptr<LifBPTT> lif_dec;

    Sequential encoder;
    Sequential decoder;

    void SetUp() override
    {
        fc_enc = std::make_shared<Linear>(kInputDim, kHidden);
        lif_enc = std::make_shared<LifBPTT>(kSteps);
        fc_dec = std::make_shared<Linear>(kHidden, kInputDim);
        lif_dec = std::make_shared<LifBPTT>(kSteps);

        encoder = Sequential({fc_enc, lif_enc});
        decoder = Sequential({fc_dec, lif_dec});

        kaimingSNNInitializer(fc_enc, nn::testing::kSeed);
        kaimingSNNInitializer(fc_dec, nn::testing::kSeed);
    }
};

TEST_F(SpikeAutoencoderTest, SpikeOutputShape_TimeMajor)
{
    // Time-major input: (T*B, F)
    auto [inp_seq, _] = generate_autoencoder_spike_data_of_ones(kBatch, kInputDim, kSteps);

    // Flatten to time-major (T*B, F)
    nn::Tensor flat(static_cast<size_t>(kSteps * kBatch), static_cast<size_t>(kInputDim));
    for (int t = 0; t < kSteps; ++t)
        for (int b = 0; b < kBatch; ++b)
            for (int f = 0; f < kInputDim; ++f)
                flat.at(static_cast<size_t>(t * kBatch + b), static_cast<size_t>(f)) =
                    inp_seq[static_cast<size_t>(t)].at(
                        static_cast<size_t>(b), static_cast<size_t>(f));

    nn::Tensor latent = encoder.forward(flat, false);
    nn::Tensor recon = decoder.forward(latent, false);

    // Output shape must match input (T*B, kInputDim)
    EXPECT_EQ(recon.rows(), static_cast<size_t>(kSteps * kBatch));
    EXPECT_EQ(recon.cols(), static_cast<size_t>(kInputDim));
}

TEST_F(SpikeAutoencoderTest, SpikeOutputIsBinary)
{
    auto [inp_seq, _] = generate_autoencoder_spike_data_of_ones(kBatch, kInputDim, kSteps);

    nn::Tensor flat(static_cast<size_t>(kSteps * kBatch), static_cast<size_t>(kInputDim));
    for (int t = 0; t < kSteps; ++t)
        for (int b = 0; b < kBatch; ++b)
            for (int f = 0; f < kInputDim; ++f)
                flat.at(static_cast<size_t>(t * kBatch + b), static_cast<size_t>(f)) =
                    inp_seq[static_cast<size_t>(t)].at(
                        static_cast<size_t>(b), static_cast<size_t>(f));

    nn::Tensor latent = encoder.forward(flat, false);
    nn::Tensor recon = decoder.forward(latent, false);

    for (size_t i = 0; i < recon.rows(); ++i)
        for (size_t j = 0; j < recon.cols(); ++j)
        {
            float v = recon.at(i, j);
            EXPECT_TRUE(v == 0.0F || v == 1.0F)
                << "Non-binary spike at (" << i << "," << j << "): " << v;
        }
}
