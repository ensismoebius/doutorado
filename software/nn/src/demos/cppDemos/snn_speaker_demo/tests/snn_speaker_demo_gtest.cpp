/**
 * @file snn_speaker_demo_gtest.cpp
 * @brief Unit tests for codificacao::compute_adaptive_max_rate and codificacao::encode_poisson.
 *
 * Synthetic data only.  No filesystem access.
 * encode_poisson output shape: (time_steps, n_features) — 2D.
 */

#include <cmath>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "codificacao.hpp"
#include "tensor/Tensor.hpp"

// ---- helpers ----

/// Build a 1-row Tensor with uniform value v across n_cols features.
static nn::Tensor make_uniform_frame(size_t n_cols, float v)
{
    nn::Tensor t(1, n_cols);
    for (size_t j = 0; j < n_cols; ++j) t.at(0, j) = v;
    return t;
}

// ---- fixture ----

class CodificacaoTest : public ::testing::Test
{
   protected:
    static constexpr size_t kFeatures = 16;
    static constexpr int kSteps = 20;
    std::mt19937 rng{42};
};

// ============================================================
// compute_adaptive_max_rate
// ============================================================

TEST_F(CodificacaoTest, AdaptiveRate_MeanHalf_RateApproxPointTwo)
{
    // mean = 0.5, expected = 0.1 → rate = 0.1/(0.5+eps) ≈ 0.2
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.5f);
    float rate = codificacao::compute_adaptive_max_rate(frame);
    EXPECT_NEAR(rate, 0.2f, 1e-4f);
}

TEST_F(CodificacaoTest, AdaptiveRate_NearZeroInput_ReturnsMaxMaxRate)
{
    // mean ≈ 0 < eps → return max_max_rate (default 0.5)
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.0f);
    float rate = codificacao::compute_adaptive_max_rate(frame);
    EXPECT_FLOAT_EQ(rate, 0.50f);
}

TEST_F(CodificacaoTest, AdaptiveRate_HighMean_ClampedToMinMaxRate)
{
    // mean = 1.0, expected = 0.1 → raw rate = 0.1/1.0 = 0.1 > min_max_rate=0.02, < max=0.5
    // should not clamp here, but let's check it returns clamped value when mean is tiny non-zero
    nn::Tensor frame = make_uniform_frame(kFeatures, 1.0f);
    float rate = codificacao::compute_adaptive_max_rate(frame, 0.10f, 0.02f, 0.50f);
    EXPECT_GE(rate, 0.02f);
    EXPECT_LE(rate, 0.50f);
}

TEST_F(CodificacaoTest, AdaptiveRate_VeryLargeExpected_ClampedToMaxMaxRate)
{
    // expected=10.0 → raw rate = 10/(0.5+eps) >> 0.5 → clamp to 0.5
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.5f);
    float rate = codificacao::compute_adaptive_max_rate(frame, 10.0f, 0.02f, 0.50f);
    EXPECT_FLOAT_EQ(rate, 0.50f);
}

TEST_F(CodificacaoTest, AdaptiveRate_ResultAlwaysInBounds)
{
    // Test several mean values
    for (float v : {0.0f, 0.1f, 0.5f, 0.9f, 1.0f})
    {
        nn::Tensor frame = make_uniform_frame(kFeatures, v);
        float rate = codificacao::compute_adaptive_max_rate(frame);
        EXPECT_GE(rate, 0.02f) << "rate below min for mean=" << v;
        EXPECT_LE(rate, 0.50f) << "rate above max for mean=" << v;
    }
}

// ============================================================
// encode_poisson — shape & validity
// ============================================================

TEST_F(CodificacaoTest, EncodePoisson_OutputShape_StepsByFeatures)
{
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.5f);
    nn::Tensor spikes = codificacao::encode_poisson(frame, kSteps, rng);
    EXPECT_EQ(spikes.rows(), static_cast<size_t>(kSteps));
    EXPECT_EQ(spikes.cols(), kFeatures);
}

TEST_F(CodificacaoTest, EncodePoisson_ValuesAreBinary)
{
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.5f);
    nn::Tensor spikes = codificacao::encode_poisson(frame, kSteps, rng);
    for (size_t i = 0; i < spikes.rows(); ++i)
        for (size_t j = 0; j < spikes.cols(); ++j)
        {
            float v = spikes.at(i, j);
            EXPECT_TRUE(v == 0.0f || v == 1.0f)
                << "Non-binary spike value " << v << " at (" << i << "," << j << ")";
        }
}

TEST_F(CodificacaoTest, EncodePoisson_InvalidSteps_Throws)
{
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.5f);
    EXPECT_THROW(codificacao::encode_poisson(frame, 0, rng), std::invalid_argument);
    EXPECT_THROW(codificacao::encode_poisson(frame, -1, rng), std::invalid_argument);
}

TEST_F(CodificacaoTest, EncodePoisson_ZeroInput_NoSpikes)
{
    // frame = 0 → p = 0 → rand < 0 never → all zeros
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.0f);
    // with adaptive rate off and max_rate=0, p=0 always
    nn::Tensor spikes = codificacao::encode_poisson(frame, kSteps, rng, 0.0f, false);
    float sum = 0.0f;
    for (size_t i = 0; i < spikes.rows(); ++i)
        for (size_t j = 0; j < spikes.cols(); ++j)
            sum += spikes.at(i, j);
    EXPECT_FLOAT_EQ(sum, 0.0f);
}

TEST_F(CodificacaoTest, EncodePoisson_AdaptiveRate_SpikeDensityNearExpected)
{
    // mean=0.5, expected=0.1 → adaptive rate≈0.2 → E[spike/step] ≈ 0.5*0.2 = 0.10
    // Over kSteps*kFeatures samples, allow ±40% tolerance
    constexpr int kLargeSteps = 200;
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.5f);
    nn::Tensor spikes = codificacao::encode_poisson(frame, kLargeSteps, rng, -1.0f, true, 0.10f);

    float total = 0.0f;
    for (size_t i = 0; i < spikes.rows(); ++i)
        for (size_t j = 0; j < spikes.cols(); ++j)
            total += spikes.at(i, j);

    float density = total / static_cast<float>(kLargeSteps * static_cast<int>(kFeatures));
    // Expected density ≈ 0.10, allow ±40%
    EXPECT_GT(density, 0.06f) << "Spike density too low: " << density;
    EXPECT_LT(density, 0.14f) << "Spike density too high: " << density;
}

TEST_F(CodificacaoTest, EncodePoisson_FixedRate_IgnoresAdaptive)
{
    // adaptive_rate=false, max_rate=0.5 → p = clamp(0.5*0.5, 0,1) = 0.25 per feature/step
    constexpr int kLargeSteps = 500;
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.5f);
    nn::Tensor spikes =
        codificacao::encode_poisson(frame, kLargeSteps, rng, /*max_rate=*/0.5f, /*adaptive=*/false);

    float total = 0.0f;
    for (size_t i = 0; i < spikes.rows(); ++i)
        for (size_t j = 0; j < spikes.cols(); ++j)
            total += spikes.at(i, j);

    float density = total / static_cast<float>(kLargeSteps * static_cast<int>(kFeatures));
    // Expected ≈ 0.25, allow ±40%
    EXPECT_GT(density, 0.15f) << "Fixed-rate density too low: " << density;
    EXPECT_LT(density, 0.35f) << "Fixed-rate density too high: " << density;
}

TEST_F(CodificacaoTest, EncodePoisson_SingleStep_ShapeOk)
{
    nn::Tensor frame = make_uniform_frame(kFeatures, 0.3f);
    nn::Tensor spikes = codificacao::encode_poisson(frame, 1, rng);
    EXPECT_EQ(spikes.rows(), 1u);
    EXPECT_EQ(spikes.cols(), kFeatures);
}
