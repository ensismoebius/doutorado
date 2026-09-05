/**
 * @file fundamental_mechanisms_spiking_gtest.cpp
 * @brief Fundamental-mechanism correctness for ThresholdDependentBatchNorm, PoissonLatentLayer,
 * Leaky, LifBPTT, and LifIntegrator.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "FundamentalMechanismsHelpers.hpp"
#include "layers/activations/LeakyReLU.hpp"
#include "layers/activations/ReLU.hpp"
#include "layers/activations/Sigmoid.hpp"
#include "layers/activations/Tanh.hpp"
#include "layers/convolution/Conv1d.hpp"
#include "layers/convolution/MaxPool1d.hpp"
#include "layers/convolution/MaxPool2d.hpp"
#include "layers/dense/Linear.hpp"
#include "layers/losses/CrossEntropyLoss.hpp"
#include "layers/losses/MAELoss.hpp"
#include "layers/losses/SpikeTimeLoss.hpp"
#include "layers/lstm/LSTMLayer.hpp"
#include "layers/residual/ResNetBlock.hpp"
#include "layers/residual/ResidualBlock.hpp"
#include "layers/residual/SimpleResNet.hpp"
#include "layers/spiking/Lif.hpp"
#include "layers/spiking/LifBPTT.hpp"
#include "layers/spiking/LifIntegrator.hpp"
#include "layers/spiking/PoissonLatentLayer.hpp"
#include "layers/spiking/ThresholdDependentBatchNorm.hpp"
#include "tensor/Tensor.hpp"

using Tensor = nn::Tensor;
using Backend = nn::Backend;

// ===========================================================================
// TdBNTest — ThresholdDependentBatchNorm
// Ref: Zheng et al., AAAI 2021 (tdBN)
// Canonical formula: output = γ·(α·V_th·x_norm) + β, with mean/var pooled over
// batch AND time (all T·B rows of a feature column), β unscaled. No √T factor.
// ===========================================================================

TEST(TdBNTest, ZeroMean)
{
    // gamma=1, beta=0, zero-mean input → output scaled by α·V_th (α=1 here).
    const int F = 4;
    const float vth = 1.0f;
    const int T = 4;
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, vth, T);

    // (T*B, F) input with B=2, ±1 per feature → pooled mean 0, var 1.
    Tensor x(T * 2, F);
    for (int t = 0; t < T; ++t)
    {
        for (int f = 0; f < F; ++f)
        {
            x.at(t * 2, f) = 1.0f;
            x.at(t * 2 + 1, f) = -1.0f;
        }
    }
    Tensor out = tdbn.forward(x, false);

    // Canonical scale is α·V_th (no √T). x_norm = ±1/sqrt(1+eps).
    const float scale = 1.0f * vth; // α·V_th
    const float expected_abs = scale / std::sqrt(1.0f + 1e-5f);
    for (int t = 0; t < T; ++t)
    {
        for (int f = 0; f < F; ++f)
        {
            float a = out.at(t * 2, f);
            float b = out.at(t * 2 + 1, f);
            EXPECT_NEAR(a + b, 0.0f, 1e-4f); // zero mean preserved
            EXPECT_NEAR(std::abs(a), expected_abs, 1e-5f);
            EXPECT_NEAR(std::abs(b), expected_abs, 1e-5f);
        }
    }
}

TEST(TdBNTest, VthScalesOutput)
{
    // Double V_th → double output magnitude (output ∝ α·V_th).
    const int F = 2;
    ThresholdDependentBatchNormImpl<Backend> tdbn1(F, 1.0f, 1);
    ThresholdDependentBatchNormImpl<Backend> tdbn2(F, 2.0f, 1);

    Tensor x(2, F); // T=1, B=2
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(1, 0) = -1.f;
    x.at(1, 1) = -2.f;

    Tensor out1 = tdbn1.forward(x, false);
    Tensor out2 = tdbn2.forward(x, false);

    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c)
            EXPECT_NEAR(out2.at(r, c), 2.0f * out1.at(r, c), 1e-4f);
}

TEST(TdBNTest, AlphaScalesOutput)
{
    // Canonical α hyperparameter scales the output linearly (target std = α·V_th).
    const int F = 2;
    ThresholdDependentBatchNormImpl<Backend> tdbn_a1(F, 1.0f, 1, 1.0f);
    ThresholdDependentBatchNormImpl<Backend> tdbn_a3(F, 1.0f, 1, 3.0f);

    Tensor x(2, F);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(1, 0) = -1.f;
    x.at(1, 1) = -2.f;

    Tensor o1 = tdbn_a1.forward(x, false);
    Tensor o3 = tdbn_a3.forward(x, false);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c)
            EXPECT_NEAR(o3.at(r, c), 3.0f * o1.at(r, c), 1e-4f);
}

TEST(TdBNTest, TimeStepsDoNotChangeScale)
{
    // Canonical tdBN pools statistics over batch AND time, so replicating the same
    // per-feature distribution across more time steps leaves the output unchanged
    // (contrast with the removed per-step V_th/√T heuristic).
    const int F = 2;
    ThresholdDependentBatchNormImpl<Backend> tdbn1(F, 1.0f, 1);
    ThresholdDependentBatchNormImpl<Backend> tdbn4(F, 1.0f, 4);

    Tensor x1(2, F);
    x1.at(0, 0) = 1.f;
    x1.at(0, 1) = 2.f;
    x1.at(1, 0) = -1.f;
    x1.at(1, 1) = -2.f;

    Tensor x4(8, F);
    for (int t = 0; t < 4; ++t)
    {
        x4.at(t * 2, 0) = 1.f;
        x4.at(t * 2, 1) = 2.f;
        x4.at(t * 2 + 1, 0) = -1.f;
        x4.at(t * 2 + 1, 1) = -2.f;
    }

    Tensor out1 = tdbn1.forward(x1, false);
    Tensor out4 = tdbn4.forward(x4, false);

    // Same distribution pooled over T·B rows → identical normalized output.
    EXPECT_NEAR(out4.at(0, 0), out1.at(0, 0), 1e-4f);
    EXPECT_NEAR(out4.at(0, 1), out1.at(0, 1), 1e-4f);
}

TEST(TdBNTest, GammaGradExact)
{
    // Exact d_gamma from asymmetric upstream gradient and explicit x_norm.
    const int F = 3;
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, 1.0f, 1);
    Tensor x(2, F);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(0, 2) = 3.f;
    x.at(1, 0) = 3.f;
    x.at(1, 1) = 6.f;
    x.at(1, 2) = 9.f;
    tdbn.forward(x, true);
    Tensor go(2, F);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c) go.at(r, c) = 1.0f;
    // Non-uniform gradient: x_norm has zero mean so uniform go gives zero d_gamma.
    // Use different go per batch sample so Σ(go_b * x_norm_b) ≠ 0.
    go.at(0, 0) = 2.f;
    go.at(0, 1) = 2.f;
    go.at(0, 2) = 2.f;
    go.at(1, 0) = 1.f;
    go.at(1, 1) = 1.f;
    go.at(1, 2) = 1.f;
    tdbn.backward(go);

    Tensor dgamma = tdbn.gamma.grad();
    for (size_t c = 0; c < static_cast<size_t>(F); ++c)
    {
        float x0 = x.at(0, c);
        float x1 = x.at(1, c);
        float mean = 0.5f * (x0 + x1);
        float var = 0.5f * ((x0 - mean) * (x0 - mean) + (x1 - mean) * (x1 - mean));
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        float x_norm0 = (x0 - mean) * inv_std;
        float x_norm1 = (x1 - mean) * inv_std;
        float expected = go.at(0, c) * x_norm0 + go.at(1, c) * x_norm1;
        EXPECT_NEAR(dgamma.at(0, c), expected, 1e-5f);
    }
}

TEST(TdBNTest, BetaGradExact)
{
    // d_beta = sum over batch of (dout * tdbn_scale)
    const int F = 2;
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, 1.0f, 1);
    Tensor x(2, F);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(1, 0) = -1.f;
    x.at(1, 1) = -2.f;
    tdbn.forward(x, true);
    Tensor go(2, F);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c) go.at(r, c) = 1.0f;
    tdbn.backward(go);

    Tensor dbeta = tdbn.beta.grad();
    // d_beta = Σ dout over the 2 pooled samples = 2 (β is not scaled by α·V_th).
    EXPECT_NEAR(dbeta.at(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(dbeta.at(0, 1), 2.0f, 1e-5f);
}

TEST(TdBNTest, PoolsStatisticsOverBatchAndTime)
{
    // Canonical tdBN pools mean/var over ALL T·B rows of a feature, not per time
    // step. Build T=2, B=2 where the two time steps have different per-step means
    // but a known pooled mean/var; verify normalization uses the pooled statistics.
    const int F = 1;
    const int T = 2;
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, 1.0f, T); // α=1, V_th=1 → scale 1

    Tensor x(4, F); // rows: t0b0, t0b1, t1b0, t1b1
    x.at(0, 0) = 0.f;
    x.at(1, 0) = 2.f; // t0 mean = 1
    x.at(2, 0) = 4.f;
    x.at(3, 0) = 6.f; // t1 mean = 5  (per-step means differ: 1 vs 5)
    Tensor out = tdbn.forward(x, false);

    // Pooled stats over all 4 rows: mean = 3, var = (9+1+1+9)/4 = 5.
    const float mean = 3.0f, var = 5.0f;
    const float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (size_t r = 0; r < 4; ++r) EXPECT_NEAR(out.at(r, 0), (x.at(r, 0) - mean) * inv_std, 1e-4f);
}

TEST(TdBNTest, InferenceUsesRunningStats)
{
    // After training-mode forwards accumulate running stats, eval mode (train(false))
    // must normalize with those running stats, independent of the eval batch.
    const int F = 1;
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, 1.0f, 1);
    tdbn.momentum = 1.0f; // running stat := last batch stat (easy to predict)

    Tensor train_x(2, F);
    train_x.at(0, 0) = 0.f;
    train_x.at(1, 0) = 4.f; // batch mean 2, var 4
    tdbn.train(true);
    tdbn.forward(train_x, false); // updates running_mean=2, running_var=4

    EXPECT_NEAR(tdbn.running_mean.at(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(tdbn.running_var.at(0, 0), 4.0f, 1e-5f);

    // Eval on a DIFFERENT batch: normalization must use running (2,4), not the new
    // batch's own statistics.
    tdbn.train(false);
    Tensor eval_x(2, F);
    eval_x.at(0, 0) = 2.f; // equals running mean → normalized to 0
    eval_x.at(1, 0) = 6.f;
    Tensor out = tdbn.forward(eval_x, false);
    const float inv_std = 1.0f / std::sqrt(4.0f + 1e-5f);
    EXPECT_NEAR(out.at(0, 0), (2.0f - 2.0f) * inv_std, 1e-4f);
    EXPECT_NEAR(out.at(1, 0), (6.0f - 2.0f) * inv_std, 1e-4f);
}

TEST(TdBNTest, ZeroVarianceIsNumericallyStable)
{
    // Constant input → var 0; the eps guard must keep the output finite (no NaN/Inf).
    const int F = 2;
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, 1.0f, 1);
    Tensor x(3, F);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c) x.at(r, c) = 5.0f;
    Tensor out = tdbn.forward(x, true);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c)
            EXPECT_TRUE(std::isfinite(out.at(r, c)));

    Tensor go(3, F);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c) go.at(r, c) = 1.0f;
    Tensor gin = tdbn.backward(go);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < static_cast<size_t>(F); ++c)
            EXPECT_TRUE(std::isfinite(gin.at(r, c)));
}

TEST(TdBNTest, InputGradientMatchesFiniteDifference)
{
    // Validate the batch-norm input gradient (with batch+time coupling) against a
    // central finite difference of the scalar loss L = Σ go·out.
    const int F = 2;
    const size_t N = 4;                                              // pooled samples per feature
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, 1.5f, 1, 1.0f); // scale α·V_th = 1.5

    Tensor x(N, F);
    const float vals[4][2] = {{1.0f, -2.0f}, {3.0f, 0.5f}, {-1.0f, 2.0f}, {2.0f, -0.5f}};
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < 2; ++c) x.at(r, c) = vals[r][c];

    Tensor go(N, F);
    const float gv[4][2] = {{0.5f, 1.0f}, {-1.0f, 0.25f}, {2.0f, -0.5f}, {0.75f, 1.5f}};
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < 2; ++c) go.at(r, c) = gv[r][c];

    tdbn.forward(x, true);
    Tensor analytic = tdbn.backward(go);

    auto loss_at = [&](const Tensor& in) -> double
    {
        Tensor out = tdbn.forward(in, false); // batch stats (training_ still true)
        double L = 0.0;
        for (size_t r = 0; r < N; ++r)
            for (size_t c = 0; c < 2; ++c) L += static_cast<double>(go.at(r, c) * out.at(r, c));
        return L;
    };

    const float h = 1e-2f;
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < 2; ++c)
        {
            Tensor xp = x, xm = x;
            xp.at(r, c) += h;
            xm.at(r, c) -= h;
            const double num = (loss_at(xp) - loss_at(xm)) / (2.0 * static_cast<double>(h));
            EXPECT_NEAR(analytic.at(r, c), static_cast<float>(num), 2e-2f)
                << "grad mismatch at (" << r << "," << c << ")";
        }
}

// ===========================================================================
// TdBNTest additions — training loop: gamma and beta must update
// ===========================================================================

TEST(TdBNTest, GammaAndBetaGradsExact)
{
    const int F = 3;
    // Default α=1, eps=1e-5 → scale α·V_th = 1, matching the expected values below.
    ThresholdDependentBatchNormImpl<Backend> tdbn(F, 1.0f, 1);

    Tensor x(2, F);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(0, 2) = 3.f;
    x.at(1, 0) = 3.f;
    x.at(1, 1) = 6.f;
    x.at(1, 2) = 9.f;
    tdbn.forward(x, true);

    Tensor go(2, F);
    go.at(0, 0) = 2.f;
    go.at(0, 1) = 2.f;
    go.at(0, 2) = 2.f;
    go.at(1, 0) = 1.f;
    go.at(1, 1) = 1.f;
    go.at(1, 2) = 1.f;
    tdbn.backward(go);

    const float eps = 1e-5f;
    for (int f = 0; f < F; ++f)
    {
        const float x0 = x.at(0, static_cast<size_t>(f));
        const float x1 = x.at(1, static_cast<size_t>(f));
        const float mean = 0.5f * (x0 + x1);
        const float var = 0.5f * ((x0 - mean) * (x0 - mean) + (x1 - mean) * (x1 - mean));
        const float inv_std = 1.0f / std::sqrt(var + eps);
        const float x_norm0 = (x0 - mean) * inv_std;
        const float x_norm1 = (x1 - mean) * inv_std;

        const float expected_dgamma =
            go.at(0, static_cast<size_t>(f)) * x_norm0 + go.at(1, static_cast<size_t>(f)) * x_norm1;
        const float expected_dbeta =
            go.at(0, static_cast<size_t>(f)) + go.at(1, static_cast<size_t>(f));

        EXPECT_NEAR(tdbn.gamma.grad().at(0, static_cast<size_t>(f)), expected_dgamma, 1e-5f);
        EXPECT_NEAR(tdbn.beta.grad().at(0, static_cast<size_t>(f)), expected_dbeta, 1e-5f);
    }
}

// ===========================================================================
// PoissonLatentTest
// Ref: Kamata et al., AAAI 2022 (FSVAE); Chen et al., arXiv:2310.14839 (ESVAE)
// ===========================================================================

TEST(PoissonLatentTest, InferenceRateSoftplusExact)
{
    // Inference mode returns λ=softplus(z) exactly.
    PoissonLatentLayerImpl<Backend> layer(1, 0.1f, 0.0f);
    Tensor z(1, 4);
    z.at(0, 0) = -10.f;
    z.at(0, 1) = -1.f;
    z.at(0, 2) = 0.f;
    z.at(0, 3) = 10.f;
    Tensor rates = layer.forward(z, false); // inference: output IS the rate

    for (size_t c = 0; c < 4; ++c)
    {
        const float zv = z.at(0, c);
        const float expected = (zv > 20.0f) ? zv : std::log1p(std::exp(zv));
        EXPECT_NEAR(rates.at(0, c), expected, 1e-5f);
    }
}

TEST(PoissonLatentTest, KLExactKnownValue)
{
    // KL(Poisson(λ) || Poisson(λ0)) = λ0 - λ + λ*log(λ/λ0), averaged over features.
    PoissonLatentLayerImpl<Backend> layer(1, 0.1f, 1.0f);
    Tensor z(1, 4);
    z.at(0, 0) = -1.f;
    z.at(0, 1) = 0.f;
    z.at(0, 2) = 1.f;
    z.at(0, 3) = 5.f;
    layer.forward(z, true);

    const float lambda0 = 0.1f;
    float expected = 0.0f;
    for (size_t c = 0; c < 4; ++c)
    {
        const float zv = z.at(0, c);
        const float lam = (zv > 20.0f) ? zv : std::log1p(std::exp(zv));
        expected += lambda0 - lam + lam * std::log(lam / lambda0 + 1e-8f);
    }
    expected /= 4.0f;

    EXPECT_NEAR(layer.kl_loss(), expected, 1e-6f);
}

TEST(PoissonLatentTest, KLZeroAtPrior)
{
    // KL = 0 when λ = λ₀
    // softplus(z) = λ₀ → z = log(exp(λ₀) - 1)
    const float lambda0 = 0.1f;
    const float z_val = std::log(std::expm1(lambda0)); // softplus^{-1}(lambda0)

    PoissonLatentLayerImpl<Backend> layer(1, lambda0, 1.0f);
    Tensor z(1, 1);
    z.at(0, 0) = z_val;
    layer.forward(z, true);
    // KL should be near 0 when λ = λ₀
    EXPECT_NEAR(layer.kl_loss(), 0.0f, 1e-3f);
}

TEST(PoissonLatentTest, InferenceMean)
{
    // requires_grad=false → return λ directly (no stochastic sampling)
    PoissonLatentLayerImpl<Backend> layer(1, 0.1f, 0.0f);
    Tensor z(1, 2);
    z.at(0, 0) = 0.f; // softplus(0) = log(2) ≈ 0.6931
    z.at(0, 1) = 1.f; // softplus(1) = log(1+e) ≈ 1.3133
    Tensor out = layer.forward(z, false);
    EXPECT_NEAR(out.at(0, 0), std::log1p(std::exp(0.f)), 1e-4f);
    EXPECT_NEAR(out.at(0, 1), std::log1p(std::exp(1.f)), 1e-4f);
}

TEST(PoissonLatentTest, GradExactNoKL)
{
    // With beta_kl=0 and T=1: dL/dz = dL/d_out * sigmoid(z)
    PoissonLatentLayerImpl<Backend> layer(1, 0.1f, 0.0f);
    Tensor z(1, 3);
    z.at(0, 0) = -1.f;
    z.at(0, 1) = 0.f;
    z.at(0, 2) = 1.f;
    layer.forward(z, true);
    Tensor go(1, 3);
    go.at(0, 0) = 1.f;
    go.at(0, 1) = 1.f;
    go.at(0, 2) = 1.f;
    Tensor dz = layer.backward(go);

    EXPECT_NEAR(dz.at(0, 0), nn::activation::sigmoid(-1.0f), 1e-5f);
    EXPECT_NEAR(dz.at(0, 1), nn::activation::sigmoid(0.0f), 1e-5f);
    EXPECT_NEAR(dz.at(0, 2), nn::activation::sigmoid(1.0f), 1e-5f);
}

// ===========================================================================
// LeakyTest additions — V_th gradient and full training loop
// ===========================================================================

TEST(LeakyTest, VthreshGradExact)
{
    // Two-step deterministic setup: warm-up to create nonzero v(t-1), then spike.
    LifImpl<Backend> layer(1.0f, 1.0f, 1.0f, 0.5f);
    Tensor x_warm(1, 1);
    x_warm.at(0, 0) = 0.3f;
    layer.forward(x_warm, false);

    Tensor x(1, 1);
    x.at(0, 0) = 2.0f;
    layer.forward(x, true);
    Tensor go = Tensor::ones(1, 1);
    layer.backward(go);

    const float beta = std::exp(-1.0f);
    const float v_pre = beta * 0.3f + 2.0f;
    const float surr = std::exp(-std::abs(v_pre - 0.5f));
    const float expected_dVth = -surr;
    EXPECT_NEAR(layer.voltage_threshold.grad().at(0, 0), expected_dVth, 1e-5f);
}

TEST(LeakyTest, AllParamsGradExact)
{
    // For one feature: dVth=-surr, dR=dC=(surr*v_prev)*d_beta_d{R,C}, with
    // d_beta_dR=d_beta_dC=exp(-1).
    LifImpl<Backend> layer(1.0f, 1.0f, 1.0f, 0.5f);
    Tensor x_warm(1, 1);
    x_warm.at(0, 0) = 0.3f;
    layer.forward(x_warm, false);

    Tensor x(1, 1);
    x.at(0, 0) = 2.0f;
    layer.forward(x, true);
    Tensor go = Tensor::ones(1, 1);
    layer.backward(go);

    const float beta = std::exp(-1.0f);
    const float v_pre = beta * 0.3f + 2.0f;
    const float surr = std::exp(-std::abs(v_pre - 0.5f));
    const float dL_dbeta = surr * 0.3f;
    const float expected_dVth = -surr;
    const float expected_dR = dL_dbeta * beta;
    const float expected_dC = dL_dbeta * beta;

    EXPECT_NEAR(layer.voltage_threshold.grad().at(0, 0), expected_dVth, 1e-5f);
    EXPECT_NEAR(layer.resistance.grad().at(0, 0), expected_dR, 1e-5f);
    EXPECT_NEAR(layer.capacitance.grad().at(0, 0), expected_dC, 1e-5f);
}

// ===========================================================================
// LifBPTTTest additions — training loop verification
// ===========================================================================

TEST(LifBPTTTest, AllParamsGradExact)
{
    // T=2, B=1, F=1 deterministic sequence.
    LifBPTTImpl<Backend> layer(2, 1.0f, 1.0f, 1.0f, 0.5f);
    Tensor x(2, 1);
    x.at(0, 0) = 0.3f;
    x.at(1, 0) = 3.0f;
    layer.forward(x, true);
    Tensor go = Tensor::ones(2, 1);
    layer.backward(go);

    const float beta = std::exp(-1.0f);
    const float v0_pre = 0.3f;
    const float v1_pre = beta * 0.3f + 3.0f;
    const float surr0 = std::exp(-std::abs(v0_pre - 0.5f));
    const float surr1 = std::exp(-std::abs(v1_pre - 0.5f));

    // Derived from current LifBPTT backward equations for reset_zero=true.
    const float expected_dVth = -surr1 - surr0 + (surr1 * beta * (surr0 * (v0_pre - 0.0f)));
    const float expected_dR = surr1 * 0.3f * beta;
    const float expected_dC = surr1 * 0.3f * beta;

    EXPECT_NEAR(layer.voltage_threshold.grad().at(0, 0), expected_dVth, 1e-5f);
    EXPECT_NEAR(layer.resistance.grad().at(0, 0), expected_dR, 1e-5f);
    EXPECT_NEAR(layer.capacitance.grad().at(0, 0), expected_dC, 1e-5f);
}

// ===========================================================================
// LifIntegratorTest — known-value forward + V_th grad zero + RC training loop
// ===========================================================================

TEST(LifIntegratorTest, KnownValueForward)
{
    // V[t] = beta * V[t-1] + input[t], beta = exp(-dt/(R*C)).
    // dt=1, R=1, C=1 → beta = exp(-1).
    // Step 1 from zero: V[0] = exp(-1)*0 + 2 = 2.
    // Step 2 same input: V[1] = exp(-1)*2 + 2.
    LifIntegratorImpl<Backend> layer(1.0f, 1.0f, 1.0f);
    Tensor x(1, 1);
    x.at(0, 0) = 2.0f;

    Tensor out1 = layer.forward(x, false);
    Tensor out2 = layer.forward(x, false);

    const float beta = std::exp(-1.0f);
    EXPECT_NEAR(out1.at(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(out2.at(0, 0), beta * 2.0f + 2.0f, 1e-4f);
}

TEST(LifIntegratorTest, VthreshGradAlwaysZero)
{
    // voltage_threshold is in params() (inherited from Lif) but LifIntegrator
    // never uses it in forward/backward — its gradient must always be exactly zero.
    LifIntegratorImpl<Backend> layer(1.0f, 1.0f, 1.0f);
    Tensor x(1, 2);
    x.at(0, 0) = 1.0f;
    x.at(0, 1) = 2.0f;
    layer.forward(x, true);
    Tensor go(1, 2);
    go.at(0, 0) = 1.0f;
    go.at(0, 1) = 1.0f;
    layer.backward(go);
    // V_th gradient must be exactly 0 — no spike path, so surrogate is never evaluated
    EXPECT_EQ(layer.voltage_threshold.grad().at(0, 0), 0.0f)
        << "LifIntegrator: V_th never used in forward/backward, gradient must be 0";
}

TEST(LifIntegratorTest, RCParamsGradExact)
{
    // Two-step deterministic protocol: warm-up to set v(t-1), then exact R/C gradients.
    LifIntegratorImpl<Backend> layer(1.0f, 1.0f, 1.0f);

    Tensor x_warm(1, 1);
    x_warm.at(0, 0) = 0.3f;
    layer.forward(x_warm, false);

    Tensor x(1, 1);
    x.at(0, 0) = 2.0f;
    layer.forward(x, true);

    Tensor go = Tensor::ones(1, 1);
    layer.backward(go);

    const float beta = std::exp(-1.0f);
    const float expected_dR = 0.3f * beta;
    const float expected_dC = 0.3f * beta;

    EXPECT_NEAR(layer.resistance.grad().at(0, 0), expected_dR, 1e-5f);
    EXPECT_NEAR(layer.capacitance.grad().at(0, 0), expected_dC, 1e-5f);
    EXPECT_NEAR(layer.voltage_threshold.grad().at(0, 0), 0.0f, 1e-5f);
}
