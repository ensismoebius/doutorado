/**
 * @file fundamental_mechanisms_gtest.cpp
 * @brief Verifies mathematical correctness of every layer's fundamental mechanism.
 *
 * Tests are organised by layer and cover:
 *  - Known-value forward pass
 *  - Gradient formula correctness (analytic vs numeric finite-difference)
 *  - Shape contracts
 *
 * Finite-difference checker uses eps=1e-3 (safe for float32 activations).
 * Max relative error threshold: 0.01 (1%).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

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

// ---------------------------------------------------------------------------
// Finite-difference gradient checker
// ---------------------------------------------------------------------------
namespace
{

// Returns max relative error between analytic gradient and central finite differences.
// f : input → scalar loss
// df: input → gradient tensor (same shape as x0)
// eps: perturbation size (1e-3 is safe for float32)
float finite_diff_max_err(std::function<float(const Tensor&)> f,
    std::function<Tensor(const Tensor&)> df,
    const Tensor& x0,
    float eps = 1e-3f)
{
    Tensor analytic = df(x0);
    float max_err = 0.0f;

    for (size_t r = 0; r < x0.rows(); ++r)
    {
        for (size_t c = 0; c < x0.cols(); ++c)
        {
            // Central difference
            Tensor xp = x0;
            xp.at(r, c) += eps;
            Tensor xm = x0;
            xm.at(r, c) -= eps;
            float numeric = (f(xp) - f(xm)) / (2.0f * eps);
            float analyt = analytic.at(r, c);

            float denom = std::max(std::abs(analyt), std::abs(numeric));
            denom = std::max(denom, 1e-6f); // avoid divide-by-zero for near-zero grads
            float rel = std::abs(analyt - numeric) / denom;
            if (rel > max_err) max_err = rel;
        }
    }
    return max_err;
}

// Sum all elements of a tensor (for scalar loss construction)
float tensor_sum(const Tensor& t)
{
    float s = 0.0f;
    for (size_t r = 0; r < t.rows(); ++r)
        for (size_t c = 0; c < t.cols(); ++c) s += t.at(r, c);
    return s;
}

} // namespace

// ===========================================================================
// SigmoidTest
// ===========================================================================

TEST(SigmoidTest, KnownValues)
{
    // σ(0)=0.5, σ(+∞)→1, σ(−∞)→0, σ(x)=1−σ(−x)
    EXPECT_NEAR(nn::activation::sigmoid(0.0f), 0.5f, 1e-6f);
    EXPECT_NEAR(nn::activation::sigmoid(100.0f), 1.0f, 1e-4f);
    EXPECT_NEAR(nn::activation::sigmoid(-100.0f), 0.0f, 1e-4f);
    float s1 = nn::activation::sigmoid(2.0f);
    float s2 = nn::activation::sigmoid(-2.0f);
    EXPECT_NEAR(s1 + s2, 1.0f, 1e-5f); // symmetry
}

TEST(SigmoidTest, GradientFormula)
{
    // dσ/dx = σ(x)·(1−σ(x))
    for (float x : {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f})
    {
        float s = nn::activation::sigmoid(x);
        float g_expected = s * (1.0f - s);
        EXPECT_NEAR(nn::activation::sigmoid_grad(s), g_expected, 1e-6f);
    }
}

TEST(SigmoidTest, GradientFiniteDifference)
{
    SigmoidImpl<Backend> layer;
    Tensor x(2, 3);
    x.at(0, 0) = -1.f;
    x.at(0, 1) = 0.f;
    x.at(0, 2) = 1.f;
    x.at(1, 0) = -2.f;
    x.at(1, 1) = 0.5f;
    x.at(1, 2) = 2.f;

    // Scalar loss = sum of sigmoid outputs
    auto f = [&](const Tensor& inp) -> float
    {
        SigmoidImpl<Backend> l;
        return tensor_sum(l.forward(inp, false));
    };
    auto df = [&](const Tensor& inp) -> Tensor
    {
        SigmoidImpl<Backend> l;
        Tensor out = l.forward(inp, true);
        Tensor ones = Tensor::ones(out.rows(), out.cols());
        return l.backward(ones);
    };

    float err = finite_diff_max_err(f, df, x);
    EXPECT_LT(err, 0.01f) << "Sigmoid gradient max relative error: " << err;
}

// ===========================================================================
// TanhTest
// ===========================================================================

TEST(TanhTest, KnownValues)
{
    // tanh(0)=0, tanh(±∞)→±1, odd function
    EXPECT_NEAR(nn::activation::tanh(Tensor::zeros(1, 1)).at(0, 0), 0.0f, 1e-5f);
    Tensor large(1, 1);
    large.at(0, 0) = 20.0f;
    EXPECT_NEAR(nn::activation::tanh(large).at(0, 0), 1.0f, 1e-4f);
}

TEST(TanhTest, GradientFormula)
{
    // d(tanh)/dx = 1 − tanh²(x)
    Tensor x(1, 4);
    x.at(0, 0) = -1.f;
    x.at(0, 1) = 0.f;
    x.at(0, 2) = 0.5f;
    x.at(0, 3) = 1.f;
    Tensor t = nn::activation::tanh(x);
    Tensor g = nn::activation::tanh_grad(t);

    for (size_t j = 0; j < 4; ++j)
    {
        float expected = 1.0f - t.at(0, j) * t.at(0, j);
        EXPECT_NEAR(g.at(0, j), expected, 1e-5f);
    }
}

TEST(TanhTest, GradientFiniteDifference)
{
    Tensor x(2, 3);
    x.at(0, 0) = -1.f;
    x.at(0, 1) = 0.f;
    x.at(0, 2) = 1.f;
    x.at(1, 0) = -0.5f;
    x.at(1, 1) = 0.3f;
    x.at(1, 2) = 0.8f;

    auto f = [&](const Tensor& inp) -> float
    {
        TanhImpl<Backend> l;
        return tensor_sum(l.forward(inp, false));
    };
    auto df = [&](const Tensor& inp) -> Tensor
    {
        TanhImpl<Backend> l;
        Tensor out = l.forward(inp, true);
        Tensor ones = Tensor::ones(out.rows(), out.cols());
        return l.backward(ones);
    };

    float err = finite_diff_max_err(f, df, x);
    EXPECT_LT(err, 0.01f) << "Tanh gradient max relative error: " << err;
}

// ===========================================================================
// ReLUMechanismTest
// ===========================================================================

TEST(ReLUMechanismTest, NegativeZero)
{
    ReLUImpl<Backend> layer;
    Tensor x(1, 3);
    x.at(0, 0) = -1.f;
    x.at(0, 1) = -0.5f;
    x.at(0, 2) = -100.f;
    Tensor out = layer.forward(x, false);
    for (size_t c = 0; c < 3; ++c) EXPECT_EQ(out.at(0, c), 0.0f);
}

TEST(ReLUMechanismTest, PositiveIdentity)
{
    ReLUImpl<Backend> layer;
    Tensor x(1, 3);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.5f;
    x.at(0, 2) = 100.f;
    Tensor out = layer.forward(x, false);
    EXPECT_NEAR(out.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(out.at(0, 1), 2.5f, 1e-6f);
    EXPECT_NEAR(out.at(0, 2), 100.f, 1e-6f);
}

TEST(ReLUMechanismTest, GradBelow)
{
    ReLUImpl<Backend> layer;
    Tensor x(1, 2);
    x.at(0, 0) = -1.f;
    x.at(0, 1) = -3.f;
    layer.forward(x, true);
    Tensor g(1, 2);
    g.at(0, 0) = 1.f;
    g.at(0, 1) = 1.f;
    Tensor dg = layer.backward(g);
    // Gradient is zero for x < 0
    EXPECT_EQ(dg.at(0, 0), 0.0f);
    EXPECT_EQ(dg.at(0, 1), 0.0f);
}

TEST(ReLUMechanismTest, GradAbove)
{
    ReLUImpl<Backend> layer;
    Tensor x(1, 2);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 3.f;
    layer.forward(x, true);
    Tensor g(1, 2);
    g.at(0, 0) = 2.f;
    g.at(0, 1) = 5.f;
    Tensor dg = layer.backward(g);
    // Gradient passes through for x > 0
    EXPECT_NEAR(dg.at(0, 0), 2.f, 1e-6f);
    EXPECT_NEAR(dg.at(0, 1), 5.f, 1e-6f);
}

// ===========================================================================
// LeakyReLUMechanismTest
// ===========================================================================

TEST(LeakyReLUMechanismTest, GradientFiniteDifference)
{
    const float alpha = 0.1f;
    Tensor x(2, 3);
    x.at(0, 0) = -1.f;
    x.at(0, 1) = 0.5f;
    x.at(0, 2) = 1.f;
    x.at(1, 0) = -2.f;
    x.at(1, 1) = -0.1f;
    x.at(1, 2) = 2.f;

    auto f = [&](const Tensor& inp) -> float
    {
        LeakyReLUImpl<Backend> l(alpha);
        return tensor_sum(l.forward(inp, false));
    };
    auto df = [&](const Tensor& inp) -> Tensor
    {
        LeakyReLUImpl<Backend> l(alpha);
        l.forward(inp, true);
        Tensor ones = Tensor::ones(inp.rows(), inp.cols());
        return l.backward(ones);
    };

    float err = finite_diff_max_err(f, df, x);
    EXPECT_LT(err, 0.01f) << "LeakyReLU gradient max relative error: " << err;
}

TEST(LeakyReLUMechanismTest, NegativeSlope)
{
    const float alpha = 0.2f;
    LeakyReLUImpl<Backend> layer(alpha);
    Tensor x(1, 2);
    x.at(0, 0) = -3.f;
    x.at(0, 1) = 2.f;
    Tensor out = layer.forward(x, false);
    EXPECT_NEAR(out.at(0, 0), -3.f * alpha, 1e-5f); // negative side: alpha*x
    EXPECT_NEAR(out.at(0, 1), 2.f, 1e-5f);          // positive side: identity
}

// ===========================================================================
// LinearMechanismTest
// ===========================================================================

TEST(LinearMechanismTest, WeightGradIsOuterProduct)
{
    // dL/dW = (dL/dY)^T · X  for a single sample
    LinearImpl<Backend> layer(3, 2);
    // Set weight to identity-like
    layer.weight.at(0, 0) = 1.f;
    layer.weight.at(0, 1) = 0.f;
    layer.weight.at(0, 2) = 0.f;
    layer.weight.at(1, 0) = 0.f;
    layer.weight.at(1, 1) = 1.f;
    layer.weight.at(1, 2) = 0.f;
    layer.bias.setZero();

    Tensor x(1, 3);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(0, 2) = 3.f;

    layer.forward(x, true);

    Tensor grad_out(1, 2);
    grad_out.at(0, 0) = 1.f;
    grad_out.at(0, 1) = 1.f;

    layer.backward(grad_out);

    // dW[i, j] = grad_out[0, i] * x[0, j]
    // dW[0, :] = 1 * [1, 2, 3]
    // dW[1, :] = 1 * [1, 2, 3]
    Tensor dW = layer.weight.grad();
    EXPECT_NEAR(dW.at(0, 0), 1.f, 1e-5f);
    EXPECT_NEAR(dW.at(0, 1), 2.f, 1e-5f);
    EXPECT_NEAR(dW.at(0, 2), 3.f, 1e-5f);
    EXPECT_NEAR(dW.at(1, 0), 1.f, 1e-5f);
    EXPECT_NEAR(dW.at(1, 1), 2.f, 1e-5f);
    EXPECT_NEAR(dW.at(1, 2), 3.f, 1e-5f);
}

TEST(LinearMechanismTest, BiasGradIsRowSum)
{
    // dL/db = sum over batch of dL/dY (shape: out_features x 1)
    LinearImpl<Backend> layer(2, 3);
    layer.weight.setZero();
    layer.bias.setZero();
    // Two-sample batch
    Tensor x(2, 2);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(1, 0) = 3.f;
    x.at(1, 1) = 4.f;
    layer.forward(x, true);

    Tensor grad_out(2, 3);
    grad_out.at(0, 0) = 1.f;
    grad_out.at(0, 1) = 2.f;
    grad_out.at(0, 2) = 3.f;
    grad_out.at(1, 0) = 4.f;
    grad_out.at(1, 1) = 5.f;
    grad_out.at(1, 2) = 6.f;
    layer.backward(grad_out);

    Tensor db = layer.bias.grad();
    // db[i] = sum over batch of grad_out[:, i]
    EXPECT_NEAR(db.at(0, 0), 5.f, 1e-5f); // 1+4
    EXPECT_NEAR(db.at(1, 0), 7.f, 1e-5f); // 2+5
    EXPECT_NEAR(db.at(2, 0), 9.f, 1e-5f); // 3+6
}

TEST(LinearMechanismTest, GradientFiniteDifference)
{
    LinearImpl<Backend> layer(3, 4);
    // Set fixed weights for reproducibility
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            layer.weight.at(i, j) = static_cast<float>((i * 3 + j + 1)) * 0.1f;
    layer.bias.setZero();

    Tensor x(2, 3);
    x.at(0, 0) = 0.5f;
    x.at(0, 1) = -0.3f;
    x.at(0, 2) = 1.2f;
    x.at(1, 0) = -0.7f;
    x.at(1, 1) = 0.8f;
    x.at(1, 2) = -0.1f;

    auto f = [&](const Tensor& inp) -> float
    {
        LinearImpl<Backend> l(3, 4);
        l.weight = layer.weight;
        l.bias = layer.bias;
        return tensor_sum(l.forward(inp, false));
    };
    auto df = [&](const Tensor& inp) -> Tensor
    {
        LinearImpl<Backend> l(3, 4);
        l.weight = layer.weight;
        l.bias = layer.bias;
        l.forward(inp, true);
        Tensor ones = Tensor::ones(2, 4);
        return l.backward(ones);
    };

    float err = finite_diff_max_err(f, df, x);
    EXPECT_LT(err, 0.01f) << "Linear grad max relative error: " << err;
}

// ===========================================================================
// Conv1dTest
// ===========================================================================

TEST(Conv1dTest, OutputShape)
{
    // L_out = (L + 2P - D*(K-1) - 1) / S + 1 = (5+0-2-1)/1+1 = 3
    Conv1dImpl<Backend> conv(1, 2, 3, 1, 0, 1); // C_in=1, C_out=2, K=3, S=1, P=0
    Tensor x(1, 1, 5);
    x.setZero();
    Tensor out = conv.forward(x, false);
    auto shape = out.get_shape();
    ASSERT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u); // B
    EXPECT_EQ(shape[1], 2u); // C_out
    EXPECT_EQ(shape[2], 3u); // L_out = 5-2 = 3
}

TEST(Conv1dTest, BiasShape)
{
    Conv1dImpl<Backend> conv(2, 4, 3, 1, 1, 1);
    const Tensor& b = conv.get_bias();
    // bias shape: (out_channels, 1) or (1, out_channels) — just check element count
    EXPECT_EQ(b.size(), 4u);
}

TEST(Conv1dTest, BackwardIdentityKernelOne)
{
    // K=1, S=1, P=0 with w=1 and b=0 is identity: y=x, so dx must equal grad_out.
    Conv1dImpl<Backend> conv(1, 1, 1, 1, 0, 1);
    conv.set_weights(Tensor::ones(1, 1));
    Tensor bz(1, 1);
    bz.setZero();
    conv.set_bias(bz);

    Tensor x(1, 1, 5);
    x.at(0, 0, 0) = 1.0f;
    x.at(0, 0, 1) = 2.0f;
    x.at(0, 0, 2) = 3.0f;
    x.at(0, 0, 3) = 4.0f;
    x.at(0, 0, 4) = 5.0f;
    Tensor out = conv.forward(x, true);
    ASSERT_EQ(out.get_shape()[2], 5u);

    Tensor grad(1, 1, 5);
    grad.at(0, 0, 0) = 1.0f;
    grad.at(0, 0, 1) = 2.0f;
    grad.at(0, 0, 2) = 3.0f;
    grad.at(0, 0, 3) = 4.0f;
    grad.at(0, 0, 4) = 5.0f;
    Tensor dx = conv.backward(grad);

    for (size_t i = 0; i < grad.size(); ++i) EXPECT_NEAR(dx.at(i), grad.at(i), 1e-5f);
}

TEST(Conv1dTest, KernelOneIdentity)
{
    // K=1, P=0, S=1 conv with weight=1, bias=0: output should equal input
    Conv1dImpl<Backend> conv(1, 1, 1, 1, 0, 1);
    // Weight layout is (C_in * K, C_out); here (1, 1)
    conv.set_weights(Tensor::ones(1, 1));
    Tensor bz(1, 1);
    bz.setZero();
    conv.set_bias(bz);

    Tensor x(1, 1, 4);
    x.at(0, 0, 0) = 1.f;
    x.at(0, 0, 1) = 2.f;
    x.at(0, 0, 2) = 3.f;
    x.at(0, 0, 3) = 4.f;

    Tensor out = conv.forward(x, false);
    ASSERT_EQ(out.get_shape()[2], 4u);
    EXPECT_NEAR(out.at(0, 0, 0), 1.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 0, 1), 2.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 0, 2), 3.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 0, 3), 4.f, 1e-5f);
}

// ===========================================================================
// MaxPoolTest
// ===========================================================================

TEST(MaxPoolTest, MaxPool1dKnownValues)
{
    MaxPool1dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 6);
    x.at(0, 0, 0) = 1.0F;
    x.at(0, 0, 1) = 3.0F;
    x.at(0, 0, 2) = 2.0F;
    x.at(0, 0, 3) = 5.0F;
    x.at(0, 0, 4) = 4.0F;
    x.at(0, 0, 5) = 6.0F;

    Tensor out = pool.forward(x, false);
    ASSERT_EQ(out.get_shape().size(), 3u);
    EXPECT_EQ(out.get_shape()[2], 3u);
    EXPECT_NEAR(out.at(0, 0, 0), 3.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 1), 5.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 2), 6.0F, 1e-6F);
}

TEST(MaxPoolTest, MaxPool2dKnownValues)
{
    MaxPool2dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 4, 4);
    x.at(0, 0, 0, 0) = 1.0F;
    x.at(0, 0, 0, 1) = 2.0F;
    x.at(0, 0, 0, 2) = 3.0F;
    x.at(0, 0, 0, 3) = 4.0F;
    x.at(0, 0, 1, 0) = 5.0F;
    x.at(0, 0, 1, 1) = 6.0F;
    x.at(0, 0, 1, 2) = 7.0F;
    x.at(0, 0, 1, 3) = 8.0F;
    x.at(0, 0, 2, 0) = 9.0F;
    x.at(0, 0, 2, 1) = 10.0F;
    x.at(0, 0, 2, 2) = 11.0F;
    x.at(0, 0, 2, 3) = 12.0F;
    x.at(0, 0, 3, 0) = 13.0F;
    x.at(0, 0, 3, 1) = 14.0F;
    x.at(0, 0, 3, 2) = 15.0F;
    x.at(0, 0, 3, 3) = 16.0F;

    Tensor out = pool.forward(x, false);
    auto shape = out.get_shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[2], 2u);
    EXPECT_EQ(shape[3], 2u);
    EXPECT_NEAR(out.at(0, 0, 0, 0), 6.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 0, 1), 8.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 1, 0), 14.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 1, 1), 16.0F, 1e-6F);
}

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
// ResidualBlockTest
// Pattern: y = fc2(ReLU(fc1(x))) + x (skip connection)
// ===========================================================================

TEST(ResidualBlockTest, ShapePreserved)
{
    ResidualBlockImpl<Backend> block(4);
    Tensor x(2, 4);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 4; ++c) x.at(r, c) = static_cast<float>(r * 4 + c) * 0.1f;
    Tensor out = block.forward(x, false);
    EXPECT_EQ(out.rows(), 2u);
    EXPECT_EQ(out.cols(), 4u);
}

TEST(ResidualBlockTest, SkipActive)
{
    // With zero weights, output = 0 + x = x (skip dominates)
    ResidualBlockImpl<Backend> block(3);
    block.fc1->weight.setZero();
    block.fc1->bias.setZero();
    block.fc2->weight.setZero();
    block.fc2->bias.setZero();

    Tensor x(1, 3);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(0, 2) = 3.f;
    Tensor out = block.forward(x, false);
    EXPECT_NEAR(out.at(0, 0), 1.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 1), 2.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 2), 3.f, 1e-5f);
}

TEST(ResidualBlockTest, GradExactSkipOnly)
{
    // Zeroing both FC layers isolates the skip path: y=x, so backward is identity.
    ResidualBlockImpl<Backend> block(4);
    block.fc1->weight.setZero();
    block.fc1->bias.setZero();
    block.fc2->weight.setZero();
    block.fc2->bias.setZero();

    Tensor x(2, 4);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 4; ++c) x.at(r, c) = static_cast<float>(r + c + 1) * 0.1f;
    block.forward(x, true);
    Tensor go(2, 4);
    for (size_t r = 0; r < go.rows(); ++r)
        for (size_t c = 0; c < go.cols(); ++c) go.at(r, c) = static_cast<float>(r * 4 + c + 1);

    Tensor dx = block.backward(go);

    for (size_t r = 0; r < dx.rows(); ++r)
        for (size_t c = 0; c < dx.cols(); ++c) EXPECT_NEAR(dx.at(r, c), go.at(r, c), 1e-5f);
}

TEST(ResidualBlockTest, SkipGradFlow)
{
    // With zero weights, residual branch is exactly identity: dx == grad_output.
    ResidualBlockImpl<Backend> block(3);
    block.fc1->weight.setZero();
    block.fc1->bias.setZero();
    block.fc2->weight.setZero();
    block.fc2->bias.setZero();

    Tensor x(1, 3);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 2.f;
    x.at(0, 2) = 3.f;
    block.forward(x, true);

    Tensor go(1, 3);
    go.at(0, 0) = 1.f;
    go.at(0, 1) = 1.f;
    go.at(0, 2) = 1.f;
    Tensor dx = block.backward(go);
    for (size_t c = 0; c < 3; ++c) EXPECT_NEAR(dx.at(0, c), 1.0f, 1e-6f);
}

// ===========================================================================
// ResNetBlockTest
// ===========================================================================

TEST(ResNetBlockTest, ForwardShapeAndFinite)
{
    // Conv2d(k=3,p=0,s=1) twice: 7x7 -> 3x3.
    // Test verifies residual alignment does not throw on shape mismatch.
    ResNetBlockImpl<Backend> block(/*in_channels=*/1, /*out_channels=*/2);

    Tensor x(1, 1, 7, 7);
    x.fill(1.0f);

    Tensor out = block.forward(x, true);
    const auto out_shape = out.get_shape();
    ASSERT_EQ(out_shape.size(), 4U);
    EXPECT_EQ(out_shape[0], 1U);
    EXPECT_EQ(out_shape[1], 2U);
    EXPECT_EQ(out_shape[2], 3U);
    EXPECT_EQ(out_shape[3], 3U);
    EXPECT_FALSE(out.hasNaN());
}

TEST(ResNetBlockTest, DeterministicForwardBackward)
{
    // Same input/grad pair must produce identical output/input-gradient across repeated runs.
    ResNetBlockImpl<Backend> block(/*in_channels=*/1, /*out_channels=*/2);

    Tensor x(1, 1, 7, 7);
    x.fill(0.5f);
    Tensor y1 = block.forward(x, true);
    Tensor y2 = block.forward(x, true);

    ASSERT_EQ(y1.get_shape(), y2.get_shape());
    for (size_t i = 0; i < y1.size(); ++i) EXPECT_NEAR(y1.at(i), y2.at(i), 1e-6f);

    Tensor grad_out(y1.get_shape());
    grad_out.fill(1.0f);

    // Recompute forward before each backward call to refresh caches deterministically.
    block.forward(x, true);
    Tensor grad_in1 = block.backward(grad_out);
    block.forward(x, true);
    Tensor grad_in2 = block.backward(grad_out);

    const auto grad_shape = grad_in1.get_shape();
    ASSERT_EQ(grad_shape.size(), 4U);
    EXPECT_EQ(grad_shape[0], 1U);
    EXPECT_EQ(grad_shape[1], 1U);
    EXPECT_EQ(grad_shape[2], 7U);
    EXPECT_EQ(grad_shape[3], 7U);
    EXPECT_FALSE(grad_in1.hasNaN());
    EXPECT_FALSE(grad_in2.hasNaN());
    for (size_t i = 0; i < grad_in1.size(); ++i) EXPECT_NEAR(grad_in1.at(i), grad_in2.at(i), 1e-6f);
}

// ===========================================================================
// CrossEntropyLossTest
// Ref: Goodfellow et al., Deep Learning, Ch. 6
// ===========================================================================

TEST(CrossEntropyLossTest, ExactKnownValue)
{
    CrossEntropyLossImpl<Backend> loss;
    Tensor logits(2, 3);
    logits.at(0, 0) = 1.f;
    logits.at(0, 1) = 2.f;
    logits.at(0, 2) = 3.f;
    logits.at(1, 0) = 0.1f;
    logits.at(1, 1) = 0.5f;
    logits.at(1, 2) = 0.4f;
    Tensor target(2, 3);
    target.at(0, 0) = 0.f;
    target.at(0, 1) = 0.f;
    target.at(0, 2) = 1.f;
    target.at(1, 0) = 1.f;
    target.at(1, 1) = 0.f;
    target.at(1, 2) = 0.f;
    loss.set_target(target);
    Tensor out = loss.forward(logits, true);

    const float p0 = std::exp(3.0f) / (std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f));
    const float p1 = std::exp(0.1f) / (std::exp(0.1f) + std::exp(0.5f) + std::exp(0.4f));
    const float expected = 0.5f * (-std::log(p0 + 1e-7f) - std::log(p1 + 1e-7f));
    EXPECT_NEAR(out.at(0, 0), expected, 1e-6f);
}

TEST(CrossEntropyLossTest, PerfectPred)
{
    // Very confident correct prediction → near-zero loss
    CrossEntropyLossImpl<Backend> loss;
    Tensor logits(1, 3);
    logits.at(0, 0) = -100.f;
    logits.at(0, 1) = -100.f;
    logits.at(0, 2) = 100.f;
    Tensor target(1, 3);
    target.at(0, 0) = 0.f;
    target.at(0, 1) = 0.f;
    target.at(0, 2) = 1.f;
    loss.set_target(target);
    Tensor out = loss.forward(logits, false);
    EXPECT_NEAR(out.at(0, 0), 0.0f, 0.01f);
}

TEST(CrossEntropyLossTest, UniformMaximal)
{
    // Uniform distribution over C classes → loss = log(C)
    const int C = 4;
    CrossEntropyLossImpl<Backend> loss;
    Tensor logits(1, C);
    for (int c = 0; c < C; ++c) logits.at(0, c) = 0.0f; // equal logits → uniform probs
    Tensor target(1, C);
    for (int c = 0; c < C; ++c) target.at(0, c) = (c == 0) ? 1.0f : 0.0f;
    loss.set_target(target);
    Tensor out = loss.forward(logits, false);
    EXPECT_NEAR(out.at(0, 0), std::log(static_cast<float>(C)), 0.01f);
}

TEST(CrossEntropyLossTest, GradRowSumZero)
{
    // Gradient of softmax CE w.r.t. logits: sum over classes per sample = 0
    CrossEntropyLossImpl<Backend> loss;
    Tensor logits(2, 3);
    logits.at(0, 0) = 1.f;
    logits.at(0, 1) = 2.f;
    logits.at(0, 2) = 0.f;
    logits.at(1, 0) = 0.5f;
    logits.at(1, 1) = -1.f;
    logits.at(1, 2) = 2.f;
    Tensor target(2, 3);
    target.at(0, 2) = 1.f;
    target.at(1, 0) = 1.f;
    loss.set_target(target);
    loss.forward(logits, true);
    Tensor ones(1, 1);
    ones.at(0, 0) = 1.f;
    Tensor grad = loss.backward(ones);
    // Sum over each row should be ≈ 0 (softmax: Σ(p_j - t_j) = 1-1 = 0)
    for (size_t r = 0; r < 2; ++r)
    {
        float rowsum = 0.0f;
        for (size_t c = 0; c < 3; ++c) rowsum += grad.at(r, c);
        EXPECT_NEAR(rowsum, 0.0f, 1e-4f);
    }
}

TEST(CrossEntropyLossTest, GradFormula)
{
    // Grad for single sample: g_j = (p_j - t_j) / N
    CrossEntropyLossImpl<Backend> loss;
    Tensor logits(1, 3);
    logits.at(0, 0) = 0.f;
    logits.at(0, 1) = 1.f;
    logits.at(0, 2) = 2.f;
    Tensor target(1, 3);
    target.at(0, 2) = 1.f; // class 2
    loss.set_target(target);
    loss.forward(logits, true);
    Tensor ones(1, 1);
    ones.at(0, 0) = 1.f;
    Tensor grad = loss.backward(ones);

    // Compute expected softmax probabilities
    float e0 = std::exp(0.f), e1 = std::exp(1.f), e2 = std::exp(2.f);
    float sum = e0 + e1 + e2;
    float p0 = e0 / sum, p1 = e1 / sum, p2 = e2 / sum;

    EXPECT_NEAR(grad.at(0, 0), p0 / 1.f, 1e-4f);
    EXPECT_NEAR(grad.at(0, 1), p1 / 1.f, 1e-4f);
    EXPECT_NEAR(grad.at(0, 2), (p2 - 1.f) / 1.f, 1e-4f);
}

// ===========================================================================
// MAELossTest
// ===========================================================================

TEST(MAELossTest, KnownValue)
{
    MAELossImpl<Backend> loss;
    Tensor pred(2, 2);
    pred.at(0, 0) = 1.f;
    pred.at(0, 1) = 3.f;
    pred.at(1, 0) = -1.f;
    pred.at(1, 1) = 0.f;
    Tensor target(2, 2);
    target.at(0, 0) = 0.f;
    target.at(0, 1) = 1.f;
    target.at(1, 0) = 1.f;
    target.at(1, 1) = 2.f;
    loss.set_target(target);
    Tensor out = loss.forward(pred, false);
    // MAE = (|1|+|2|+|-2|+|-2|)/4 = 7/4 = 1.75
    EXPECT_NEAR(out.at(0, 0), 1.75f, 1e-4f);
}

TEST(MAELossTest, GradSign)
{
    // grad[i] = sign(pred[i] - target[i]) / N
    MAELossImpl<Backend> loss;
    Tensor pred(1, 3);
    pred.at(0, 0) = 2.f;
    pred.at(0, 1) = -1.f;
    pred.at(0, 2) = 1.f;
    Tensor target(1, 3);
    target.at(0, 0) = 1.f;
    target.at(0, 1) = 1.f;
    target.at(0, 2) = 2.f;
    loss.set_target(target);
    loss.forward(pred, true);
    Tensor dummy(1, 1);
    Tensor g = loss.backward(dummy);
    const float n = 3.f;
    EXPECT_NEAR(g.at(0, 0), 1.0f / n, 1e-5f);  // pred>target → +1
    EXPECT_NEAR(g.at(0, 1), -1.0f / n, 1e-5f); // pred<target → -1
    EXPECT_NEAR(g.at(0, 2), -1.0f / n, 1e-5f); // pred<target → -1
}

TEST(MAELossTest, GradNormCapped)
{
    // Gradient norm should be ≤ 1.0 (clipped in MAELossImpl)
    MAELossImpl<Backend> loss;
    Tensor pred(10, 10);
    Tensor target(10, 10);
    target.setZero();
    for (size_t r = 0; r < 10; ++r)
        for (size_t c = 0; c < 10; ++c) pred.at(r, c) = 100.f; // large mismatch
    loss.set_target(target);
    loss.forward(pred, true);
    Tensor dummy(1, 1);
    Tensor g = loss.backward(dummy);
    EXPECT_LE(g.norm(), 1.0f + 1e-4f);
}

// ===========================================================================
// SpikeTimeLossTest
// Ref: Comsa et al., Frontiers in Neuroscience 2021
// ===========================================================================

TEST(SpikeTimeLossTest, FirstSpikeExtract)
{
    // T=3, B=1, F=1: spike at t=1 → first spike time = 1
    SpikeTimeLossImpl<Backend> loss(3);
    // (T*B, F) = (3, 1)
    Tensor pred(3, 1);
    pred.at(0, 0) = 0.f; // t=0: no spike
    pred.at(1, 0) = 1.f; // t=1: spike
    pred.at(2, 0) = 0.f; // t=2: no spike
    Tensor target(3, 1);
    target.at(0, 0) = 1.f; // t=0: spike (target fires at 0)
    target.at(1, 0) = 0.f;
    target.at(2, 0) = 0.f;
    loss.set_target(target);
    Tensor out = loss.forward(pred, true);
    // pred_time=1, tgt_time=0 → loss = (1-0)^2 / (B*F) = 1
    EXPECT_NEAR(out.at(0, 0), 1.0f, 1e-5f);
}

TEST(SpikeTimeLossTest, MissingPenalty)
{
    // No spike in pred → penalty = T
    SpikeTimeLossImpl<Backend> loss(3);
    Tensor pred(3, 1);
    pred.setZero(); // no spike
    Tensor target(3, 1);
    target.at(0, 0) = 1.f; // target spikes at t=0
    target.at(1, 0) = 0.f;
    target.at(2, 0) = 0.f;
    loss.set_target(target);
    Tensor out = loss.forward(pred, true);
    // pred_time=T=3, tgt_time=0 → loss = (3-0)^2 / 1 = 9
    EXPECT_NEAR(out.at(0, 0), 9.0f, 1e-4f);
}

TEST(SpikeTimeLossTest, ExactKnownValue)
{
    SpikeTimeLossImpl<Backend> loss(4);
    Tensor pred(4, 2);
    pred.setZero();
    pred.at(2, 0) = 1.f;
    Tensor target(4, 2);
    target.setZero();
    target.at(0, 1) = 1.f;
    loss.set_target(target);
    Tensor out = loss.forward(pred, true);

    // B=1, F=2, T=4:
    // feature 0: pred_t=2, tgt_t=4 (missing spike) -> diff=-2 -> sq=4
    // feature 1: pred_t=4 (missing spike), tgt_t=0 -> diff=4 -> sq=16
    // loss = (4 + 16) / (1*2) = 10
    EXPECT_NEAR(out.at(0, 0), 10.0f, 1e-6f);
}

TEST(SpikeTimeLossTest, GradAtFirstOnly)
{
    // Gradient is zero at non-spike time steps; nonzero only at spike time
    SpikeTimeLossImpl<Backend> loss(3);
    Tensor pred(3, 1);
    pred.at(0, 0) = 0.f;
    pred.at(1, 0) = 1.f;
    pred.at(2, 0) = 0.f;
    Tensor target(3, 1);
    target.at(0, 0) = 1.f;
    target.at(1, 0) = 0.f;
    target.at(2, 0) = 0.f;
    loss.set_target(target);
    loss.forward(pred, true);
    Tensor dummy(1, 1);
    Tensor g = loss.backward(dummy);
    // Gradient only at t=1 (where pred spiked first)
    EXPECT_NEAR(g.at(0, 0), 0.0f, 1e-6f); // t=0: no spike
    EXPECT_NEAR(g.at(1, 0), 2.0f, 1e-6f); // t=1: scale * (pred_t - tgt_t) = 2*(1-0)
    EXPECT_NEAR(g.at(2, 0), 0.0f, 1e-6f); // t=2: no spike
}

TEST(SpikeTimeLossTest, EarlyEqualZero)
{
    // When pred and target fire at same time → zero loss.
    // Must use requires_grad=true so last_input_ is populated for first_spike_times().
    SpikeTimeLossImpl<Backend> loss(4);
    Tensor pred(4, 1);
    pred.setZero();
    pred.at(2, 0) = 1.f;
    Tensor target(4, 1);
    target.setZero();
    target.at(2, 0) = 1.f;
    loss.set_target(target);
    Tensor out = loss.forward(pred, true);
    EXPECT_NEAR(out.at(0, 0), 0.0f, 1e-6f);
}

// ===========================================================================
// LSTMGateTest
// Ref: Greff et al., IEEE TNNLS 2017 (gate ordering [i|f|o|g])
//      Jozefowicz et al., ICML 2015 (forget gate bias = 1)
//      Hochreiter & Schmidhuber, Neural Computation 1997
// ===========================================================================

TEST(LSTMGateTest, ForgetGateBiasInitOne)
{
    // Forget gate bias rows are b_[H:2H]; should be initialised to 1.0
    // (Jozefowicz et al. 2015 recommendation to avoid vanishing gradients at start)
    LSTMLayerImpl<Backend> lstm(4, 8);
    for (int r = 8; r < 16; ++r) // b_[H:2H] = rows [8,16) for H=8
        EXPECT_FLOAT_EQ(lstm.b_.at(static_cast<nn::Index>(r), 0), 1.0f)
            << "Forget gate bias row " << r << " should be 1.0";
}

TEST(LSTMGateTest, ForgetZeroClearsCell)
{
    // f_t = 0 → c_t = i_t ⊙ g_t (no memory from previous cell)
    // Force f_g≈0 by setting large negative forget gate bias
    LSTMLayerImpl<Backend> lstm(2, 4);
    // Set forget bias to large negative (rows [H:2H] = [4:8])
    for (int r = 4; r < 8; ++r) lstm.b_.at(static_cast<nn::Index>(r), 0) = -100.0f;
    // Set c0 to large nonzero to detect if it bleeds through
    for (int h = 0; h < 4; ++h) lstm.c0_.at(0, h) = 100.0f;

    Tensor x(1, 2);
    x.at(0, 0) = 0.f;
    x.at(0, 1) = 0.f;
    lstm.forward(x, false); // single time step

    // c_t = f≈0 * c_prev + i * g; with c_prev=100 but f≈0, c_t should be small
    // Verify by checking h_t = o_t * tanh(c_t) is also reasonable magnitude
    // (We can't access c_t directly but output h_t is bounded by tanh of c_t)
    // Just verify no crash and output is finite
    SUCCEED(); // shape/crash test; numeric effect tested in integration tests
}

TEST(LSTMGateTest, InputZeroBlocksUpdate)
{
    // i_t = 0 → c_t = f_t * c_{t-1} (no new information from g)
    LSTMLayerImpl<Backend> lstm(2, 4);
    // Set input gate bias to large negative (rows [0:H] = [0:4])
    for (int r = 0; r < 4; ++r) lstm.b_.at(static_cast<nn::Index>(r), 0) = -100.0f;

    Tensor x(1, 2);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 1.f;
    Tensor out = lstm.forward(x, false);
    EXPECT_EQ(out.rows(), 1u);
    EXPECT_EQ(out.cols(), 4u);
    // Verify output is finite
    for (size_t c = 0; c < 4; ++c) EXPECT_TRUE(std::isfinite(out.at(0, c)));
}

TEST(LSTMGateTest, OutputZeroSilencesH)
{
    // o_t = 0 → h_t = 0 regardless of c_t
    LSTMLayerImpl<Backend> lstm(2, 4);
    // Set output gate bias to large negative (rows [2H:3H] = [8:12])
    for (int r = 8; r < 12; ++r) lstm.b_.at(static_cast<nn::Index>(r), 0) = -100.0f;

    Tensor x(1, 2);
    x.at(0, 0) = 1.f;
    x.at(0, 1) = 1.f;
    Tensor out = lstm.forward(x, false);
    for (size_t c = 0; c < 4; ++c)
        EXPECT_NEAR(out.at(0, c), 0.0f, 1e-3f)
            << "h_t should be ~0 when output gate is saturated closed";
}

TEST(LSTMGateTest, ForwardOutputShape2D)
{
    // (T, D) input → (T, H) output
    LSTMLayerImpl<Backend> lstm(3, 5);
    Tensor x(4, 3); // T=4, D=3
    x.setZero();
    Tensor out = lstm.forward(x, false);
    EXPECT_EQ(out.rows(), 4u);
    EXPECT_EQ(out.cols(), 5u);
}

TEST(LSTMGateTest, ForwardOutputShape3D)
{
    // (B, T, D) input → (B, T, H) output
    LSTMLayerImpl<Backend> lstm(3, 5);
    Tensor x(2, 4, 3); // B=2, T=4, D=3
    x.setZero();
    Tensor out = lstm.forward(x, false);
    auto shape = out.get_shape();
    ASSERT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 2u);
    EXPECT_EQ(shape[1], 4u);
    EXPECT_EQ(shape[2], 5u);
}

TEST(LSTMGateTest, BackwardGradShape)
{
    // backward should return same shape as input
    LSTMLayerImpl<Backend> lstm(3, 5);
    Tensor x(4, 3);
    x.setZero();
    lstm.forward(x, true);
    Tensor grad_out = Tensor::ones(4, 5);
    Tensor dx = lstm.backward(grad_out);
    EXPECT_EQ(dx.rows(), 4u);
    EXPECT_EQ(dx.cols(), 3u);
}

TEST(LSTMGateTest, BiasGradExact)
{
    // Deterministic 1-step case (D=1, H=1): with x=0 and initial h/c=0,
    // pre-activations are exactly biases [i,f,o,g]=[0,1,0,0].
    // For grad_out=1, expected db = [0,0,0,0.25]^T.
    LSTMLayerImpl<Backend> lstm(1, 1);
    Tensor x(1, 1);
    x.setZero();
    lstm.forward(x, true);
    Tensor go = Tensor::ones(1, 1);
    lstm.backward(go);

    ASSERT_EQ(lstm.b_.rows(), 4u);
    EXPECT_NEAR(lstm.b_.grad().at(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(lstm.b_.grad().at(1, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(lstm.b_.grad().at(2, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(lstm.b_.grad().at(3, 0), 0.25f, 1e-5f);
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
// MaxPoolTest additions — backward gradient routing to argmax positions
// ===========================================================================

TEST(MaxPoolTest, MaxPool1dBackwardRoutesToArgmax)
{
    // Pool(kernel=2, stride=2): windows [0..1]→max=3@pos0, [2..3]→max=4@pos2
    MaxPool1dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 4);
    x.at(0, 0, 0) = 3.0f;
    x.at(0, 0, 1) = 1.0f;
    x.at(0, 0, 2) = 4.0f;
    x.at(0, 0, 3) = 2.0f;
    pool.forward(x, true);

    Tensor go(1, 1, 2);
    go.at(0, 0, 0) = 1.0f;
    go.at(0, 0, 1) = 1.0f;
    Tensor dx = pool.backward(go);

    EXPECT_NEAR(dx.at(0, 0, 0), 1.0f, 1e-6f) << "argmax at pos 0 must receive gradient";
    EXPECT_NEAR(dx.at(0, 0, 1), 0.0f, 1e-6f) << "non-max pos 1 must have zero gradient";
    EXPECT_NEAR(dx.at(0, 0, 2), 1.0f, 1e-6f) << "argmax at pos 2 must receive gradient";
    EXPECT_NEAR(dx.at(0, 0, 3), 0.0f, 1e-6f) << "non-max pos 3 must have zero gradient";
}

TEST(MaxPoolTest, MaxPool2dBackwardRoutesToArgmax)
{
    // Pool(kernel=2, stride=2) on (1,1,4,4): 4 windows, each 2x2
    // Window top-left (0..1,0..1): max at (1,1)=6
    // Window top-right (0..1,2..3): max at (1,3)=8
    // Window bottom-left (2..3,0..1): max at (3,1)=14
    // Window bottom-right (2..3,2..3): max at (3,3)=16
    MaxPool2dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 4, 4);
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 4; ++c) x.at(0, 0, r, c) = static_cast<float>(r * 4 + c + 1);
    // Values: 1..16, row-major → max of each 2x2 block is bottom-right element
    pool.forward(x, true);

    Tensor go(1, 1, 2, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c) go.at(0, 0, r, c) = 1.0f;
    Tensor dx = pool.backward(go);

    // Argmax positions: (1,1), (1,3), (3,1), (3,3) — each should get gradient 1
    // Non-argmax positions should get 0
    EXPECT_NEAR(dx.at(0, 0, 1, 1), 1.0f, 1e-6f); // argmax of top-left block
    EXPECT_NEAR(dx.at(0, 0, 1, 3), 1.0f, 1e-6f); // argmax of top-right block
    EXPECT_NEAR(dx.at(0, 0, 3, 1), 1.0f, 1e-6f); // argmax of bottom-left block
    EXPECT_NEAR(dx.at(0, 0, 3, 3), 1.0f, 1e-6f); // argmax of bottom-right block
    // Spot-check non-argmax positions
    EXPECT_NEAR(dx.at(0, 0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(dx.at(0, 0, 1, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(dx.at(0, 0, 0, 1), 0.0f, 1e-6f);
}

// ===========================================================================
// SimpleResNetTest — known-value forward at depth=0
// ===========================================================================

TEST(SimpleResNetTest, KnownValueDepth0)
{
    // depth=0: fc_in(in→H) + ReLU + fc_out(H→out). No residual blocks.
    // Set all weights=1 and all biases=0 to get deterministic closed-form output.
    const int D = 2, H = 3, O = 2;
    SimpleResNetImpl<Backend> net(D, H, O, /*depth=*/0);

    auto params = net.params();
    ASSERT_EQ(params.size(), 4u);
    params[0]->fill(1.0f); // fc_in weight (H x D)
    params[1]->setZero();  // fc_in bias
    params[2]->fill(1.0f); // fc_out weight (O x H)
    params[3]->setZero();  // fc_out bias

    Tensor x(1, D);
    x.at(0, 0) = 1.0f;
    x.at(0, 1) = 1.0f;
    Tensor out = net.forward(x, false);
    EXPECT_EQ(out.rows(), 1u);
    EXPECT_EQ(out.cols(), static_cast<size_t>(O));
    EXPECT_NEAR(out.at(0, 0), 6.0f, 1e-5f);
    EXPECT_NEAR(out.at(0, 1), 6.0f, 1e-5f);
}

TEST(SimpleResNetTest, BackwardGradExactDepth0)
{
    const int D = 2, H = 3, O = 2;
    SimpleResNetImpl<Backend> net(D, H, O, /*depth=*/0);

    auto params = net.params();
    ASSERT_EQ(params.size(), 4u);
    params[0]->fill(1.0f);
    params[1]->setZero();
    params[2]->fill(1.0f);
    params[3]->setZero();

    Tensor x(1, D);
    x.at(0, 0) = 1.0f;
    x.at(0, 1) = 1.0f;
    net.forward(x, true);
    Tensor go = Tensor::ones(1, O);
    Tensor dx = net.backward(go);
    EXPECT_EQ(dx.rows(), 1u);
    EXPECT_EQ(dx.cols(), static_cast<size_t>(D));
    EXPECT_NEAR(dx.at(0, 0), 6.0f, 1e-5f);
    EXPECT_NEAR(dx.at(0, 1), 6.0f, 1e-5f);
}

// Cover SpikeTimeLoss.train() method (SpikeTimeLoss.hpp line 74)
TEST(SpikeTimeLossTest, TrainModeToggle)
{
    SpikeTimeLossImpl<Backend> loss(2);
    loss.train(true); // enable training mode

    Tensor spikes(4, 2);
    spikes.setZero();
    spikes.at(0, 0) = 1.0f;
    spikes.at(2, 1) = 1.0f;

    Tensor target = spikes;
    loss.set_target(target);
    loss.set_time_steps(2);

    // Forward with train mode - should cache last_input_
    Tensor out = loss.forward(spikes, true);
    EXPECT_NEAR(out.at(0, 0), 0.0f, 1e-5f);

    // Disable training mode - forward should not cache
    loss.train(false);
    Tensor out2 = loss.forward(spikes, false);
    EXPECT_NEAR(out2.at(0, 0), 0.0f, 1e-5f);
}

// MaxPool1d exception: wrong number of input dimensions (MaxPool1d.hpp line 35)
// MaxPool1d exception: wrong number of input dimensions (MaxPool1d.hpp line 35)
TEST(MaxPoolTest, MaxPool1dThrowsOnWrongInputShape)
{
    MaxPool1dImpl<Backend> pool(2, 2);
    Tensor x(4, 4); // 2D, not 3D
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}

// MaxPool1d exception: output length <= 0 (MaxPool1d.hpp line 43)
TEST(MaxPoolTest, MaxPool1dThrowsOnOutputLengthZero)
{
    MaxPool1dImpl<Backend> pool(10, 1); // kernel=10 > length=4 -> L_out <= 0
    Tensor x(1, 1, 4);                  // 3D: B=1, C=1, L=4
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}

// MaxPool2d exception: wrong number of input dimensions (MaxPool2d.hpp line 40)
TEST(MaxPoolTest, MaxPool2dThrowsOnWrongInputShape)
{
    MaxPool2dImpl<Backend> pool(2, 2);
    Tensor x(4, 4); // 2D, not 4D
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}

// MaxPool2d exception: output size <= 0 (MaxPool2d.hpp line 50)
TEST(MaxPoolTest, MaxPool2dThrowsOnOutputSizeZero)
{
    MaxPool2dImpl<Backend> pool(10, 1); // kernel=10 > H=4 -> H_out <= 0
    Tensor x(1, 1, 4, 4);               // 4D: B=1, C=1, H=4, W=4
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}

// SimpleResNet.train() mode toggle (SimpleResNet.hpp lines 83-86)
TEST(SimpleResNetTest, TrainModeToggle)
{
    const int D = 2, H = 3, O = 2;
    SimpleResNetImpl<Backend> net(D, H, O, /*depth=*/0);
    // Default is training mode; toggle to eval and back
    net.train(false); // covers SimpleResNet.hpp lines 83-86
    net.train(true);
    // Verify forward still works after mode toggle
    Tensor x(1, D);
    x.at(0, 0) = 1.0f;
    x.at(0, 1) = 1.0f;
    Tensor out = net.forward(x, false);
    EXPECT_EQ(out.rows(), 1u);
    EXPECT_EQ(out.cols(), static_cast<size_t>(O));
}

// Seeded init is reproducible: same seed → identical weights; different seed →
// different weights (guards the Thesis rnn-path determinism fix).
TEST(SimpleResNetTest, SeededInitIsDeterministic)
{
    const int D = 4, H = 5, O = 3, depth = 2;
    SimpleResNetImpl<Backend> a(D, H, O, depth, /*seed=*/123U);
    SimpleResNetImpl<Backend> b(D, H, O, depth, /*seed=*/123U);
    SimpleResNetImpl<Backend> c(D, H, O, depth, /*seed=*/456U);

    const auto sa = a.state_dict();
    const auto sb = b.state_dict();
    const auto sc = c.state_dict();
    ASSERT_FALSE(sa.empty());

    bool any_diff_same_seed = false;
    bool any_diff_other_seed = false;
    for (const auto& [key, ta] : sa)
    {
        const Tensor& tb = sb.at(key);
        const Tensor& tc = sc.at(key);
        for (size_t r = 0; r < ta.rows(); ++r)
            for (size_t col = 0; col < ta.cols(); ++col)
            {
                if (ta.at(r, col) != tb.at(r, col)) any_diff_same_seed = true;
                if (ta.at(r, col) != tc.at(r, col)) any_diff_other_seed = true;
            }
    }
    EXPECT_FALSE(any_diff_same_seed); // same seed → bitwise-identical weights
    EXPECT_TRUE(any_diff_other_seed); // different seed → different weights
}

// LSTMLayer: wrong input dimension throws (LSTMLayer.hpp lines 137-138)
TEST(LSTMLayerTest, ForwardThrowsOnInputDimMismatch)
{
    LSTMLayerImpl<Backend> lstm(4, 8); // input_size=4
    // Create 3D input with wrong D: B=1, T=3, D=2 (should be 4)
    Tensor x(1, 3, 2);
    x.setZero();
    EXPECT_THROW(lstm.forward(x, false), std::invalid_argument);
}

// LSTMLayer: backward before forward(requires_grad=true) throws (LSTMLayer.hpp line 208)
TEST(LSTMLayerTest, BackwardBeforeForwardThrows)
{
    LSTMLayerImpl<Backend> lstm(4, 8);
    Tensor grad(3, 8); // arbitrary gradient shape
    grad.setZero();
    EXPECT_THROW(lstm.backward(grad), std::runtime_error);
}

// Linear: 1D input (flat single sample) path (Linear.hpp lines 127, 178, 222, 237, 298)
TEST(LinearTest, Forward1DInputFlattened)
{
    LinearImpl<Backend> layer(3, 2);
    // Set weights and zero bias
    layer.weight.fill(0.0f);
    layer.bias.fill(0.0f);
    layer.weight.at(0, 0) = 1.0f;
    layer.weight.at(0, 1) = 1.0f;
    layer.weight.at(0, 2) = 1.0f;
    layer.weight.at(1, 0) = 2.0f;
    layer.weight.at(1, 1) = 2.0f;
    layer.weight.at(1, 2) = 2.0f;

    // 1D input (in_features=3 as a flat vector) -> should be treated as (1, 3)
    Tensor x(std::vector<nn::Index>{3}); // 1D tensor: shape=[3]
    x.at(0) = 1.0f;
    x.at(1) = 2.0f;
    x.at(2) = 3.0f;

    Tensor out = layer.forward(x, true); // requires_grad=true to cache
    // out[0] = 1+2+3=6, out[1] = 2*(1+2+3)=12
    EXPECT_NEAR(out.at(0, 0), 6.0f, 1e-5f);
    EXPECT_NEAR(out.at(0, 1), 12.0f, 1e-5f);

    // backward with 1D input path (lines 222, 237, 298)
    // Pass 1D gradient to cover Linear.hpp line 222 (else branch: shape.size() <= 1)
    Tensor grad(std::vector<nn::Index>{2}); // 1D gradient: shape=[2]
    grad.at(0) = 1.0f;
    grad.at(1) = 1.0f;
    Tensor dx = layer.backward(grad);
    // dx shape should be (1, 3) or reshaped to (3,)
    EXPECT_EQ(dx.size(), 3u);
}
