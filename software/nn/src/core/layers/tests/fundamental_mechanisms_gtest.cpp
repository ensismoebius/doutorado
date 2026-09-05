/**
 * @file fundamental_mechanisms_gtest.cpp
 * @brief Verifies mathematical correctness of every layer's fundamental mechanism:
 *        activations (Sigmoid, Tanh, ReLU, LeakyReLU) and the Linear layer.
 *
 * Tests are organised by layer and cover:
 *  - Known-value forward pass
 *  - Gradient formula correctness (analytic vs numeric finite-difference)
 *  - Shape contracts
 *
 * Finite-difference checker uses eps=1e-3 (safe for float32 activations).
 * Max relative error threshold: 0.01 (1%).
 *
 * The rest of this file's original coverage (convolution/pooling, spiking
 * layers, losses, LSTM/residual) moved to the sibling
 * fundamental_mechanisms_*_gtest.cpp files in this directory once this file
 * crossed LOC_CRITICAL; FundamentalMechanismsHelpers.hpp holds the shared
 * finite-difference checker used by all of them.
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
