// layernorm_gtest.cpp — Correctness for LayerNormImpl.
//
// Load-bearing test: central finite-difference gradient check of the input
// gradient and of gamma / beta against the analytic backward pass. Plus the
// forward statistics (zero mean, unit variance per row at gamma=1, beta=0) and
// the shape/contract guards.

#include <gtest/gtest.h>

#include <cmath>

#include "layers/normalization/LayerNorm.hpp"
#include "tensor/Tensor.hpp"

using Backend = nn::Backend;
using Tensor = nn::TensorImpl<Backend>;

namespace
{

constexpr int kN = 4;
constexpr int kD = 6;

auto make_input(unsigned seed) -> Tensor
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-2.0f, 3.0f);
    Tensor x(kN, kD);
    for (nn::Index i = 0; i < kN; ++i)
        for (nn::Index j = 0; j < kD; ++j) x.at(i, j) = d(rng);
    return x;
}

auto quad_loss(const Tensor& y) -> float
{
    float acc = 0.0f;
    for (nn::Index i = 0; i < y.rows(); ++i)
        for (nn::Index j = 0; j < y.cols(); ++j) acc += 0.5f * y.at(i, j) * y.at(i, j);
    return acc;
}

template <typename Mutate>
auto num_grad(LayerNormImpl<Backend>& ln, const Tensor& x, Mutate&& set, float base) -> float
{
    constexpr float kEps = 1e-3f;
    set(base + kEps);
    const float lp = quad_loss(ln.forward(x, false));
    set(base - kEps);
    const float lm = quad_loss(ln.forward(x, false));
    set(base);
    return (lp - lm) / (2.0f * kEps);
}

void expect_close(float analytic, float numeric, const char* what)
{
    const float tol = std::max(2e-2f * std::fabs(analytic), 2e-3f);
    EXPECT_NEAR(analytic, numeric, tol) << what << " a=" << analytic << " n=" << numeric;
}

} // namespace

TEST(LayerNorm, NormalizesRowsAtDefaultParams)
{
    LayerNormImpl<Backend> ln(kD);
    const Tensor x = make_input(1u);
    const Tensor y = ln.forward(x, false);

    for (nn::Index i = 0; i < kN; ++i)
    {
        float mean = 0.0f;
        for (nn::Index j = 0; j < kD; ++j) mean += y.at(i, j);
        mean /= static_cast<float>(kD);
        EXPECT_NEAR(mean, 0.0f, 1e-4f);

        float var = 0.0f;
        for (nn::Index j = 0; j < kD; ++j) var += (y.at(i, j) - mean) * (y.at(i, j) - mean);
        var /= static_cast<float>(kD);
        EXPECT_NEAR(var, 1.0f, 1e-3f); // eps makes it slightly < 1
    }
}

TEST(LayerNorm, FiniteDifferenceGradients)
{
    LayerNormImpl<Backend> ln(kD);
    // Perturb gamma/beta off their defaults so their gradients are non-trivial.
    std::mt19937 rng(4u);
    std::uniform_real_distribution<float> d(0.5f, 1.5f);
    for (nn::Index j = 0; j < kD; ++j)
    {
        ln.gamma.at(0, j) = d(rng);
        ln.beta.at(0, j) = d(rng) - 1.0f;
    }

    Tensor x = make_input(7u);
    const Tensor y = ln.forward(x, true);
    const Tensor dx = ln.backward(y); // dL/dy = y
    const Tensor dg = ln.gamma.grad();
    const Tensor dbt = ln.beta.grad();

    for (nn::Index j = 0; j < kD; ++j)
    {
        const float bg = ln.gamma.at(0, j);
        expect_close(
            dg.at(0, j), num_grad(ln, x, [&](float v) { ln.gamma.at(0, j) = v; }, bg), "dgamma");
        const float bb = ln.beta.at(0, j);
        expect_close(
            dbt.at(0, j), num_grad(ln, x, [&](float v) { ln.beta.at(0, j) = v; }, bb), "dbeta");
    }

    for (nn::Index i = 0; i < kN; ++i)
        for (nn::Index j = 0; j < kD; ++j)
        {
            const float base = x.at(i, j);
            expect_close(
                dx.at(i, j), num_grad(ln, x, [&](float v) { x.at(i, j) = v; }, base), "dx");
        }
}

TEST(LayerNorm, RejectsWrongFeatureCount)
{
    LayerNormImpl<Backend> ln(kD);
    EXPECT_THROW(ln.forward(Tensor::zeros(kN, kD + 1), false), std::invalid_argument);
}

TEST(LayerNorm, BackwardBeforeForwardThrows)
{
    LayerNormImpl<Backend> ln(kD);
    EXPECT_THROW(ln.backward(Tensor::zeros(kN, kD)), std::runtime_error);
}
