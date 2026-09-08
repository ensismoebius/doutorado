// gru_layer_gtest.cpp — Correctness for GRULayerImpl.
//
// The layer is its own reference, so the load-bearing test is a central
// finite-difference gradient check of every parameter (W, U, b) and of the
// input gradient against the analytic BPTT pass. Plus shape/contract and a
// batch-equals-loop-over-singletons parity check.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layers/gru/GRULayer.hpp"
#include "tensor/Tensor.hpp"

using Backend = nn::Backend;
using Tensor = nn::TensorImpl<Backend>;

namespace
{

constexpr int kD = 3;
constexpr int kH = 4;
constexpr int kT = 5;
constexpr int kB = 2;

// L = 0.5 * sum(out^2)  ⇒  dL/dout = out
auto quad_loss(const Tensor& out) -> float
{
    float acc = 0.0f;
    const auto& s = out.get_shape();
    for (nn::Index b = 0; b < static_cast<nn::Index>(s[0]); ++b)
        for (nn::Index t = 0; t < static_cast<nn::Index>(s[1]); ++t)
            for (nn::Index k = 0; k < static_cast<nn::Index>(s[2]); ++k)
            {
                const float v = out.at(b, t, k);
                acc += 0.5f * v * v;
            }
    return acc;
}

auto make_input(unsigned seed) -> Tensor
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    Tensor x(static_cast<nn::Index>(kB), static_cast<nn::Index>(kT), static_cast<nn::Index>(kD));
    for (nn::Index b = 0; b < kB; ++b)
        for (nn::Index t = 0; t < kT; ++t)
            for (nn::Index k = 0; k < kD; ++k) x.at(b, t, k) = d(rng);
    return x;
}

// Central finite difference of quad_loss wrt one element, via a mutator.
template <typename Mutate>
auto num_grad(GRULayerImpl<Backend>& layer, const Tensor& x, Mutate&& set, float base) -> float
{
    constexpr float kEps = 1e-3f;
    set(base + kEps);
    layer.reset_state();
    const float lp = quad_loss(layer.forward(x, false));
    set(base - kEps);
    layer.reset_state();
    const float lm = quad_loss(layer.forward(x, false));
    set(base); // restore
    return (lp - lm) / (2.0f * kEps);
}

void expect_close(float analytic, float numeric, const char* what)
{
    const float tol = std::max(2e-2f * std::fabs(analytic), 2e-3f);
    EXPECT_NEAR(analytic, numeric, tol)
        << what << " (analytic=" << analytic << " numeric=" << numeric << ")";
}

} // namespace

TEST(GRULayer, FiniteDifferenceGradients)
{
    GRULayerImpl<Backend> layer(kD, kH);
    const Tensor x = make_input(7u);

    layer.reset_state();
    const Tensor out = layer.forward(x, true);
    ASSERT_EQ(out.get_shape().size(), 3u);
    Tensor grad_out = out; // dL/dout for the quadratic loss
    const Tensor dx = layer.backward(grad_out);

    const Tensor dW = layer.W_.grad();
    const Tensor dU = layer.U_.grad();
    const Tensor db = layer.b_.grad();

    for (nn::Index i = 0; i < layer.W_.rows(); ++i)
        for (nn::Index j = 0; j < layer.W_.cols(); ++j)
        {
            const float base = layer.W_.at(i, j);
            const float g = num_grad(layer, x, [&](float v) { layer.W_.at(i, j) = v; }, base);
            expect_close(dW.at(i, j), g, "dW");
        }

    for (nn::Index i = 0; i < layer.U_.rows(); ++i)
        for (nn::Index j = 0; j < layer.U_.cols(); ++j)
        {
            const float base = layer.U_.at(i, j);
            const float g = num_grad(layer, x, [&](float v) { layer.U_.at(i, j) = v; }, base);
            expect_close(dU.at(i, j), g, "dU");
        }

    for (nn::Index i = 0; i < layer.b_.rows(); ++i)
    {
        const float base = layer.b_.at(i, 0);
        const float g = num_grad(layer, x, [&](float v) { layer.b_.at(i, 0) = v; }, base);
        expect_close(db.at(i, 0), g, "db");
    }

    // Input gradient.
    Tensor xm = x;
    for (nn::Index b = 0; b < kB; ++b)
        for (nn::Index t = 0; t < kT; ++t)
            for (nn::Index k = 0; k < kD; ++k)
            {
                const float base = x.at(b, t, k);
                const float g = num_grad(layer, xm, [&](float v) { xm.at(b, t, k) = v; }, base);
                expect_close(dx.at(b, t, k), g, "dx");
            }
}

TEST(GRULayer, ShapeContract2DAnd3D)
{
    GRULayerImpl<Backend> layer(kD, kH);

    layer.reset_state();
    const Tensor o2 = layer.forward(Tensor::zeros(kT, kD), false);
    ASSERT_EQ(o2.get_shape().size(), 2u);
    EXPECT_EQ(o2.get_shape()[0], static_cast<std::size_t>(kT));
    EXPECT_EQ(o2.get_shape()[1], static_cast<std::size_t>(kH));

    layer.reset_state();
    const Tensor o3 = layer.forward(Tensor::zeros(kB, kT, kD), false);
    ASSERT_EQ(o3.get_shape().size(), 3u);
    EXPECT_EQ(o3.get_shape()[0], static_cast<std::size_t>(kB));
    EXPECT_EQ(o3.get_shape()[2], static_cast<std::size_t>(kH));
}

TEST(GRULayer, RejectsWrongInputDim)
{
    GRULayerImpl<Backend> layer(kD, kH);
    EXPECT_THROW(layer.forward(Tensor::zeros(kT, kD + 1), false), std::invalid_argument);
}

TEST(GRULayer, BackwardBeforeForwardThrows)
{
    GRULayerImpl<Backend> layer(kD, kH);
    EXPECT_THROW(layer.backward(Tensor::zeros(kT, kH)), std::runtime_error);
}

TEST(GRULayer, ResetStateClearsHidden)
{
    GRULayerImpl<Backend> layer(kD, kH);
    const Tensor x = make_input(3u);

    layer.reset_state();
    const Tensor a = layer.forward(x.slice_batch(0), false); // 2D path, mutates h0_
    layer.reset_state();
    const Tensor b = layer.forward(x.slice_batch(0), false);
    for (nn::Index t = 0; t < kT; ++t)
        for (nn::Index k = 0; k < kH; ++k) EXPECT_FLOAT_EQ(a.at(t, k), b.at(t, k));
}

TEST(GRULayer, BatchEqualsLoopOverSingletons)
{
    GRULayerImpl<Backend> layer(kD, kH);
    const Tensor x = make_input(11u);

    layer.reset_state();
    const Tensor batched = layer.forward(x, false); // (B, T, H)

    for (nn::Index b = 0; b < kB; ++b)
    {
        layer.reset_state();
        const Tensor single = layer.forward(x.slice_batch(b), false); // (T, H)
        for (nn::Index t = 0; t < kT; ++t)
            for (nn::Index k = 0; k < kH; ++k)
                EXPECT_NEAR(batched.at(b, t, k), single.at(t, k), 1e-5f)
                    << "b=" << b << " t=" << t << " k=" << k;
    }
}
