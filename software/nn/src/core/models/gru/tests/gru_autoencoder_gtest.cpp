/**
 * @file gru_autoencoder_gtest.cpp
 * @brief Tests for nn::models::gru::GRUAutoencoder — shape contract, state_dict
 *        round-trip, a composed finite-difference gradient check, and an
 *        overfit-one-batch sanity check that the end-to-end wiring learns.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "models/gru/GRUAutoencoder.hpp"
#include "tensor/Tensor.hpp"

using nn::models::gru::GRUAutoencoder;
using nn::models::gru::GRUAutoencoderConfig;
using Tensor = nn::Tensor;

namespace
{

auto small_cfg() -> GRUAutoencoderConfig
{
    GRUAutoencoderConfig c;
    c.input_size = 3;
    c.hidden_size = 5;
    c.latent_size = 2;
    c.num_layers = 1;
    c.seq_len = 4;
    return c;
}

auto rand_seq(unsigned seed, int T, int D) -> Tensor
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    Tensor x(static_cast<nn::Index>(T), static_cast<nn::Index>(D));
    for (nn::Index t = 0; t < T; ++t)
        for (nn::Index k = 0; k < D; ++k) x.at(t, k) = d(rng);
    return x;
}

auto sse(const Tensor& a, const Tensor& b) -> float
{
    float acc = 0.0f;
    for (nn::Index i = 0; i < a.rows(); ++i)
        for (nn::Index j = 0; j < a.cols(); ++j)
        {
            const float e = a.at(i, j) - b.at(i, j);
            acc += 0.5f * e * e;
        }
    return acc;
}

} // namespace

TEST(GRUAutoencoderConfig, DefaultValues)
{
    GRUAutoencoderConfig c;
    EXPECT_EQ(c.input_size, 64);
    EXPECT_EQ(c.hidden_size, 128);
    EXPECT_EQ(c.latent_size, 16);
    EXPECT_EQ(c.num_layers, 1);
}

TEST(GRUAutoencoder, ForwardBackwardShape2D)
{
    GRUAutoencoder ae(small_cfg());
    const Tensor x = rand_seq(1u, 4, 3);

    ae.reset_state();
    const Tensor recon = ae.forward(x, true);
    EXPECT_EQ(recon.rows(), 4u);
    EXPECT_EQ(recon.cols(), 3u);

    const Tensor dx = ae.backward(recon - x);
    EXPECT_EQ(dx.rows(), 4u);
    EXPECT_EQ(dx.cols(), 3u);
}

TEST(GRUAutoencoder, ForwardShape3DBatch)
{
    GRUAutoencoder ae(small_cfg());
    Tensor x = Tensor::zeros(2, 4, 3);
    ae.reset_state();
    const Tensor recon = ae.forward(x, false);
    ASSERT_EQ(recon.get_shape().size(), 3u);
    EXPECT_EQ(recon.get_shape()[0], 2u);
    EXPECT_EQ(recon.get_shape()[1], 4u);
    EXPECT_EQ(recon.get_shape()[2], 3u);
}

TEST(GRUAutoencoder, StateDictRoundTrip)
{
    GRUAutoencoder a(small_cfg());
    GRUAutoencoder b(small_cfg());
    const Tensor x = rand_seq(2u, 4, 3);

    a.reset_state();
    b.reset_state();
    const Tensor ra = a.forward(x, false);
    const Tensor rb_before = b.forward(x, false);
    // Different init seeds are not used here (both fixed), but the projection
    // weights are seeded identically, so perturb b then restore via load.
    for (Tensor* p : b.params()) p->at(0) += 0.5f;
    b.load_state_dict(a.state_dict());
    b.reset_state();
    const Tensor rb_after = b.forward(x, false);

    for (nn::Index i = 0; i < ra.rows(); ++i)
        for (nn::Index j = 0; j < ra.cols(); ++j)
            EXPECT_NEAR(ra.at(i, j), rb_after.at(i, j), 1e-6f);
    (void) rb_before;
}

TEST(GRUAutoencoder, ComposedFiniteDifferenceOnEncoderProjection)
{
    GRUAutoencoder ae(small_cfg());
    const Tensor x = rand_seq(5u, 4, 3);

    ae.reset_state();
    const Tensor recon = ae.forward(x, true);
    ae.backward(recon - x); // dL/drecon for L = 0.5||recon - x||^2

    Tensor& W = ae.enc_proj_->weight;
    const Tensor gW = W.grad();

    constexpr float kEps = 1e-3f;
    for (nn::Index i = 0; i < W.rows(); ++i)
        for (nn::Index j = 0; j < W.cols(); ++j)
        {
            const float base = W.at(i, j);
            W.at(i, j) = base + kEps;
            ae.reset_state();
            const float lp = sse(ae.forward(x, false), x);
            W.at(i, j) = base - kEps;
            ae.reset_state();
            const float lm = sse(ae.forward(x, false), x);
            W.at(i, j) = base;
            const float num = (lp - lm) / (2.0f * kEps);
            const float tol = std::max(3e-2f * std::fabs(gW.at(i, j)), 3e-3f);
            EXPECT_NEAR(gW.at(i, j), num, tol) << "i=" << i << " j=" << j;
        }
}

TEST(GRUAutoencoder, OverfitsSingleSequence)
{
    GRUAutoencoder ae(small_cfg());

    // A smooth, low-rank target — compressible through the dim-2 tanh bottleneck,
    // so a correctly wired model should drive the reconstruction error well down.
    Tensor x(4, 3);
    for (nn::Index t = 0; t < 4; ++t)
        for (nn::Index k = 0; k < 3; ++k)
            x.at(t, k) = std::sin(0.7f * static_cast<float>(t) + static_cast<float>(k));

    ae.reset_state();
    const float loss0 = sse(ae.forward(x, false), x);

    constexpr float kLr = 0.05f;
    for (int step = 0; step < 400; ++step)
    {
        ae.reset_state();
        const Tensor recon = ae.forward(x, true);
        ae.backward(recon - x);
        for (Tensor* p : ae.params())
        {
            const Tensor g = p->grad();
            for (nn::Index k = 0; k < static_cast<nn::Index>(p->size()); ++k)
                p->at(k) -= kLr * g.at(k);
        }
    }

    ae.reset_state();
    const float loss1 = sse(ae.forward(x, false), x);
    EXPECT_LT(loss1, 0.5f * loss0) << "loss0=" << loss0 << " loss1=" << loss1;
}
