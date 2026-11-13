/**
 * @file transformer_autoencoder_gtest.cpp
 * @brief Tests for the bottlenecked Transformer autoencoder: shape contract,
 *        state_dict round-trip, a composed finite-difference gradient check on
 *        representative parameters, and overfit-one-sequence.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "core/models/tests/seeded_init_checks.hpp"
#include "models/transformer/TransformerAutoencoder.hpp"
#include "tensor/Tensor.hpp"

using nn::models::transformer::TransformerAutoencoder;
using nn::models::transformer::TransformerAutoencoderConfig;
using Tensor = nn::Tensor;

namespace
{

auto small_cfg() -> TransformerAutoencoderConfig
{
    TransformerAutoencoderConfig c;
    c.input_size = 3;
    c.seq_len = 4;
    c.d_model = 4;
    c.n_heads = 2;
    c.n_layers = 1;
    c.d_ff = 6;
    c.latent_size = 2;
    return c;
}

auto smooth_target(int T, int D) -> Tensor
{
    Tensor x(static_cast<nn::Index>(T), static_cast<nn::Index>(D));
    for (nn::Index t = 0; t < T; ++t)
        for (nn::Index k = 0; k < D; ++k)
            x.at(t, k) = std::sin(0.6f * static_cast<float>(t) + 0.9f * static_cast<float>(k));
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

TEST(TransformerAutoencoder, ForwardBackwardShape)
{
    TransformerAutoencoder ae(small_cfg());
    const Tensor x = smooth_target(4, 3);

    ae.reset_state();
    const Tensor recon = ae.forward(x, true);
    EXPECT_EQ(recon.rows(), 4u);
    EXPECT_EQ(recon.cols(), 3u);

    const Tensor dx = ae.backward(recon - x);
    EXPECT_EQ(dx.rows(), 4u);
    EXPECT_EQ(dx.cols(), 3u);
}

TEST(TransformerAutoencoder, RejectsBadConfig)
{
    auto c = small_cfg();
    c.n_heads = 3; // does not divide d_model=4
    EXPECT_THROW(
        { [[maybe_unused]] const TransformerAutoencoder object{c}; }, std::invalid_argument);
}

TEST(TransformerAutoencoder, StateDictRoundTrip)
{
    TransformerAutoencoder a(small_cfg());
    TransformerAutoencoder b(small_cfg());
    const Tensor x = smooth_target(4, 3);

    a.reset_state();
    const Tensor ra = a.forward(x, false);

    for (Tensor* p : b.params()) p->at(0) += 0.3f;
    b.load_state_dict(a.state_dict());
    b.reset_state();
    const Tensor rb = b.forward(x, false);

    for (nn::Index i = 0; i < ra.rows(); ++i)
        for (nn::Index j = 0; j < ra.cols(); ++j) EXPECT_NEAR(ra.at(i, j), rb.at(i, j), 1e-6f);
}

TEST(TransformerAutoencoder, ComposedFiniteDifference)
{
    TransformerAutoencoder ae(small_cfg());
    Tensor x = smooth_target(4, 3);

    ae.reset_state();
    const Tensor recon = ae.forward(x, true);
    ae.backward(recon - x); // dL/drecon for L = 0.5||recon - x||^2

    constexpr float kEps = 1e-3f;
    auto check = [&](Tensor& p, const char* tag)
    {
        const Tensor g = p.grad();
        for (nn::Index k = 0; k < static_cast<nn::Index>(p.size()); ++k)
        {
            const float base = p.at(k);
            p.at(k) = base + kEps;
            ae.reset_state();
            const float lp = sse(ae.forward(x, false), x);
            p.at(k) = base - kEps;
            ae.reset_state();
            const float lm = sse(ae.forward(x, false), x);
            p.at(k) = base;
            const float num = (lp - lm) / (2.0f * kEps);
            const float tol = std::max(5e-2f * std::fabs(g.at(k)), 5e-3f);
            EXPECT_NEAR(g.at(k), num, tol) << tag << " k=" << k;
        }
    };

    check(ae.embed_->weight, "embed.W");
    check(ae.to_latent_->weight, "to_latent.W");
    check(ae.from_latent_->bias, "from_latent.b");
    check(ae.out_proj_->weight, "out_proj.W");
    check(ae.enc_blocks_[0]->ff1_->weight, "enc0.ff1.W");
    check(ae.dec_blocks_[0]->mha_->lin_v_->weight, "dec0.mha.Wv");
}

// Pre-norm blocks plus the two final LayerNorms (Xiong et al. 2020): every new parameter is
// in params() and the state dict, and the composed gradient is right through all of them.
TEST(TransformerAutoencoder, PreNormAddsTwoFinalNormsAndHasCorrectGradients)
{
    auto c = small_cfg();
    TransformerAutoencoder post(c);
    c.norm_first = true;
    TransformerAutoencoder ae(c);
    ASSERT_TRUE(ae.enc_final_ln_ && ae.dec_final_ln_);
    EXPECT_FALSE(post.enc_final_ln_ || post.dec_final_ln_);
    EXPECT_EQ(ae.params().size(), post.params().size() + 4u); // gamma, beta per final norm
    EXPECT_EQ(ae.state_dict().count("enc_ln.gamma") + ae.state_dict().count("dec_ln.beta"), 2u);

    Tensor x = smooth_target(4, 3);
    // Non-trivial final-norm scales, so their gradients are not the identity's.
    for (nn::Index j = 0; j < 4; ++j)
    {
        ae.enc_final_ln_->gamma.at(0, j) = 0.8f + 0.1f * static_cast<float>(j);
        ae.dec_final_ln_->gamma.at(0, j) = 1.2f - 0.1f * static_cast<float>(j);
    }
    ae.reset_state();
    const Tensor recon = ae.forward(x, true);
    ae.backward(recon - x);

    constexpr float kEps = 1e-3f;
    auto check = [&](Tensor& p, const char* tag)
    {
        const Tensor g = p.grad();
        for (nn::Index k = 0; k < static_cast<nn::Index>(p.size()); ++k)
        {
            const float base = p.at(k);
            p.at(k) = base + kEps;
            ae.reset_state();
            const float lp = sse(ae.forward(x, false), x);
            p.at(k) = base - kEps;
            ae.reset_state();
            const float lm = sse(ae.forward(x, false), x);
            p.at(k) = base;
            const float num = (lp - lm) / (2.0f * kEps);
            const float tol = std::max(5e-2f * std::fabs(g.at(k)), 5e-3f);
            EXPECT_NEAR(g.at(k), num, tol) << tag << " k=" << k;
        }
    };
    check(ae.embed_->weight, "pre embed.W");
    check(ae.enc_blocks_[0]->ln1_->gamma, "pre enc0.ln1.gamma");
    check(ae.enc_final_ln_->gamma, "pre enc_ln.gamma");
    check(ae.enc_final_ln_->beta, "pre enc_ln.beta");
    check(ae.to_latent_->weight, "pre to_latent.W");
    check(ae.dec_blocks_[0]->ff2_->bias, "pre dec0.ff2.b");
    check(ae.dec_final_ln_->gamma, "pre dec_ln.gamma");
    check(ae.out_proj_->weight, "pre out_proj.W");

    // The state dict carries the final norms: a second pre-norm model reproduces the output.
    TransformerAutoencoder b(c);
    b.load_state_dict(ae.state_dict());
    ae.reset_state();
    b.reset_state();
    const Tensor ra = ae.forward(x, false);
    const Tensor rb = b.forward(x, false);
    for (nn::Index k = 0; k < static_cast<nn::Index>(ra.size()); ++k)
        EXPECT_NEAR(ra.at(k), rb.at(k), 1e-6f);
}

TEST(TransformerAutoencoder, OverfitsSingleSequence)
{
    TransformerAutoencoder ae(small_cfg());
    const Tensor x = smooth_target(4, 3);

    ae.reset_state();
    const float loss0 = sse(ae.forward(x, false), x);

    constexpr float kLr = 0.02f;
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
    EXPECT_LT(loss1, 0.6f * loss0) << "loss0=" << loss0 << " loss1=" << loss1;
}

// init_seed (2026-11-04): unset keeps the fixed weights (embed_ from mt19937(4100), the blocks
// from 200 + l / 300 + l); set, every random tensor of every block is drawn from the seed, in
// the post-norm and the pre-norm wiring alike.
TEST(TransformerAutoencoderInitSeed, UnsetKeepsTheFixedWeightsAndASeedReachesEveryTensor)
{
    auto c = small_cfg();
    c.n_layers = 2;
    nn::test::expect_init_seed_contract<TransformerAutoencoder>(c);
    c.norm_first = true;
    nn::test::expect_init_seed_contract<TransformerAutoencoder>(c);

    c.norm_first = false;
    TransformerAutoencoder fixed(c);
    std::mt19937 rng(4100u);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    EXPECT_EQ(fixed.embed_->weight.at(0), dist(rng));
}
