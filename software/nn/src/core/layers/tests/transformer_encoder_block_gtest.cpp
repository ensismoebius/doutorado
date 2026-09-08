// transformer_encoder_block_gtest.cpp — Composed correctness for TransformerEncoderBlockImpl.
//
// Every sub-module (MHA, LayerNorm, Linear, ReLU) has its own gradient check; this
// file verifies the block WIRES them correctly via a central finite-difference
// check of representative parameters (one from each sub-module) and of the input
// gradient, plus shape/contract guards.

#include <gtest/gtest.h>

#include <cmath>

#include "layers/attention/TransformerEncoderBlock.hpp"
#include "tensor/Tensor.hpp"

using Backend = nn::Backend;
using Tensor = nn::TensorImpl<Backend>;
using Block = TransformerEncoderBlockImpl<Backend>;

namespace
{

constexpr int kT = 3;
constexpr int kDModel = 4;
constexpr int kHeads = 2;
constexpr int kDFF = 6;

auto make_input(unsigned seed) -> Tensor
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    Tensor x(kT, kDModel);
    for (nn::Index i = 0; i < kT; ++i)
        for (nn::Index j = 0; j < kDModel; ++j) x.at(i, j) = d(rng);
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
auto num_grad(Block& blk, const Tensor& x, Mutate&& set, float base) -> float
{
    constexpr float kEps = 1e-3f;
    set(base + kEps);
    blk.reset_state();
    const float lp = quad_loss(blk.forward(x, false));
    set(base - kEps);
    blk.reset_state();
    const float lm = quad_loss(blk.forward(x, false));
    set(base);
    return (lp - lm) / (2.0f * kEps);
}

void expect_close(float a, float n, const char* what)
{
    const float tol = std::max(4e-2f * std::fabs(a), 4e-3f);
    EXPECT_NEAR(a, n, tol) << what << " a=" << a << " n=" << n;
}

void check_tensor(Block& blk, const Tensor& x, Tensor& p, const char* tag)
{
    const Tensor g = p.grad();
    for (nn::Index k = 0; k < static_cast<nn::Index>(p.size()); ++k)
    {
        const float base = p.at(k);
        expect_close(g.at(k), num_grad(blk, x, [&](float v) { p.at(k) = v; }, base), tag);
    }
}

} // namespace

TEST(TransformerEncoderBlock, FiniteDifferenceGradientsAcrossSubmodules)
{
    Block blk(kDModel, kHeads, kDFF, /*seed*/ 3u);
    Tensor x = make_input(11u);

    // Perturb the LayerNorm scales so their grads are non-trivial.
    std::mt19937 rng(1u);
    std::uniform_real_distribution<float> d(0.7f, 1.3f);
    for (nn::Index j = 0; j < kDModel; ++j)
    {
        blk.ln1_->gamma.at(0, j) = d(rng);
        blk.ln2_->gamma.at(0, j) = d(rng);
    }

    blk.reset_state();
    const Tensor out = blk.forward(x, true);
    const Tensor dx = blk.backward(out); // dL/dout = out

    check_tensor(blk, x, blk.mha_->lin_q_->weight, "mha.Wq");
    check_tensor(blk, x, blk.mha_->lin_o_->bias, "mha.bo");
    check_tensor(blk, x, blk.ff1_->weight, "ff1.W");
    check_tensor(blk, x, blk.ff2_->bias, "ff2.b");
    check_tensor(blk, x, blk.ln1_->gamma, "ln1.gamma");
    check_tensor(blk, x, blk.ln2_->beta, "ln2.beta");

    for (nn::Index i = 0; i < kT; ++i)
        for (nn::Index j = 0; j < kDModel; ++j)
        {
            const float base = x.at(i, j);
            expect_close(
                dx.at(i, j), num_grad(blk, x, [&](float v) { x.at(i, j) = v; }, base), "dx");
        }
}

TEST(TransformerEncoderBlock, ShapeAndContract)
{
    Block blk(kDModel, kHeads, kDFF);
    blk.reset_state();
    const Tensor y = blk.forward(Tensor::zeros(kT, kDModel), false);
    EXPECT_EQ(y.rows(), static_cast<nn::Index>(kT));
    EXPECT_EQ(y.cols(), static_cast<nn::Index>(kDModel));
    EXPECT_THROW(blk.forward(Tensor::zeros(kT, kDModel + 1), false), std::invalid_argument);

    Block fresh(kDModel, kHeads, kDFF);
    EXPECT_THROW(fresh.backward(Tensor::zeros(kT, kDModel)), std::runtime_error);
}

TEST(TransformerEncoderBlock, RejectsBadDims)
{
    EXPECT_THROW(Block(0, kHeads, kDFF), std::invalid_argument);
    EXPECT_THROW(Block(kDModel, kHeads, 0), std::invalid_argument);
    EXPECT_THROW(Block(kDModel, 3, kDFF), std::invalid_argument); // 3 does not divide 4
}
