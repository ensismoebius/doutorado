// multi_head_attention_gtest.cpp — Correctness for MultiHeadAttentionImpl.
//
// Load-bearing test: central finite-difference gradient check of every Q/K/V/O
// projection parameter and of the input gradient against the analytic backward
// pass. Plus softmax-row property (attention weights sum to 1) and contract guards.

#include <gtest/gtest.h>

#include <cmath>

#include "layers/attention/MultiHeadAttention.hpp"
#include "tensor/Tensor.hpp"

using Backend = nn::Backend;
using Tensor = nn::TensorImpl<Backend>;
using MHA = MultiHeadAttentionImpl<Backend>;

namespace
{

constexpr int kT = 3;
constexpr int kDModel = 4;
constexpr int kHeads = 2;

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
auto num_grad(MHA& mha, const Tensor& x, Mutate&& set, float base) -> float
{
    constexpr float kEps = 1e-3f;
    set(base + kEps);
    mha.reset_state();
    const float lp = quad_loss(mha.forward(x, false));
    set(base - kEps);
    mha.reset_state();
    const float lm = quad_loss(mha.forward(x, false));
    set(base);
    return (lp - lm) / (2.0f * kEps);
}

void expect_close(float analytic, float numeric, const char* what)
{
    const float tol = std::max(3e-2f * std::fabs(analytic), 3e-3f);
    EXPECT_NEAR(analytic, numeric, tol) << what << " a=" << analytic << " n=" << numeric;
}

void check_linear(MHA& mha, const Tensor& x, LinearImpl<Backend>& lin, const char* tag)
{
    const Tensor gw = lin.weight.grad();
    for (nn::Index k = 0; k < static_cast<nn::Index>(lin.weight.size()); ++k)
    {
        const float base = lin.weight.at(k);
        expect_close(gw.at(k), num_grad(mha, x, [&](float v) { lin.weight.at(k) = v; }, base), tag);
    }
    const Tensor gb = lin.bias.grad();
    for (nn::Index k = 0; k < static_cast<nn::Index>(lin.bias.size()); ++k)
    {
        const float base = lin.bias.at(k);
        expect_close(gb.at(k), num_grad(mha, x, [&](float v) { lin.bias.at(k) = v; }, base), tag);
    }
}

} // namespace

TEST(MultiHeadAttention, AttentionWeightsSumToOne)
{
    MHA mha(kDModel, kHeads);
    const Tensor x = make_input(1u);
    mha.reset_state();
    mha.forward(x, true);

    ASSERT_EQ(mha.head_cache_.size(), static_cast<std::size_t>(kHeads));
    for (const auto& c : mha.head_cache_)
        for (nn::Index i = 0; i < c.a.rows(); ++i)
        {
            float s = 0.0f;
            for (nn::Index j = 0; j < c.a.cols(); ++j) s += c.a.at(i, j);
            EXPECT_NEAR(s, 1.0f, 1e-5f);
        }
}

TEST(MultiHeadAttention, FiniteDifferenceGradients)
{
    MHA mha(kDModel, kHeads, /*seed*/ 5u);
    Tensor x = make_input(7u);

    mha.reset_state();
    const Tensor out = mha.forward(x, true);
    const Tensor dx = mha.backward(out); // dL/dout = out

    check_linear(mha, x, *mha.lin_q_, "dWq/dbq");
    check_linear(mha, x, *mha.lin_k_, "dWk/dbk");
    check_linear(mha, x, *mha.lin_v_, "dWv/dbv");
    check_linear(mha, x, *mha.lin_o_, "dWo/dbo");

    for (nn::Index i = 0; i < kT; ++i)
        for (nn::Index j = 0; j < kDModel; ++j)
        {
            const float base = x.at(i, j);
            expect_close(
                dx.at(i, j), num_grad(mha, x, [&](float v) { x.at(i, j) = v; }, base), "dx");
        }
}

TEST(MultiHeadAttention, RejectsBadHeadCount)
{
    EXPECT_THROW(MHA(4, 3), std::invalid_argument);
    EXPECT_THROW(MHA(4, 0), std::invalid_argument);
}

TEST(MultiHeadAttention, ForwardShapeAndContract)
{
    MHA mha(kDModel, kHeads);
    mha.reset_state();
    const Tensor y = mha.forward(Tensor::zeros(kT, kDModel), false);
    EXPECT_EQ(y.rows(), static_cast<nn::Index>(kT));
    EXPECT_EQ(y.cols(), static_cast<nn::Index>(kDModel));
    EXPECT_THROW(mha.forward(Tensor::zeros(kT, kDModel + 1), false), std::invalid_argument);

    MHA fresh(kDModel, kHeads);
    EXPECT_THROW(fresh.backward(Tensor::zeros(kT, kDModel)), std::runtime_error);
}
