/**
 * @file lstm_batch_gtest.cpp
 * @brief Tests for LSTMLayer batch support via 3D forward/backward (B, T, D).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layers/lstm/LSTMLayer.hpp"
#include "tensor/Tensor.hpp"

namespace
{

constexpr int D = 4; // input features
constexpr int H = 8; // hidden size
constexpr int T = 3; // sequence length

static nn::Tensor make_sample(float fill_val)
{
    nn::Tensor s(T, D);
    for (nn::Index i = 0; i < static_cast<nn::Index>(T); ++i)
        for (nn::Index j = 0; j < static_cast<nn::Index>(D); ++j)
            s.at(i, j) = fill_val + static_cast<float>(i * D + j) * 0.01F;
    return s;
}

// Stack B independent (T, D) tensors into a (B, T, D) tensor.
static nn::Tensor stack_samples(const std::vector<nn::Tensor>& samples)
{
    const int B = static_cast<int>(samples.size());
    nn::Tensor out = nn::Tensor::zeros(B, T, D);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D; ++d)
                out.at(static_cast<nn::Index>(b),
                    static_cast<nn::Index>(t),
                    static_cast<nn::Index>(d)) = samples[b].at(t, d);
    return out;
}

static nn::Tensor ones_grad_2d()
{
    nn::Tensor g(T, H);
    for (nn::Index k = 0; k < static_cast<nn::Index>(g.size()); ++k) g.at(k) = 1.0F;
    return g;
}

static nn::Tensor ones_grad_3d(int B)
{
    nn::Tensor g = nn::Tensor::zeros(B, T, H);
    for (nn::Index k = 0; k < static_cast<nn::Index>(g.size()); ++k) g.at(k) = 1.0F;
    return g;
}

// ---- Tests ----

TEST(LSTMBatchTest, BatchSizeOne_OutputMatchesSingle)
{
    nn::models::lstm::LSTMLayer layer(D, H);
    nn::Tensor sample = make_sample(0.5F);

    // Single 2D forward
    layer.reset_state();
    nn::Tensor out_single = layer.forward(sample, false);

    // 3D forward with B=1
    layer.reset_state();
    nn::Tensor out_3d = layer.forward(stack_samples({sample}), false);

    // out_3d shape: (1, T, H) — compare [0, t, h] vs [t, h]
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            EXPECT_NEAR(out_3d.at(0, t, h), out_single.at(t, h), 1e-5F)
                << "mismatch at t=" << t << " h=" << h;
}

TEST(LSTMBatchTest, OutputShape)
{
    nn::models::lstm::LSTMLayer layer(D, H);
    nn::Tensor batched = stack_samples({make_sample(0.1F), make_sample(0.2F)});

    nn::Tensor out = layer.forward(batched, false);
    const auto& shape = out.get_shape();

    ASSERT_EQ(shape.size(), 3u);
    EXPECT_EQ(static_cast<int>(shape[0]), 2);
    EXPECT_EQ(static_cast<int>(shape[1]), T);
    EXPECT_EQ(static_cast<int>(shape[2]), H);
}

TEST(LSTMBatchTest, GradientNormScalesWithBatchSize)
{
    nn::models::lstm::LSTMLayer layer(D, H);
    nn::Tensor s = make_sample(0.3F);

    // Reference gradients for a single sample.
    layer.reset_state();
    layer.forward(s, true);
    layer.backward(ones_grad_2d());

    float dW_single_norm = 0.0F, dU_single_norm = 0.0F;
    for (nn::Index k = 0; k < static_cast<nn::Index>(layer.W_.grad().size()); ++k)
        dW_single_norm += layer.W_.grad().at(k) * layer.W_.grad().at(k);
    for (nn::Index k = 0; k < static_cast<nn::Index>(layer.U_.grad().size()); ++k)
        dU_single_norm += layer.U_.grad().at(k) * layer.U_.grad().at(k);

    // Batch with duplicated samples should scale gradients by factor 2.
    layer.reset_state();
    layer.forward(stack_samples({s, s}), true);
    layer.backward(ones_grad_3d(2));

    float dW_norm = 0.0F, dU_norm = 0.0F;
    for (nn::Index k = 0; k < static_cast<nn::Index>(layer.W_.grad().size()); ++k)
        dW_norm += layer.W_.grad().at(k) * layer.W_.grad().at(k);
    for (nn::Index k = 0; k < static_cast<nn::Index>(layer.U_.grad().size()); ++k)
        dU_norm += layer.U_.grad().at(k) * layer.U_.grad().at(k);

    EXPECT_NEAR(std::sqrt(dW_norm), 2.0F * std::sqrt(dW_single_norm), 1e-4F);
    EXPECT_NEAR(std::sqrt(dU_norm), 2.0F * std::sqrt(dU_single_norm), 1e-4F);
}

TEST(LSTMBatchTest, BatchEquivalence_GradsAreTwoTimeSingle)
{
    nn::models::lstm::LSTMLayer ref_layer(D, H);
    nn::Tensor sample = make_sample(0.2F);
    nn::Tensor g2d = ones_grad_2d();

    ref_layer.reset_state();
    ref_layer.forward(sample, true);
    ref_layer.backward(g2d);

    nn::Tensor ref_dW = ref_layer.W_.grad();
    nn::Tensor ref_dU = ref_layer.U_.grad();

    nn::models::lstm::LSTMLayer batch_layer(D, H);
    batch_layer.W_ = ref_layer.W_;
    batch_layer.U_ = ref_layer.U_;
    batch_layer.b_ = ref_layer.b_;

    batch_layer.forward(stack_samples({sample, sample}), true);
    batch_layer.backward(ones_grad_3d(2));

    nn::Tensor batch_dW = batch_layer.W_.grad();
    nn::Tensor batch_dU = batch_layer.U_.grad();

    for (nn::Index k = 0; k < static_cast<nn::Index>(ref_dW.size()); ++k)
        EXPECT_NEAR(batch_dW.at(k), 2.0F * ref_dW.at(k), 1e-4F) << "dW mismatch at k=" << k;

    for (nn::Index k = 0; k < static_cast<nn::Index>(ref_dU.size()); ++k)
        EXPECT_NEAR(batch_dU.at(k), 2.0F * ref_dU.at(k), 1e-4F) << "dU mismatch at k=" << k;
}

} // namespace
