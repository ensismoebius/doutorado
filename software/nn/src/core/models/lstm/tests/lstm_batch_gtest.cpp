/**
 * @file lstm_batch_gtest.cpp
 * @brief Tests for LSTMLayer batch support (forward_batch / backward_batch).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "nn/layers/lstm/LSTMLayer.hpp"
#include "nn/tensor/Tensor.hpp"

namespace
{

constexpr int D = 4;  // input features
constexpr int H = 8;  // hidden size
constexpr int T = 3;  // sequence length

// Produce a deterministic (T, D) input tensor
static nn::Tensor make_sample(float fill_val)
{
    nn::Tensor s(T, D);
    for (nn::Index i = 0; i < static_cast<nn::Index>(T); ++i)
        for (nn::Index j = 0; j < static_cast<nn::Index>(D); ++j)
            s.at(i, j) = fill_val + static_cast<float>(i * D + j) * 0.01F;
    return s;
}

static nn::Tensor ones_grad()
{
    nn::Tensor g(T, H);
    for (nn::Index k = 0; k < static_cast<nn::Index>(g.size()); ++k)
        g.at(k) = 1.0F;
    return g;
}

// ---- Tests ----

TEST(LSTMBatchTest, BatchSizeOne_OutputMatchesSingle)
{
    nn::models::lstm::LSTMLayer layer(D, H);
    nn::Tensor sample = make_sample(0.5F);

    // Single forward
    layer.reset_state();
    nn::Tensor out_single = layer.forward(sample, false);

    // Batch of one
    layer.reset_state();
    auto outputs = layer.forward_batch({sample}, false);

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].rows(), out_single.rows());
    ASSERT_EQ(outputs[0].cols(), out_single.cols());

    for (nn::Index k = 0; k < static_cast<nn::Index>(out_single.size()); ++k)
        EXPECT_NEAR(outputs[0].at(k), out_single.at(k), 1e-5F);
}

TEST(LSTMBatchTest, OutputShape)
{
    nn::models::lstm::LSTMLayer layer(D, H);
    nn::Tensor s1 = make_sample(0.1F);
    nn::Tensor s2 = make_sample(0.2F);

    auto outputs = layer.forward_batch({s1, s2}, false);

    ASSERT_EQ(outputs.size(), 2u);
    for (const auto& out : outputs)
    {
        EXPECT_EQ(out.rows(), static_cast<std::size_t>(T));
        EXPECT_EQ(out.cols(), static_cast<std::size_t>(H));
    }
}

TEST(LSTMBatchTest, GradientAccumulationNonzero)
{
    nn::models::lstm::LSTMLayer layer(D, H);
    nn::Tensor s = make_sample(0.3F);

    layer.forward_batch({s, s}, true);
    nn::Tensor g = ones_grad();
    layer.backward_batch({g, g});

    // W, U, b grads must be nonzero
    float dW_norm = 0.0F, dU_norm = 0.0F;
    for (nn::Index k = 0; k < static_cast<nn::Index>(layer.W_.grad().size()); ++k)
        dW_norm += layer.W_.grad().at(k) * layer.W_.grad().at(k);
    for (nn::Index k = 0; k < static_cast<nn::Index>(layer.U_.grad().size()); ++k)
        dU_norm += layer.U_.grad().at(k) * layer.U_.grad().at(k);

    EXPECT_GT(std::sqrt(dW_norm), 1e-6F);
    EXPECT_GT(std::sqrt(dU_norm), 1e-6F);
}

TEST(LSTMBatchTest, BatchEquivalence_GradsAreTwoTimeSingle)
{
    // Reference: single sample forward + backward
    nn::models::lstm::LSTMLayer ref_layer(D, H);
    nn::Tensor sample = make_sample(0.2F);
    nn::Tensor g      = ones_grad();

    ref_layer.reset_state();
    ref_layer.forward(sample, true);
    ref_layer.backward(g);

    // Collect single-sample W grad
    nn::Tensor ref_dW = ref_layer.W_.grad();
    nn::Tensor ref_dU = ref_layer.U_.grad();

    // Batch of 2 identical samples
    nn::models::lstm::LSTMLayer batch_layer(D, H);
    // Copy same weights so results are comparable
    batch_layer.W_ = ref_layer.W_;
    batch_layer.U_ = ref_layer.U_;
    batch_layer.b_ = ref_layer.b_;

    batch_layer.forward_batch({sample, sample}, true);
    batch_layer.backward_batch({g, g});

    nn::Tensor batch_dW = batch_layer.W_.grad();
    nn::Tensor batch_dU = batch_layer.U_.grad();

    for (nn::Index k = 0; k < static_cast<nn::Index>(ref_dW.size()); ++k)
        EXPECT_NEAR(batch_dW.at(k), 2.0F * ref_dW.at(k), 1e-4F)
            << "dW mismatch at k=" << k;

    for (nn::Index k = 0; k < static_cast<nn::Index>(ref_dU.size()); ++k)
        EXPECT_NEAR(batch_dU.at(k), 2.0F * ref_dU.at(k), 1e-4F)
            << "dU mismatch at k=" << k;
}

TEST(LSTMBatchTest, BackwardBatchSizeMismatch_Throws)
{
    nn::models::lstm::LSTMLayer layer(D, H);
    nn::Tensor s = make_sample(0.1F);

    layer.forward_batch({s, s}, true);

    EXPECT_THROW(layer.backward_batch({ones_grad()}), std::runtime_error);
}

} // namespace
