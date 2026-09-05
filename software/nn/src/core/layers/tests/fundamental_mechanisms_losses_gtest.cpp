/**
 * @file fundamental_mechanisms_losses_gtest.cpp
 * @brief Fundamental-mechanism correctness for CrossEntropyLoss, MAELoss, and SpikeTimeLoss.
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
