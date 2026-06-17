/**
 * @file snn_spike_plotter_gtest.cpp
 * @brief Unit tests for the Lif neuron layer.
 *
 * Tests the LIF dynamics independently from the spike plotter's OpenGL/GLFW/ImGui
 * rendering infrastructure.  No GUI headers are included here.
 *
 * Invariants validated:
 * - Output values in {0,1}
 * - Output shape matches input shape
 * - reset_state() clears v_mem
 * - Neuron fires when driven above threshold
 * - Neuron sub-threshold never fires
 * - Membrane leaks: v_mem decays toward zero with no input
 * - Backward gradient shape matches forward input shape
 */

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include "layers/Layers.hpp"
#include "layers/spiking/ExponentialSurrogate.hpp"
#include "tensor/Tensor.hpp"

// Convenience: build a constant-valued 2D tensor
static nn::Tensor make_const(size_t rows, size_t cols, float val)
{
    nn::Tensor t(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            t.at(i, j) = val;
    return t;
}

// ---- fixture ----

class LifTest : public ::testing::Test
{
   protected:
    static constexpr size_t kBatch = 4;
    static constexpr size_t kFeatures = 8;

    // Standard neuron: dt=1, R=1, C=1, V_th=1
    nn::Lif neuron{1.0F, 1.0F, 1.0F, 1.0F, true, 0.0F,
                   std::make_shared<ExponentialSurrogate>(0.5F)};

    void SetUp() override { neuron.reset_state(); }
};

// ---- output shape ----

TEST_F(LifTest, ForwardOutputShape_MatchesInput)
{
    nn::Tensor input = make_const(kBatch, kFeatures, 0.5F);
    nn::Tensor out = neuron.forward(input);
    EXPECT_EQ(out.rows(), kBatch);
    EXPECT_EQ(out.cols(), kFeatures);
}

// ---- binary output ----

TEST_F(LifTest, ForwardOutput_IsBinary)
{
    // Drive with large positive input to produce spikes
    nn::Tensor input = make_const(kBatch, kFeatures, 2.0F);
    nn::Tensor out = neuron.forward(input);
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j)
        {
            float v = out.at(i, j);
            EXPECT_TRUE(v == 0.0F || v == 1.0F)
                << "Non-binary at (" << i << "," << j << "): " << v;
        }
}

// ---- firing above threshold ----

TEST_F(LifTest, FiresAboveThreshold)
{
    // With dt=1, R=1, C=1: beta = exp(-1) ≈ 0.368.
    // Accumulated v_mem grows to > 1 (threshold) over several steps with input=2.0.
    nn::Tensor input = make_const(1, 1, 2.0F);
    bool fired = false;
    for (int t = 0; t < 10; ++t)
    {
        nn::Tensor out = neuron.forward(input, false);
        if (out.at(0, 0) == 1.0F)
        {
            fired = true;
            break;
        }
    }
    EXPECT_TRUE(fired) << "Neuron never fired with strong constant input";
}

// ---- sub-threshold: no fire ----

TEST_F(LifTest, SubThreshold_NoSpike)
{
    // Very small input → v_mem never reaches threshold
    nn::Tensor input = make_const(1, 1, 0.001F);
    for (int t = 0; t < 50; ++t)
    {
        nn::Tensor out = neuron.forward(input, false);
        EXPECT_FLOAT_EQ(out.at(0, 0), 0.0F) << "Unexpected spike at step " << t;
    }
}

// ---- reset_state clears v_mem ----

TEST_F(LifTest, ResetState_ClearsMemPotential)
{
    // Drive until v_mem is non-zero
    nn::Tensor input = make_const(1, 1, 0.5F);
    for (int t = 0; t < 5; ++t) neuron.forward(input, false);

    neuron.reset_state();

    // After reset v_mem is zero; sub-threshold input for 1 step should give same
    // result as a fresh neuron
    nn::Lif fresh{1.0F, 1.0F, 1.0F, 1.0F, true, 0.0F,
                  std::make_shared<ExponentialSurrogate>(0.5F)};
    nn::Tensor same_input = make_const(1, 1, 0.3F);
    nn::Tensor out_reset = neuron.forward(same_input, false);
    nn::Tensor out_fresh = fresh.forward(same_input, false);

    EXPECT_FLOAT_EQ(out_reset.at(0, 0), out_fresh.at(0, 0));
}

// ---- reset between independent sequences ----

TEST_F(LifTest, ResetBetweenSequences_StateDoesNotLeak)
{
    // Sequence 1: drive strongly
    nn::Tensor strong = make_const(1, 1, 3.0F);
    for (int t = 0; t < 5; ++t) neuron.forward(strong, false);

    neuron.reset_state();

    // Sequence 2: identical to Sequence 1 on a fresh neuron → same output
    nn::Lif fresh{1.0F, 1.0F, 1.0F, 1.0F, true, 0.0F,
                  std::make_shared<ExponentialSurrogate>(0.5F)};
    nn::Tensor check = make_const(1, 1, 0.05F);
    float out_reset = neuron.forward(check, false).at(0, 0);
    float out_fresh = fresh.forward(check, false).at(0, 0);
    EXPECT_FLOAT_EQ(out_reset, out_fresh);
}

// ---- membrane decay with zero input ----

TEST_F(LifTest, MembraneDecays_WithZeroInput)
{
    // Inject charge to just below threshold, then remove input → v_mem must decrease
    nn::Tensor prime = make_const(1, 1, 0.8F); // sub-threshold injection
    neuron.forward(prime, false);
    float v_after_prime = neuron.v_mem.at(0, 0);

    nn::Tensor zero = make_const(1, 1, 0.0F);
    neuron.forward(zero, false);
    float v_after_decay = neuron.v_mem.at(0, 0);

    EXPECT_LT(v_after_decay, v_after_prime) << "v_mem did not decay with zero input";
    EXPECT_GE(v_after_decay, 0.0F) << "v_mem went negative";
}

// ---- backward gradient shape ----

TEST_F(LifTest, BackwardGradient_ShapeMatchesInput)
{
    nn::Tensor input = make_const(kBatch, kFeatures, 0.5F);
    neuron.forward(input, /*requires_grad=*/true);

    // Gradient has same shape as output (ones)
    nn::Tensor grad_out = make_const(kBatch, kFeatures, 1.0F);
    nn::Tensor grad_in = neuron.backward(grad_out);

    EXPECT_EQ(grad_in.rows(), kBatch);
    EXPECT_EQ(grad_in.cols(), kFeatures);
}

// ---- backward gradient finiteness ----

TEST_F(LifTest, BackwardGradient_IsFinite)
{
    nn::Tensor input = make_const(kBatch, kFeatures, 0.5F);
    neuron.forward(input, true);

    nn::Tensor grad_out = make_const(kBatch, kFeatures, 1.0F);
    nn::Tensor grad_in = neuron.backward(grad_out);

    for (size_t i = 0; i < grad_in.rows(); ++i)
        for (size_t j = 0; j < grad_in.cols(); ++j)
            EXPECT_TRUE(std::isfinite(grad_in.at(i, j)))
                << "Non-finite gradient at (" << i << "," << j << ")";
}

// ---- shape adaptation (resize v_mem when input shape changes) ----

TEST_F(LifTest, ShapeChange_ResidesMemPotential)
{
    nn::Tensor small = make_const(1, 4, 0.3F);
    neuron.forward(small, false);

    // Change shape — neuron must not crash, must resize
    nn::Tensor big = make_const(2, 8, 0.3F);
    nn::Tensor out = neuron.forward(big, false);
    EXPECT_EQ(out.rows(), 2u);
    EXPECT_EQ(out.cols(), 8u);
}
