/**
 * @file spiking_mechanisms_gtest.cpp
 * @brief Focused exact-oracle coverage for spiking-specific mechanisms.
 */

#include <cmath>
#include <memory>
#include <stdexcept>

#include "gtest/gtest.h"
#include "layers/Layers.hpp"
#include "layers/spiking/BoxcarSurrogate.hpp"
#include "layers/spiking/ExponentialSurrogate.hpp"

namespace
{
constexpr float kTol = 1e-4F;

auto make_const(size_t rows, size_t cols, float value) -> nn::Tensor
{
    nn::Tensor tensor(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) tensor.at(i, j) = value;
    return tensor;
}

auto make_leaky(float resistance = 1.0F,
    float capacitance = 1.0F,
    float voltage_threshold = 1.0F,
    bool reset_zero = true,
    float adapt_decay = 0.9F,
    float adapt_coupling = 0.0F) -> nn::Leaky
{
    return nn::Leaky(1.0F,
        resistance,
        capacitance,
        voltage_threshold,
        reset_zero,
        0.0F,
        std::make_shared<ExponentialSurrogate>(1.0F),
        adapt_decay,
        adapt_coupling);
}

auto make_bptt(int time_steps,
    float voltage_threshold = 1.0F,
    bool reset_zero = true,
    bool readout_mode = false,
    float adapt_decay = 0.9F,
    float adapt_coupling = 0.0F) -> nn::LeakyBPTT
{
    return nn::LeakyBPTT(time_steps,
        1.0F,
        1.0F,
        1.0F,
        voltage_threshold,
        reset_zero,
        0.0F,
        readout_mode,
        std::make_shared<ExponentialSurrogate>(1.0F),
        adapt_decay,
        adapt_coupling);
}
} // namespace

TEST(LeakySpikingMechanismTest, SubThresholdInputNeverSpikes)
{
    auto layer = make_leaky(1.0F, 1.0F, 5.0F);
    for (int step = 0; step < 6; ++step)
        EXPECT_EQ(layer.forward(make_const(1, 1, 0.3F), false).at(0, 0), 0.0F);
}

TEST(LeakySpikingMechanismTest, ExactThresholdBoundaryNoSpike)
{
    auto layer = make_leaky(1.0e6F, 1.0F, 1.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.0F), false).at(0, 0), 0.0F);
}

TEST(LeakySpikingMechanismTest, TemporalIntegrationMultiStep)
{
    auto layer = make_leaky(1.0e6F, 1.0F, 1.0F);
    const nn::Tensor input = make_const(1, 1, 0.3F);
    EXPECT_EQ(layer.forward(input, false).at(0, 0), 0.0F);
    EXPECT_EQ(layer.forward(input, false).at(0, 0), 0.0F);
    EXPECT_EQ(layer.forward(input, false).at(0, 0), 0.0F);
    EXPECT_EQ(layer.forward(input, false).at(0, 0), 1.0F);
}

TEST(LeakySpikingMechanismTest, HardResetAfterSpike)
{
    auto layer = make_leaky();
    EXPECT_EQ(layer.forward(make_const(1, 1, 2.0F), false).at(0, 0), 1.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 0.0F), false).at(0, 0), 0.0F);
}

TEST(LeakySpikingMechanismTest, SoftResetAfterSpike)
{
    auto layer = make_leaky(1.0e6F, 1.0F, 1.0F, false);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.5F), false).at(0, 0), 1.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 0.0F), false).at(0, 0), 0.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 0.6F), false).at(0, 0), 1.0F);
}

TEST(LeakySpikingMechanismTest, BatchIndependence)
{
    auto layer = make_leaky();
    nn::Tensor input(2, 1);
    input.at(0, 0) = 2.0F;
    input.at(1, 0) = 0.1F;
    const nn::Tensor output = layer.forward(input, false);
    EXPECT_EQ(output.at(0, 0), 1.0F);
    EXPECT_EQ(output.at(1, 0), 0.0F);
}

TEST(LeakySpikingMechanismTest, AdaptationRaisesEffectiveThreshold)
{
    auto layer = make_leaky(1.0e6F, 1.0F, 1.0F, true, 0.9F, 0.5F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.2F), false).at(0, 0), 1.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.2F), false).at(0, 0), 0.0F);
}

TEST(LeakySpikingMechanismTest, AdaptationDecaysOverTime)
{
    auto layer = make_leaky(1.0e6F, 1.0F, 1.0F, true, 0.9F, 0.5F);
    layer.forward(make_const(1, 1, 1.2F), false);
    for (int i = 0; i < 10; ++i) layer.forward(make_const(1, 1, 0.0F), false);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.18F), false).at(0, 0), 1.0F);
}

TEST(LeakySpikingMechanismTest, ResetStateClearsMembraneAndAdaptation)
{
    auto layer = make_leaky(1.0e6F, 1.0F, 1.0F, true, 0.9F, 0.5F);
    layer.forward(make_const(1, 1, 1.2F), false);
    layer.reset_state();
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.1F), false).at(0, 0), 1.0F);
}

TEST(LeakySpikingMechanismTest, StateDictRoundTrip)
{
    nn::Leaky source(1.0F,
        2.5F,
        3.7F,
        0.8F,
        true,
        0.0F,
        std::make_shared<ExponentialSurrogate>(1.0F),
        0.9F,
        0.0F);
    nn::Leaky loaded(1.0F,
        1.0F,
        1.0F,
        1.0F,
        true,
        0.0F,
        std::make_shared<ExponentialSurrogate>(1.0F),
        0.9F,
        0.0F);
    loaded.load_state_dict(source.state_dict());
    const nn::Tensor input = make_const(1, 1, 1.5F);
    EXPECT_NEAR(source.forward(input, false).at(0, 0), loaded.forward(input, false).at(0, 0), kTol);
}

TEST(LeakySpikingMechanismTest, ExactBetaDecaySpikeTiming)
{
    auto layer = make_leaky(1.0F, 2.0F, 1.5F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.0F), false).at(0, 0), 0.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.0F), false).at(0, 0), 1.0F);
}

TEST(LeakyIntegratorSpikingMechanismTest, NeverProducesSpike)
{
    nn::LeakyIntegrator layer(1.0F, 1.0F, 1.0F);
    for (int i = 0; i < 10; ++i)
    {
        const float value = layer.forward(make_const(1, 1, 10.0F), false).at(0, 0);
        EXPECT_NE(value, 1.0F);
        EXPECT_GT(value, 0.0F);
    }
}

TEST(LeakyIntegratorSpikingMechanismTest, ExactExponentialDecay)
{
    nn::LeakyIntegrator layer(1.0F, 1.0F, 1.0F);
    const float beta = std::exp(-1.0F);
    EXPECT_NEAR(layer.forward(make_const(1, 1, 2.0F), false).at(0, 0), 2.0F, kTol);
    EXPECT_NEAR(layer.forward(make_const(1, 1, 3.0F), false).at(0, 0), 2.0F * beta + 3.0F, kTol);
}

TEST(LeakyIntegratorSpikingMechanismTest, ResetStateClearsIntegration)
{
    nn::LeakyIntegrator layer(1.0F, 1.0F, 1.0F);
    for (int i = 0; i < 4; ++i) layer.forward(make_const(1, 1, 1.0F), false);
    layer.reset_state();
    EXPECT_NEAR(layer.forward(make_const(1, 1, 0.5F), false).at(0, 0), 0.5F, kTol);
}

TEST(LeakyBPTTSpikingMechanismTest, ThrowsOnInvalidInputShape)
{
    auto layer = make_bptt(3);
    nn::Tensor bad(5, 1);
    bad.setZero();
    EXPECT_THROW(layer.forward(bad, false), std::invalid_argument);
}

TEST(LeakyBPTTSpikingMechanismTest, ExactSpikeOutputKnownSequence)
{
    auto layer = make_bptt(2, 0.5F);
    nn::Tensor input(2, 1);
    input.at(0, 0) = 1.0F;
    input.at(1, 0) = 0.1F;
    const nn::Tensor output = layer.forward(input, false);
    EXPECT_EQ(output.at(0, 0), 1.0F);
    EXPECT_EQ(output.at(1, 0), 0.0F);
}

TEST(LeakyBPTTSpikingMechanismTest, HardResetExactSequence)
{
    auto layer = make_bptt(2, 0.5F, true);
    nn::Tensor input(2, 1);
    input.at(0, 0) = 1.0F;
    input.at(1, 0) = 0.4F;
    const nn::Tensor output = layer.forward(input, false);
    EXPECT_EQ(output.at(0, 0), 1.0F);
    EXPECT_EQ(output.at(1, 0), 0.0F);
}

TEST(LeakyBPTTSpikingMechanismTest, SoftResetExactSequence)
{
    auto layer = make_bptt(2, 1.0F, false);
    nn::Tensor input(2, 1);
    input.at(0, 0) = 1.5F;
    input.at(1, 0) = 0.0F;
    const nn::Tensor output = layer.forward(input, false);
    EXPECT_EQ(output.at(0, 0), 1.0F);
    EXPECT_EQ(output.at(1, 0), 0.0F);
}

TEST(LeakyBPTTSpikingMechanismTest, StatePersistsAcrossForwardCalls)
{
    auto layer = make_bptt(1, 100.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.0F), false).at(0, 0), 0.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 1.0F), false).at(0, 0), 0.0F);
    EXPECT_EQ(layer.forward(make_const(1, 1, 0.0F), false).at(0, 0), 0.0F);
}

TEST(LeakyBPTTSpikingMechanismTest, ResetStateClearsForFreshSequence)
{
    auto layer = make_bptt(1, 0.5F);
    layer.forward(make_const(1, 1, 1.0F), false);
    layer.reset_state();
    EXPECT_EQ(layer.forward(make_const(1, 1, 0.3F), false).at(0, 0), 0.0F);
}

TEST(LeakyBPTTSpikingMechanismTest, AdaptationBPTTRaisesThreshold)
{
    auto layer = make_bptt(2, 1.0F, true, false, 0.9F, 0.5F);
    nn::Tensor input(2, 1);
    input.at(0, 0) = 1.2F;
    input.at(1, 0) = 1.2F;
    const nn::Tensor output = layer.forward(input, false);
    EXPECT_EQ(output.at(0, 0), 1.0F);
    EXPECT_EQ(output.at(1, 0), 0.0F);
}

TEST(LeakyBPTTSpikingMechanismTest, ReadoutModeOutputsVMem)
{
    auto layer = make_bptt(2, 100.0F, true, true);
    nn::Tensor input(2, 1);
    input.at(0, 0) = 2.0F;
    input.at(1, 0) = 3.0F;
    const float beta = std::exp(-1.0F);
    const nn::Tensor output = layer.forward(input, false);
    EXPECT_NEAR(output.at(0, 0), 2.0F, kTol);
    EXPECT_NEAR(output.at(1, 0), 2.0F * beta + 3.0F, kTol);
}

TEST(ThresholdDependentBatchNormTest, ThrowsOnInvalidInputShape)
{
    nn::ThresholdDependentBatchNorm layer(2, 1.0F, 3);
    nn::Tensor bad(5, 2);
    bad.setZero();
    EXPECT_THROW(layer.forward(bad, false), std::invalid_argument);
}

TEST(ThresholdDependentBatchNormTest, ForwardOutputScaledByTdbnFactor)
{
    nn::ThresholdDependentBatchNorm layer(1, 2.0F, 4);
    nn::Tensor input(8, 1);
    for (int t = 0; t < 4; ++t)
    {
        input.at(static_cast<size_t>(t) * 2, 0) = -1.0F;
        input.at(static_cast<size_t>(t) * 2 + 1, 0) = 1.0F;
    }
    const nn::Tensor output = layer.forward(input, false);
    for (int t = 0; t < 4; ++t)
    {
        EXPECT_NEAR(output.at(static_cast<size_t>(t) * 2, 0), -1.0F, 1e-3F);
        EXPECT_NEAR(output.at(static_cast<size_t>(t) * 2 + 1, 0), 1.0F, 1e-3F);
    }
}

TEST(ThresholdDependentBatchNormTest, ForwardNormalizationZeroMeanInput)
{
    nn::ThresholdDependentBatchNorm layer(2, 1.0F);
    nn::Tensor input(8, 2);
    const float values[4] = {1.0F, 2.0F, 4.0F, 7.0F};
    for (int t = 0; t < 2; ++t)
        for (int b = 0; b < 4; ++b)
            for (int f = 0; f < 2; ++f)
                input.at(static_cast<size_t>(t) * 4 + static_cast<size_t>(b),
                    static_cast<size_t>(f)) = values[b] + static_cast<float>(f);
    const nn::Tensor output = layer.forward(input, false);
    for (int t = 0; t < 2; ++t)
        for (int f = 0; f < 2; ++f)
        {
            float mean = 0.0F;
            for (int b = 0; b < 4; ++b)
                mean += output.at(
                    static_cast<size_t>(t) * 4 + static_cast<size_t>(b), static_cast<size_t>(f));
            EXPECT_NEAR(mean / 4.0F, 0.0F, 1e-5F);
        }
}

TEST(ThresholdDependentBatchNormTest, BackwardGammaGradAccumulates)
{
    nn::ThresholdDependentBatchNorm layer(1, 1.0F);
    nn::Tensor input(2, 1);
    input.at(0, 0) = 1.0F;
    input.at(1, 0) = 3.0F;
    layer.forward(input, true);

    nn::Tensor grad_output(2, 1);
    grad_output.at(0, 0) = 2.0F;
    grad_output.at(1, 0) = 4.0F;
    layer.backward(grad_output);

    auto params = layer.params();
    ASSERT_EQ(params.size(), 2u);
    EXPECT_NEAR(params[0]->grad().at(0, 0), 2.0F, 0.01F);
}

TEST(ThresholdDependentBatchNormTest, BackwardBetaGradSumsOverBatch)
{
    nn::ThresholdDependentBatchNorm layer(1, 1.0F);
    nn::Tensor input(2, 1);
    input.at(0, 0) = 1.0F;
    input.at(1, 0) = 3.0F;
    layer.forward(input, true);

    nn::Tensor grad_output(2, 1);
    grad_output.at(0, 0) = 2.0F;
    grad_output.at(1, 0) = 4.0F;
    layer.backward(grad_output);

    auto params = layer.params();
    ASSERT_EQ(params.size(), 2u);
    EXPECT_NEAR(params[1]->grad().at(0, 0), 6.0F, 0.01F);
}

TEST(ThresholdDependentBatchNormTest, ParamsExposeGammaAndBeta)
{
    nn::ThresholdDependentBatchNorm layer(4, 1.0F);
    auto params = layer.params();
    ASSERT_EQ(params.size(), 2u);
    EXPECT_NE(params[0], nullptr);
    EXPECT_NE(params[1], nullptr);
}

TEST(SurrogateGradientMechanismTest, BoxcarExactBoundaryInsideWindow)
{
    BoxcarSurrogate surrogate(0.4F);
    auto eval = [&](float value)
    {
        nn::Tensor tensor(1, 1);
        tensor.at(0, 0) = value;
        return surrogate.calculate(tensor, 1.0F).at(0, 0);
    };

    EXPECT_EQ(eval(1.1F), 1.0F);
    EXPECT_EQ(eval(0.9F), 1.0F);
    EXPECT_EQ(eval(1.2F), 0.0F);
    EXPECT_EQ(eval(0.8F), 1.0F);
    EXPECT_EQ(eval(1.3F), 0.0F);
}

TEST(SurrogateGradientMechanismTest, ExponentialSurrogateMonotonicallyDecreases)
{
    ExponentialSurrogate surrogate(1.0F);
    nn::Tensor center(1, 1);
    nn::Tensor mid(1, 1);
    nn::Tensor far(1, 1);
    center.at(0, 0) = 0.0F;
    mid.at(0, 0) = 0.5F;
    far.at(0, 0) = 1.0F;
    const float g0 = surrogate.calculate(center, 0.0F).at(0, 0);
    const float g1 = surrogate.calculate(mid, 0.0F).at(0, 0);
    const float g2 = surrogate.calculate(far, 0.0F).at(0, 0);
    EXPECT_GT(g0, g1);
    EXPECT_GT(g1, g2);
}

TEST(SurrogateGradientMechanismTest, ExponentialSurrogateSharpnessEffect)
{
    nn::Tensor tensor(1, 1);
    tensor.at(0, 0) = 0.0F;
    EXPECT_NEAR(ExponentialSurrogate(0.5F).calculate(tensor, 0.0F).at(0, 0), 2.0F, 1e-5F);
    EXPECT_NEAR(ExponentialSurrogate(2.0F).calculate(tensor, 0.0F).at(0, 0), 0.5F, 1e-5F);
}

TEST(SurrogateGradientMechanismTest, BoxcarSurrogateSymmetric)
{
    BoxcarSurrogate surrogate(1.0F);
    nn::Tensor plus(1, 1);
    nn::Tensor minus(1, 1);
    plus.at(0, 0) = 0.8F;
    minus.at(0, 0) = 0.2F;
    EXPECT_EQ(surrogate.calculate(plus, 0.5F).at(0, 0), surrogate.calculate(minus, 0.5F).at(0, 0));
}

TEST(SurrogateGradientMechanismTest, InvalidHyperparametersThrow)
{
    EXPECT_THROW(BoxcarSurrogate(0.0F), std::invalid_argument);
    EXPECT_THROW(ExponentialSurrogate(0.0F), std::invalid_argument);
}
