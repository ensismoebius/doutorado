/**
 * @file layers_spiking_basic_gtest.cpp
 * @brief Leaky/LifIntegrator/LifBPTT/surrogate-gradient/LeakyReLU layer unit tests. (Deeper
 * spiking-mechanism coverage lives in spiking_mechanisms_gtest.cpp.)
 */

#include <gtest/gtest.h>

#include <memory>

#include "core/utility/tests/test_helpers.hpp"
#include "layers/Layers.hpp"
#include "layers/convolution/Conv2d.hpp"
#include "layers/spiking/Lif.hpp"
#include "layers/spiking/LifBPTT.hpp"
#include "tensor/Tensor.hpp"

using nn::Conv2d;
using nn::LeakyReLU;
using nn::Lif;
using nn::LifIntegrator;
using nn::Linear;
using nn::MAELoss;
using nn::MSELoss;
using nn::ReLU;
using nn::Sequential;
using nn::SimpleResNet;
using nn::SpikeCountLoss;

// Teste para Lif (LIF)
TEST(LeakyLayerTest, ForwardSpikeAndReset)
{
    Lif leaky(/*dt=*/1.0F,
        /*R=*/5.0F,
        /*C=*/1.0F,
        /*V_thresh=*/2.0F,
        /*reset_zero=*/true,
        0.0F,
        std::make_shared<ExponentialSurrogate>());
    nn::Tensor in_tensor(1, 1);
    in_tensor.at(0, 0) = 3.0F; // Acima do threshold
    nn::Tensor out{leaky.forward(in_tensor)};
    // Como input > threshold, deve gerar spike (1.0)
    ASSERT_FLOAT_EQ(out.at(0, 0), 1.0F);
    // Após spike, v_mem deve ser resetado para zero
    ASSERT_FLOAT_EQ(leaky.v_mem.at(0, 0), 0.0F);
}

// Teste para Lif sem reset para zero
TEST(LeakyLayerTest, ForwardSpikeNoResetZero)
{
    Lif leaky(/*dt=*/1.0F,
        /*R=*/5.0F,
        /*C=*/1.0F,
        /*V_thresh=*/2.0F,
        /*reset_zero=*/false,
        0.0F,
        std::make_shared<ExponentialSurrogate>());
    nn::Tensor in_tensor(1, 1);
    in_tensor.at(0, 0) = 3.0F; // Acima do threshold
    nn::Tensor out{leaky.forward(in_tensor)};
    // Deve gerar spike
    ASSERT_FLOAT_EQ(out.at(0, 0), 1.0F);
    // v_mem deve ser reduzido pelo threshold
    ASSERT_FLOAT_EQ(leaky.v_mem.at(0, 0), 1.0F); // 3.0 - 2.0
}

TEST(LeakyLayerTest, ForwardAnalyticParityAcrossSteps)
{
    Lif leaky(/*dt=*/1.0F,
        /*R=*/5.0F,
        /*C=*/1.0F,
        /*V_thresh=*/2.0F,
        /*reset_zero=*/false,
        0.0F,
        std::make_shared<ExponentialSurrogate>());

    nn::Tensor step1(1, 1);
    step1.at(0, 0) = 1.5F;
    nn::Tensor out1{leaky.forward(step1, true)};

    const float beta = std::exp(-1.0F / (5.0F * 1.0F));
    const float expected_v1_pre = 0.0F * beta + 1.5F;
    const float expected_s1 = (expected_v1_pre > 2.0F) ? 1.0F : 0.0F;
    const float expected_v1_post = expected_v1_pre - expected_s1 * 2.0F;

    EXPECT_NEAR(out1.at(0, 0), expected_s1, 1e-6F);
    EXPECT_NEAR(leaky.v_mem.at(0, 0), expected_v1_post, 1e-6F);

    nn::Tensor step2(1, 1);
    step2.at(0, 0) = 1.0F;
    nn::Tensor out2{leaky.forward(step2, true)};

    const float expected_v2_pre = expected_v1_post * beta + 1.0F;
    const float expected_s2 = (expected_v2_pre > 2.0F) ? 1.0F : 0.0F;
    const float expected_v2_post = expected_v2_pre - expected_s2 * 2.0F;

    EXPECT_NEAR(out2.at(0, 0), expected_s2, 1e-6F);
    EXPECT_NEAR(leaky.v_mem.at(0, 0), expected_v2_post, 1e-6F);
}

TEST(LeakyLayerTest, ParamsExposeTrainableCapacitance)
{
    Lif leaky(/*dt=*/1.0F,
        /*R=*/5.0F,
        /*C=*/1.0F,
        /*V_thresh=*/2.0F,
        /*reset_zero=*/true,
        0.0F,
        std::make_shared<ExponentialSurrogate>());

    auto parameters = leaky.params();
    ASSERT_EQ(parameters.size(), 3);
    EXPECT_EQ(parameters[0], &leaky.resistance);
    EXPECT_EQ(parameters[1], &leaky.voltage_threshold);
    EXPECT_EQ(parameters[2], &leaky.capacitance);
}

TEST(LeakyLayerTest, BackwardComputesCapacitanceGradient)
{
    Lif leaky(/*dt=*/1.0F,
        /*R=*/5.0F,
        /*C=*/1.0F,
        /*V_thresh=*/1.1F,
        /*reset_zero=*/true,
        0.0F,
        std::make_shared<ExponentialSurrogate>());

    nn::Tensor first_input(1, 1);
    first_input.at(0, 0) = 1.0F;
    [[maybe_unused]] nn::Tensor first_output = leaky.forward(first_input, true);

    nn::Tensor second_input(1, 1);
    second_input.at(0, 0) = 0.2F;
    [[maybe_unused]] nn::Tensor second_output = leaky.forward(second_input, true);

    nn::Tensor grad_output(1, 1);
    grad_output.at(0, 0) = 1.0F;
    [[maybe_unused]] nn::Tensor grad_input = leaky.backward(grad_output);

    // Analytical expectation for this setup:
    // dL/dC = dL/dbeta * dBeta/dC
    // dL/dbeta = surrogate(v_pre, V_th) * v(t-1)
    const float R = 5.0F;
    const float C = 1.0F;
    const float dt = 1.0F;
    const float beta = std::exp(-dt / (R * C));
    const float v_pre = beta * 1.0F + 0.2F;
    const float v_th = 1.1F;
    const float surrogate = std::exp(-std::abs(v_pre - v_th));
    const float expected_d_beta_dC = (beta * dt) / (R * C * C);
    const float expected_dL_dC = surrogate * 1.0F * expected_d_beta_dC;

    EXPECT_NEAR(leaky.capacitance.grad().at(0, 0), expected_dL_dC, 1e-5F);
}

TEST(LeakyLayerTest, NonPositiveResistanceUsesStableDecay)
{
    Lif leaky(/*dt=*/1.0F,
        /*R=*/-2.0F,
        /*C=*/1.0F,
        /*V_thresh=*/100.0F,
        /*reset_zero=*/true,
        0.0F,
        std::make_shared<ExponentialSurrogate>());

    nn::Tensor input(1, 1);
    input.at(0, 0) = 1.0F;
    [[maybe_unused]] auto out = leaky.forward(input, true);

    nn::Tensor grad_output(1, 1);
    grad_output.at(0, 0) = 1.0F;
    [[maybe_unused]] auto grad_input = leaky.backward(grad_output);

    // With clamped R and a single step from zero state, v_mem becomes input exactly.
    EXPECT_NEAR(leaky.v_mem.at(0, 0), 1.0F, 1e-7F);
    // Clamp-at-use semantics block RC gradients while raw parameters are non-positive.
    EXPECT_NEAR(leaky.resistance.grad().at(0, 0), 0.0F, 1e-7F);
    EXPECT_NEAR(leaky.capacitance.grad().at(0, 0), 0.0F, 1e-7F);
}

TEST(LeakyLayerTest, ClampedNonPositiveParamsDoNotAccumulateRCGradients)
{
    Lif leaky(/*dt=*/1.0F,
        /*R=*/-2.0F,
        /*C=*/-3.0F,
        /*V_thresh=*/100.0F,
        /*reset_zero=*/true,
        0.0F,
        std::make_shared<ExponentialSurrogate>());

    nn::Tensor first_input(1, 1);
    first_input.at(0, 0) = 1.0F;
    [[maybe_unused]] auto first_out = leaky.forward(first_input, true);

    nn::Tensor second_input(1, 1);
    second_input.at(0, 0) = 1.0F;
    [[maybe_unused]] auto second_out = leaky.forward(second_input, true);

    nn::Tensor grad_output(1, 1);
    grad_output.at(0, 0) = 1.0F;
    [[maybe_unused]] auto grad_input = leaky.backward(grad_output);

    EXPECT_NEAR(leaky.resistance.grad().at(0, 0), 0.0F, 1e-7F);
    EXPECT_NEAR(leaky.capacitance.grad().at(0, 0), 0.0F, 1e-7F);
}

TEST(LifIntegratorLayerTest, BackwardComputesCapacitanceGradient)
{
    LifIntegrator integrator(/*dt=*/1.0F, /*R=*/2.0F, /*C=*/3.0F);

    nn::Tensor first_input(1, 1);
    first_input.at(0, 0) = 2.0F;
    [[maybe_unused]] nn::Tensor first_output = integrator.forward(first_input, true);

    nn::Tensor second_input(1, 1);
    second_input.at(0, 0) = 1.0F;
    [[maybe_unused]] nn::Tensor second_output = integrator.forward(second_input, true);

    nn::Tensor grad_output(1, 1);
    grad_output.at(0, 0) = 1.0F;
    nn::Tensor grad_input = integrator.backward(grad_output);

    const float R = 2.0F;
    const float C = 3.0F;
    const float dt = 1.0F;
    const float beta = std::exp(-dt / (R * C));

    // dL/dbeta = sum(grad_output * v(t-1)) = 1 * 2.0
    const float expected_dL_dbeta = 2.0F;
    const float expected_d_beta_dC = (beta * dt) / (R * C * C);
    const float expected_dL_dC = expected_dL_dbeta * expected_d_beta_dC;

    EXPECT_NEAR(grad_input.at(0, 0), 1.0F, 1e-6F);
    EXPECT_NEAR(integrator.capacitance.grad().at(0, 0), expected_dL_dC, 1e-6F);
}

TEST(LifIntegratorLayerTest, NonPositiveResistanceUsesStableDecay)
{
    LifIntegrator integrator(/*dt=*/1.0F, /*R=*/-2.0F, /*C=*/3.0F);

    nn::Tensor input(1, 1);
    input.at(0, 0) = 2.0F;
    [[maybe_unused]] auto out = integrator.forward(input, true);

    nn::Tensor grad_output(1, 1);
    grad_output.at(0, 0) = 1.0F;
    [[maybe_unused]] auto grad_input = integrator.backward(grad_output);

    EXPECT_NEAR(integrator.v_mem.at(0, 0), 2.0F, 1e-7F);
    EXPECT_NEAR(integrator.resistance.grad().at(0, 0), 0.0F, 1e-7F);
    EXPECT_NEAR(integrator.capacitance.grad().at(0, 0), 0.0F, 1e-7F);
}

TEST(LifIntegratorLayerTest, ClampedNonPositiveParamsDoNotAccumulateRCGradients)
{
    LifIntegrator integrator(/*dt=*/1.0F, /*R=*/-2.0F, /*C=*/-3.0F);

    nn::Tensor first_input(1, 1);
    first_input.at(0, 0) = 2.0F;
    [[maybe_unused]] auto first_out = integrator.forward(first_input, true);

    nn::Tensor second_input(1, 1);
    second_input.at(0, 0) = 1.0F;
    [[maybe_unused]] auto second_out = integrator.forward(second_input, true);

    nn::Tensor grad_output(1, 1);
    grad_output.at(0, 0) = 1.0F;
    [[maybe_unused]] auto grad_input = integrator.backward(grad_output);

    EXPECT_NEAR(integrator.resistance.grad().at(0, 0), 0.0F, 1e-7F);
    EXPECT_NEAR(integrator.capacitance.grad().at(0, 0), 0.0F, 1e-7F);
}

TEST(LifBPTTLayerTest, ParamsExposeTrainableCapacitance)
{
    LifBPTTImpl<nn::Backend> leaky_bptt(/*time_steps=*/2,
        /*delta_t=*/1.0F,
        /*resistance=*/2.0F,
        /*capacitance=*/3.0F,
        /*voltage_threshold=*/1.0F);

    auto parameters = leaky_bptt.params();
    ASSERT_EQ(parameters.size(), 3);
    EXPECT_EQ(parameters[0], &leaky_bptt.resistance);
    EXPECT_EQ(parameters[1], &leaky_bptt.voltage_threshold);
    EXPECT_EQ(parameters[2], &leaky_bptt.capacitance);
}

TEST(LifBPTTLayerTest, BackwardReadoutModeMatchesTemporalGradientRecurrence)
{
    LifBPTTImpl<nn::Backend> leaky_bptt(/*time_steps=*/2,
        /*delta_t=*/1.0F,
        /*resistance=*/2.0F,
        /*capacitance=*/3.0F,
        /*voltage_threshold=*/100.0F,
        /*reset_zero=*/true,
        /*reset_potential=*/0.0F,
        /*readout_mode=*/true,
        std::make_shared<ExponentialSurrogate>());

    nn::Tensor input(2, 1);
    input.at(0, 0) = 2.0F;
    input.at(1, 0) = 1.0F;
    [[maybe_unused]] auto out = leaky_bptt.forward(input, true);

    nn::Tensor grad_output(2, 1);
    grad_output.at(0, 0) = 1.0F;
    grad_output.at(1, 0) = 1.0F;
    const nn::Tensor grad_input = leaky_bptt.backward(grad_output);

    const float beta = std::exp(-1.0F / (2.0F * 3.0F));
    EXPECT_NEAR(grad_input.at(0, 0), 1.0F + beta, 1e-6F);
    EXPECT_NEAR(grad_input.at(1, 0), 1.0F, 1e-6F);

    // In readout mode with high threshold, no spikes occur and the backward approximation
    // reduces to a simple dL/dR term from t=1 only: grad_v_pre(1) * v_pre(0) * d_beta/dR.
    const float d_beta_dR = (beta * 1.0F) / (3.0F * 2.0F * 2.0F);
    const float expected_dL_dR = 1.0F * 2.0F * d_beta_dR;
    EXPECT_NEAR(leaky_bptt.resistance.grad().at(0, 0), expected_dL_dR, 1e-6F);

    const float d_beta_dC = (beta * 1.0F) / (2.0F * 3.0F * 3.0F);
    const float expected_dL_dC = 1.0F * 2.0F * d_beta_dC;
    EXPECT_NEAR(leaky_bptt.capacitance.grad().at(0, 0), expected_dL_dC, 1e-6F);

    EXPECT_NEAR(leaky_bptt.voltage_threshold.grad().at(0, 0), 0.0F, 1e-6F);
}

TEST(LifBPTTLayerTest, BackwardSpikingModeUsesSurrogateAndTemporalRecurrence)
{
    // Large boxcar window keeps surrogate derivative at 1 over this test's voltage range.
    LifBPTTImpl<nn::Backend> leaky_bptt(/*time_steps=*/2,
        /*delta_t=*/1.0F,
        /*resistance=*/2.0F,
        /*capacitance=*/3.0F,
        /*voltage_threshold=*/100.0F,
        /*reset_zero=*/true,
        /*reset_potential=*/0.0F,
        /*readout_mode=*/false,
        std::make_shared<BoxcarSurrogate>(1e6F));

    nn::Tensor input(2, 1);
    input.at(0, 0) = 2.0F;
    input.at(1, 0) = 1.0F;
    [[maybe_unused]] auto out = leaky_bptt.forward(input, true);

    nn::Tensor grad_output(2, 1);
    grad_output.at(0, 0) = 1.0F;
    grad_output.at(1, 0) = 1.0F;
    const nn::Tensor grad_input = leaky_bptt.backward(grad_output);

    const float beta = std::exp(-1.0F / (2.0F * 3.0F));
    EXPECT_NEAR(grad_input.at(0, 0), 1.0F + beta, 1e-6F);
    EXPECT_NEAR(grad_input.at(1, 0), 1.0F, 1e-6F);

    const float expected_dL_dVth = -2.0F + 2.0F * beta;
    EXPECT_NEAR(leaky_bptt.voltage_threshold.grad().at(0, 0), expected_dL_dVth, 1e-6F);

    const float d_beta_dR = (beta * 1.0F) / (3.0F * 2.0F * 2.0F);
    const float expected_dL_dR = 1.0F * 2.0F * d_beta_dR;
    EXPECT_NEAR(leaky_bptt.resistance.grad().at(0, 0), expected_dL_dR, 1e-6F);

    const float d_beta_dC = (beta * 1.0F) / (2.0F * 3.0F * 3.0F);
    const float expected_dL_dC = 1.0F * 2.0F * d_beta_dC;
    EXPECT_NEAR(leaky_bptt.capacitance.grad().at(0, 0), expected_dL_dC, 1e-6F);
}

TEST(LifBPTTLayerTest, ReadoutModeIgnoresThresholdEvenForSpikeLikeInputs)
{
    LifBPTTImpl<nn::Backend> leaky_bptt(/*time_steps=*/2,
        /*delta_t=*/1.0F,
        /*resistance=*/2.0F,
        /*capacitance=*/3.0F,
        /*voltage_threshold=*/0.1F,
        /*reset_zero=*/true,
        /*reset_potential=*/0.0F,
        /*readout_mode=*/true,
        std::make_shared<BoxcarSurrogate>(1e6F));

    nn::Tensor input(2, 1);
    input.at(0, 0) = 5.0F;
    input.at(1, 0) = 5.0F;
    [[maybe_unused]] auto out = leaky_bptt.forward(input, true);

    nn::Tensor grad_output(2, 1);
    grad_output.at(0, 0) = 1.0F;
    grad_output.at(1, 0) = 1.0F;
    [[maybe_unused]] auto grad_input = leaky_bptt.backward(grad_output);

    EXPECT_NEAR(leaky_bptt.voltage_threshold.grad().at(0, 0), 0.0F, 1e-6F);
}

TEST(LifBPTTLayerTest, ClampedNonPositiveParamsDoNotAccumulateRCGradients)
{
    LifBPTTImpl<nn::Backend> leaky_bptt(/*time_steps=*/2,
        /*delta_t=*/1.0F,
        /*resistance=*/-2.0F,
        /*capacitance=*/-3.0F,
        /*voltage_threshold=*/0.5F,
        /*reset_zero=*/true,
        /*reset_potential=*/0.0F,
        /*readout_mode=*/true,
        std::make_shared<ExponentialSurrogate>());

    nn::Tensor input(2, 1);
    input.at(0, 0) = 1.0F;
    input.at(1, 0) = 1.0F;
    [[maybe_unused]] auto out = leaky_bptt.forward(input, true);

    nn::Tensor grad_output(2, 1);
    grad_output.at(0, 0) = 1.0F;
    grad_output.at(1, 0) = 1.0F;
    const auto grad_input = leaky_bptt.backward(grad_output);

    EXPECT_NEAR(grad_input.at(0, 0), 1.0F, 1e-7F);
    EXPECT_NEAR(grad_input.at(1, 0), 1.0F, 1e-7F);
    EXPECT_NEAR(leaky_bptt.voltage_threshold.grad().at(0, 0), 0.0F, 1e-7F);
    EXPECT_NEAR(leaky_bptt.resistance.grad().at(0, 0), 0.0F, 1e-7F);
    EXPECT_NEAR(leaky_bptt.capacitance.grad().at(0, 0), 0.0F, 1e-7F);
}

TEST(LifBPTTLayerTest, SoftResetThresholdGradientMatchesAnalytic)
{
    // Locks the soft-reset branch (reset_zero=false) threshold gradient against an analytic
    // derivation. This ensures the dvpost_dVth = (-spike + Vth*surr) term and the recurrent
    // reset-path accumulation both contribute correctly.
    //
    // Setup: R=2, C=3, dt=1, Vth=0.5, inputs=[2.0, 1.0], T=2, B=1.
    // BoxcarSurrogate with enormous window guarantees surr=1 everywhere.
    //
    // Forward:
    //   beta = exp(-1/6)
    //   t=0: v_pre[0]=2.0, spike=1, v_post[0]=2.0-0.5=1.5
    //   t=1: v_pre[1]=1.5*beta+1.0, spike=1, v_post[1]=v_pre[1]-0.5
    //
    // Backward (surr=1, spike=1, dvpost_dvpre=1-1*0.5=0.5, dvpost_dVth=-1+0.5*1=-0.5):
    //   t=1: grad_v_pre[1]=1.0, dL_dVth += -1.0
    //   t=0: grad_v_pre[0]=1.0+0.5*beta, dL_dVth += -1.0 + beta*(-0.5)
    //   total dL_dVth = -2.0 - 0.5*beta
    //
    // R/C gradients use v_post_history[0]=1.5:
    //   dL_dR = 1.5 * (beta/12)   [d_beta_dR = beta/(R^2*C) = beta/12]
    //   dL_dC = 1.5 * (beta/18)   [d_beta_dC = beta/(R*C^2) = beta/18]
    LifBPTTImpl<nn::Backend> leaky_bptt(/*time_steps=*/2,
        /*delta_t=*/1.0F,
        /*resistance=*/2.0F,
        /*capacitance=*/3.0F,
        /*voltage_threshold=*/0.5F,
        /*reset_zero=*/false, // soft reset
        /*reset_potential=*/0.0F,
        /*readout_mode=*/false,
        std::make_shared<BoxcarSurrogate>(1e6F));

    nn::Tensor input(2, 1);
    input.at(0, 0) = 2.0F;
    input.at(1, 0) = 1.0F;
    [[maybe_unused]] auto out = leaky_bptt.forward(input, true);

    nn::Tensor grad_output(2, 1);
    grad_output.at(0, 0) = 1.0F;
    grad_output.at(1, 0) = 1.0F;
    [[maybe_unused]] auto grad_input = leaky_bptt.backward(grad_output);

    const float beta = std::exp(-1.0F / (2.0F * 3.0F));

    // Threshold gradient: direct spike term + recurrent reset-path term
    const float expected_dL_dVth = -2.0F - 0.5F * beta;
    EXPECT_NEAR(leaky_bptt.voltage_threshold.grad().at(0, 0), expected_dL_dVth, 1e-5F);

    // Input gradient through soft-reset recurrence
    EXPECT_NEAR(grad_input.at(0, 0), 1.0F + 0.5F * beta, 1e-6F);
    EXPECT_NEAR(grad_input.at(1, 0), 1.0F, 1e-6F);

    // R and C gradients via v_post_history[0]=1.5 (post-soft-reset state)
    const float d_beta_dR = (beta * 1.0F) / (3.0F * 2.0F * 2.0F); // beta/(C*R^2)
    const float d_beta_dC = (beta * 1.0F) / (2.0F * 3.0F * 3.0F); // beta/(R*C^2)
    EXPECT_NEAR(leaky_bptt.resistance.grad().at(0, 0), 1.5F * d_beta_dR, 1e-6F);
    EXPECT_NEAR(leaky_bptt.capacitance.grad().at(0, 0), 1.5F * d_beta_dC, 1e-6F);
}

// Teste para SurrogateGradient
TEST(SurrogateGradientTest, Exponential)
{
    ExponentialSurrogate surrogate(1.0F);
    nn::Tensor v_mem_tensor(1, 1);
    v_mem_tensor.at(0, 0) = 2.1F;
    auto grad = surrogate.calculate(v_mem_tensor, 2.0F);
    ASSERT_NEAR(grad.at(0, 0), 0.9048374, 1e-5F);
}

TEST(SurrogateGradientTest, Boxcar)
{
    BoxcarSurrogate surrogate(0.5F);
    nn::Tensor v_mem_tensor(1, 2);
    v_mem_tensor.at(0, 0) = 2.1F;
    v_mem_tensor.at(0, 1) = 2.3F;
    auto grad = surrogate.calculate(v_mem_tensor, 2.0F);
    ASSERT_FLOAT_EQ(grad.at(0, 0), 1.0F);
    ASSERT_FLOAT_EQ(grad.at(0, 1), 0.0F);
}

TEST(SurrogateGradientTest, ThrowsOnInvalidHyperparameters)
{
    EXPECT_THROW((void) ExponentialSurrogate(0.0F), std::invalid_argument);
    EXPECT_THROW((void) ExponentialSurrogate(-1.0F), std::invalid_argument);
    EXPECT_THROW((void) BoxcarSurrogate(0.0F), std::invalid_argument);
    EXPECT_THROW((void) BoxcarSurrogate(-0.5F), std::invalid_argument);
}

// BoxcarSurrogate.width() accessor (BoxcarSurrogate.hpp line 42)
TEST(SurrogateGradientTest, BoxcarSurrogateWidthAccessor)
{
    BoxcarSurrogate surrogate(0.5F);
    EXPECT_FLOAT_EQ(surrogate.width(), 0.5F);
}

// Teste para LeakyReLU
TEST(LeakyReLUTest, ForwardAndBackward)
{
    LeakyReLU leaky_relu(0.1F);
    nn::Tensor in_tensor(2, 1);
    in_tensor.at(0, 0) = 1.0F;
    in_tensor.at(1, 0) = -2.0F;
    nn::Tensor out{leaky_relu.forward(in_tensor)};
    ASSERT_NEAR(out.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(out.at(1, 0), -0.2F, 1e-5F);
    nn::Tensor grad_tensor(2, 1);
    grad_tensor.at(0, 0) = 1.0F;
    grad_tensor.at(1, 0) = 1.0F;
    nn::Tensor grad_in{leaky_relu.backward(grad_tensor)};
    ASSERT_NEAR(grad_in.at(0, 0), 1.0F, 1e-5F);
    ASSERT_NEAR(grad_in.at(1, 0), 0.1F, 1e-5F);
}
