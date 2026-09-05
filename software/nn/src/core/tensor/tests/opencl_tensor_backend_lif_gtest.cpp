/**
 * @file opencl_tensor_backend_lif_gtest.cpp
 * @brief LIF step/gradient helper and Leaky-layer parity correctness on the OpenCL backend.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/spiking/Lif.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

TEST(OpenCLTensorBackendTest, LifStepHelperCorrectness)
{
    nn::OpenCLTensorBackend v_mem(1, 4);
    v_mem.at(0, 0) = 0.0f;
    v_mem.at(0, 1) = 0.4f;
    v_mem.at(0, 2) = 0.8f;
    v_mem.at(0, 3) = -0.5f;

    nn::OpenCLTensorBackend input(1, 4);
    input.at(0, 0) = 0.3f;
    input.at(0, 1) = 0.3f;
    input.at(0, 2) = 0.3f;
    input.at(0, 3) = 0.3f;

    nn::OpenCLTensorBackend spikes(1, 4);
    spikes.fill(0.0f);

    v_mem.lif_step_inplace(input, spikes, nullptr, 0.5f, 0.6f, 0.0f, true, 0.9f, 0.2f, false);

    EXPECT_NEAR(spikes.at(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(spikes.at(0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(spikes.at(0, 2), 1.0f, 1e-6f);
    EXPECT_NEAR(spikes.at(0, 3), 0.0f, 1e-6f);

    EXPECT_NEAR(v_mem.at(0, 0), 0.3f, 1e-6f);
    EXPECT_NEAR(v_mem.at(0, 1), 0.5f, 1e-6f);
    EXPECT_NEAR(v_mem.at(0, 2), 0.0f, 1e-6f);
    EXPECT_NEAR(v_mem.at(0, 3), 0.05f, 1e-6f);
}

TEST(OpenCLTensorBackendTest, LifGradHelperCorrectness)
{
    nn::OpenCLTensorBackend v_pre(1, 3);
    v_pre.at(0, 0) = 0.6f;
    v_pre.at(0, 1) = 0.8f;
    v_pre.at(0, 2) = 0.4f;

    auto grad = v_pre.lif_grad(0.6f, 0.5f);

    EXPECT_NEAR(grad.at(0, 0), 2.0f, 1e-5f);
    const float off_center = 2.0f * std::exp(-0.4f);
    EXPECT_NEAR(grad.at(0, 1), off_center, 1e-5f);
    EXPECT_NEAR(grad.at(0, 2), off_center, 1e-5f);
}

TEST(OpenCLTensorBackendTest, LeakyLayerForwardParityOnOpenCLBackend)
{
    using OpenCLTensor = nn::TensorImpl<nn::OpenCLTensorBackend>;
    ::LifImpl<nn::OpenCLTensorBackend> leaky(/*dt=*/1.0F,
        /*R=*/5.0F,
        /*C=*/1.0F,
        /*V_thresh=*/2.0F,
        /*reset_zero=*/false,
        /*reset_potential=*/0.0F,
        std::make_shared<ExponentialSurrogate>(0.5F));

    OpenCLTensor step1(1, 1);
    step1.at(0, 0) = 1.5F;
    auto out1 = leaky.forward(step1, true);

    const float beta = std::exp(-1.0F / (5.0F * 1.0F));
    const float expected_v1_pre = 0.0F * beta + 1.5F;
    const float expected_s1 = (expected_v1_pre > 2.0F) ? 1.0F : 0.0F;
    const float expected_v1_post = expected_v1_pre - expected_s1 * 2.0F;

    EXPECT_NEAR(out1.at(0, 0), expected_s1, 1e-6F);
    EXPECT_NEAR(leaky.v_mem.at(0, 0), expected_v1_post, 1e-6F);

    OpenCLTensor step2(1, 1);
    step2.at(0, 0) = 1.0F;
    auto out2 = leaky.forward(step2, true);

    const float expected_v2_pre = expected_v1_post * beta + 1.0F;
    const float expected_s2 = (expected_v2_pre > 2.0F) ? 1.0F : 0.0F;
    const float expected_v2_post = expected_v2_pre - expected_s2 * 2.0F;

    EXPECT_NEAR(out2.at(0, 0), expected_s2, 1e-6F);
    EXPECT_NEAR(leaky.v_mem.at(0, 0), expected_v2_post, 1e-6F);
}

TEST(OpenCLTensorBackendTest, LeakyLayerBackwardExponentialSurrogateOnOpenCLBackend)
{
    using OpenCLTensor = nn::TensorImpl<nn::OpenCLTensorBackend>;
    ::LifImpl<nn::OpenCLTensorBackend> leaky(/*dt=*/1.0F,
        /*R=*/5.0F,
        /*C=*/1.0F,
        /*V_thresh=*/2.0F,
        /*reset_zero=*/false,
        /*reset_potential=*/0.0F,
        std::make_shared<ExponentialSurrogate>(0.5F));

    OpenCLTensor input(1, 1);
    input.at(0, 0) = 2.5F;
    (void) leaky.forward(input, true);

    OpenCLTensor grad_output(1, 1);
    grad_output.at(0, 0) = 1.0F;
    auto grad_input = leaky.backward(grad_output);

    const float expected_surrogate = (1.0F / 0.5F) * std::exp(-std::abs(2.5F - 2.0F) / 0.5F);
    EXPECT_NEAR(grad_input.at(0, 0), expected_surrogate, 1e-5F);
    EXPECT_NEAR(leaky.voltage_threshold.grad().at(0, 0), -expected_surrogate, 1e-5F);
}

} // namespace
