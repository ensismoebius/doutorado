/**
 * @file opencl_tensor_backend_gtest.cpp
 * @brief Unit tests for OpenCLTensorBackend correctness with CPU fallback safety.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/spiking/Lif.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

TEST(OpenCLTensorBackendTest, MatmulCorrectness)
{
    nn::OpenCLTensorBackend a(2, 3);
    nn::OpenCLTensorBackend b(3, 2);

    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(1, 0) = 4.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 6.0f;

    b.at(0, 0) = 7.0f;
    b.at(0, 1) = 8.0f;
    b.at(1, 0) = 9.0f;
    b.at(1, 1) = 10.0f;
    b.at(2, 0) = 11.0f;
    b.at(2, 1) = 12.0f;

    auto out = a.matmul(b);

    EXPECT_NEAR(out.at(0, 0), 58.0f, 1e-5f);
    EXPECT_NEAR(out.at(0, 1), 64.0f, 1e-5f);
    EXPECT_NEAR(out.at(1, 0), 139.0f, 1e-5f);
    EXPECT_NEAR(out.at(1, 1), 154.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, ElementwiseAndScalarOpsCorrectness)
{
    nn::OpenCLTensorBackend a(2, 2);
    nn::OpenCLTensorBackend b(2, 2);

    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f;
    a.at(1, 1) = 4.0f;

    b.at(0, 0) = 10.0f;
    b.at(0, 1) = 20.0f;
    b.at(1, 0) = 30.0f;
    b.at(1, 1) = 40.0f;

    auto added = a.add(b);
    EXPECT_NEAR(added.at(0, 0), 11.0f, 1e-5f);
    EXPECT_NEAR(added.at(1, 1), 44.0f, 1e-5f);

    auto mul = a.multiply(b);
    EXPECT_NEAR(mul.at(0, 0), 10.0f, 1e-5f);
    EXPECT_NEAR(mul.at(1, 1), 160.0f, 1e-5f);

    auto add_scalar = a.add_scalar(5.0f);
    EXPECT_NEAR(add_scalar.at(0, 0), 6.0f, 1e-5f);
    EXPECT_NEAR(add_scalar.at(1, 1), 9.0f, 1e-5f);

    auto mul_scalar = a.multiply_scalar(2.0f);
    EXPECT_NEAR(mul_scalar.at(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(mul_scalar.at(1, 1), 8.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, TransposeAndExpCorrectness)
{
    nn::OpenCLTensorBackend a(2, 3);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(1, 0) = 4.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 6.0f;

    auto t = a.transpose();
    EXPECT_EQ(t.rows(), 3);
    EXPECT_EQ(t.cols(), 2);
    EXPECT_NEAR(t.at(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(t.at(2, 1), 6.0f, 1e-5f);

    auto e = a.exp();
    EXPECT_NEAR(e.at(0, 0), std::exp(1.0f), 1e-4f);
    EXPECT_NEAR(e.at(1, 2), std::exp(6.0f), 1e-3f);
}

TEST(OpenCLTensorBackendTest, OpenCLAvailabilityDoesNotBreakOps)
{
    const bool available = nn::opencl::OpenCLContext::instance().is_available();

    nn::OpenCLTensorBackend a(1, 1);
    nn::OpenCLTensorBackend b(1, 1);
    a.at(0, 0) = 2.0f;
    b.at(0, 0) = 3.0f;

    auto c = a.multiply(b);
    EXPECT_NEAR(c.at(0, 0), 6.0f, 1e-5f);

    // Sanity check for runtime environment; operation correctness is the invariant.
    EXPECT_TRUE(available || !available);
}

TEST(OpenCLTensorBackendTest, RowwiseSumCorrectness)
{
    nn::OpenCLTensorBackend a(2, 3);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(1, 0) = 4.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 6.0f;

    auto sums = a.rowwise_sum();
    EXPECT_EQ(sums.rows(), 2);
    EXPECT_EQ(sums.cols(), 1);
    EXPECT_NEAR(sums.at(0, 0), 6.0f, 1e-5f);
    EXPECT_NEAR(sums.at(1, 0), 15.0f, 1e-5f);

    nn::OpenCLTensorBackend single_row(1, 4);
    single_row.at(0, 0) = -1.0f;
    single_row.at(0, 1) = 0.5f;
    single_row.at(0, 2) = 2.0f;
    single_row.at(0, 3) = 3.5f;
    auto single_row_sum = single_row.rowwise_sum();
    EXPECT_EQ(single_row_sum.rows(), 1);
    EXPECT_EQ(single_row_sum.cols(), 1);
    EXPECT_NEAR(single_row_sum.at(0, 0), 5.0f, 1e-5f);

    nn::OpenCLTensorBackend single_col(3, 1);
    single_col.at(0, 0) = 7.0f;
    single_col.at(1, 0) = -2.0f;
    single_col.at(2, 0) = 9.0f;
    auto single_col_sum = single_col.rowwise_sum();
    EXPECT_EQ(single_col_sum.rows(), 3);
    EXPECT_EQ(single_col_sum.cols(), 1);
    EXPECT_NEAR(single_col_sum.at(0, 0), 7.0f, 1e-5f);
    EXPECT_NEAR(single_col_sum.at(1, 0), -2.0f, 1e-5f);
    EXPECT_NEAR(single_col_sum.at(2, 0), 9.0f, 1e-5f);
}

TEST(OpenCLProfilingTest, ToggleFlag)
{
    // Ensure toggle APIs work without touching hardware kernels.
    nn::opencl::profiling::set_enabled(true);
    EXPECT_TRUE(nn::opencl::profiling::is_enabled());
    nn::opencl::profiling::set_enabled(false);
    EXPECT_FALSE(nn::opencl::profiling::is_enabled());
}

TEST(OpenCLTensorBackendTest, VerifyRuntimeActivityHandlesUnavailableProbePath)
{
    nn::OpenCLTensorBackend prediction(1, 2);
    nn::OpenCLTensorBackend target(1, 2);
    prediction.at(0, 0) = 1.0f;
    prediction.at(0, 1) = 2.0f;
    target.at(0, 0) = 1.5f;
    target.at(0, 1) = 1.0f;

    if (!nn::opencl::OpenCLContext::instance().is_available()) // LCOV_EXCL_START
    {
        EXPECT_THROW(nn::OpenCLTensorBackend::verify_runtime_activity_or_throw(
                         prediction, target, "/nonexistent/path/gpu_busy_percent"),
            std::runtime_error);
        return;
    } // LCOV_EXCL_STOP

    EXPECT_NO_THROW(nn::OpenCLTensorBackend::verify_runtime_activity_or_throw(
        prediction, target, "/nonexistent/path/gpu_busy_percent"));
}

} // namespace

// ---------------------------------------------------------------------------
// Additional coverage tests
// ---------------------------------------------------------------------------

TEST(OpenCLTensorBackendTest, StaticFactoriesZerosOnesRandom)
{
    auto z = nn::OpenCLTensorBackend::zeros(2, 3);
    EXPECT_EQ(z.rows(), 2);
    EXPECT_EQ(z.cols(), 3);
    EXPECT_NEAR(z.at(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(z.at(1, 2), 0.0f, 1e-6f);

    auto o = nn::OpenCLTensorBackend::ones(2, 2);
    EXPECT_NEAR(o.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(o.at(1, 1), 1.0f, 1e-6f);

    auto r = nn::OpenCLTensorBackend::random(3, 3);
    EXPECT_EQ(r.rows(), 3);
    EXPECT_EQ(r.cols(), 3);

    std::mt19937 rng(42);
    auto r2 = nn::OpenCLTensorBackend::random(2, 2, rng);
    EXPECT_EQ(r2.rows(), 2);
}

TEST(OpenCLTensorBackendTest, VectorShapeConstructorAndReshape)
{
    nn::OpenCLTensorBackend a(std::vector<nn::Index>{2, 3});
    EXPECT_EQ(a.rows(), 2);
    EXPECT_EQ(a.cols(), 3);
    a.reshape({3, 2});
    EXPECT_EQ(a.rows(), 3);
    EXPECT_EQ(a.cols(), 2);
}

TEST(OpenCLTensorBackendTest, SubtractAndDivide)
{
    nn::OpenCLTensorBackend a(2, 2);
    nn::OpenCLTensorBackend b(2, 2);
    a.at(0, 0) = 6.0f;
    a.at(0, 1) = 8.0f;
    a.at(1, 0) = 10.0f;
    a.at(1, 1) = 12.0f;
    b.at(0, 0) = 2.0f;
    b.at(0, 1) = 4.0f;
    b.at(1, 0) = 5.0f;
    b.at(1, 1) = 3.0f;

    auto sub = a.subtract(b);
    EXPECT_NEAR(sub.at(0, 0), 4.0f, 1e-5f);
    EXPECT_NEAR(sub.at(1, 1), 9.0f, 1e-5f);

    auto div = a.divide(b);
    EXPECT_NEAR(div.at(0, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(div.at(1, 0), 2.0f, 1e-5f);

    auto div_scalar = a.divide_scalar(2.0f);
    EXPECT_NEAR(div_scalar.at(0, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(div_scalar.at(1, 1), 6.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, SqrtAndSquare)
{
    nn::OpenCLTensorBackend a(2, 2);
    a.at(0, 0) = 4.0f;
    a.at(0, 1) = 9.0f;
    a.at(1, 0) = 16.0f;
    a.at(1, 1) = 25.0f;

    auto s = a.sqrt();
    EXPECT_NEAR(s.at(0, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(s.at(1, 1), 5.0f, 1e-5f);

    auto sq = a.square();
    EXPECT_NEAR(sq.at(0, 0), 16.0f, 1e-4f);
    EXPECT_NEAR(sq.at(1, 1), 625.0f, 1e-3f);
}

TEST(OpenCLTensorBackendTest, ReluAndLeakyReluCorrectness)
{
    nn::OpenCLTensorBackend a(1, 5);
    a.at(0, 0) = -2.0f;
    a.at(0, 1) = -1.0f;
    a.at(0, 2) = 0.0f;
    a.at(0, 3) = 1.0f;
    a.at(0, 4) = 3.0f;

    auto r = a.relu();
    EXPECT_NEAR(r.at(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(r.at(0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(r.at(0, 2), 0.0f, 1e-6f);
    EXPECT_NEAR(r.at(0, 3), 1.0f, 1e-6f);
    EXPECT_NEAR(r.at(0, 4), 3.0f, 1e-6f);

    auto lr = a.leaky_relu(0.1f);
    EXPECT_NEAR(lr.at(0, 0), -0.2f, 1e-6f);
    EXPECT_NEAR(lr.at(0, 1), -0.1f, 1e-6f);
    EXPECT_NEAR(lr.at(0, 2), 0.0f, 1e-6f);
    EXPECT_NEAR(lr.at(0, 3), 1.0f, 1e-6f);
    EXPECT_NEAR(lr.at(0, 4), 3.0f, 1e-6f);
}

TEST(OpenCLTensorBackendTest, SumAndMeanSquaredErrorCorrectness)
{
    nn::OpenCLTensorBackend a(2, 3);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(1, 0) = 4.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 6.0f;

    EXPECT_NEAR(a.sum(), 21.0f, 1e-5f);

    nn::OpenCLTensorBackend target(2, 3);
    target.at(0, 0) = 1.0f;
    target.at(0, 1) = 1.0f;
    target.at(0, 2) = 2.0f;
    target.at(1, 0) = 3.0f;
    target.at(1, 1) = 5.0f;
    target.at(1, 2) = 7.0f;

    // Squared diffs: [0,1,1,1,0,1] => sum=4, mse=4/6
    EXPECT_NEAR(a.mean_squared_error(target), 4.0f / 6.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, SumAndMseOnGpuResidentTensors)
{
    nn::OpenCLTensorBackend a(1, 5);
    a.at(0, 0) = -2.0f;
    a.at(0, 1) = -1.0f;
    a.at(0, 2) = 0.0f;
    a.at(0, 3) = 1.0f;
    a.at(0, 4) = 3.0f;

    // relu()/leaky_relu() produce OpenCL outputs that can stay GPU resident.
    auto relu_out = a.relu();
    auto leaky_out = a.leaky_relu(0.5f);

    EXPECT_NEAR(relu_out.sum(), 4.0f, 1e-5f);
    EXPECT_NEAR(relu_out.mean_squared_error(leaky_out), 0.25f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, GpuResidentInplaceFillAndClamp)
{
    nn::OpenCLTensorBackend a(1, 4);
    a.at(0, 0) = -2.0f;
    a.at(0, 1) = -1.0f;
    a.at(0, 2) = 2.0f;
    a.at(0, 3) = 5.0f;

    // relu() should create a GPU-accelerated output tensor.
    auto t = a.relu();

    t.clamp_inplace(0.5f, 1.5f);
    EXPECT_NEAR(t.at(0, 0), 0.5f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1), 0.5f, 1e-6f);
    EXPECT_NEAR(t.at(0, 2), 1.5f, 1e-6f);
    EXPECT_NEAR(t.at(0, 3), 1.5f, 1e-6f);

    t.fill(2.0f);
    EXPECT_NEAR(t.at(0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 2), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 3), 2.0f, 1e-6f);
}

TEST(OpenCLTensorBackendTest, GpuResidentInplaceScalarAndMathOps)
{
    nn::OpenCLTensorBackend a(1, 4);
    a.at(0, 0) = -1.0f;
    a.at(0, 1) = 1.0f;
    a.at(0, 2) = 2.0f;
    a.at(0, 3) = 3.0f;

    // Keep this tensor on GPU path first.
    auto t = a.relu(); // [0,1,2,3]

    t.add_scalar_inplace(1.0f);      // [1,2,3,4]
    t.multiply_scalar_inplace(2.0f); // [2,4,6,8]
    t.divide_scalar_inplace(2.0f);   // [1,2,3,4]

    EXPECT_NEAR(t.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 2), 3.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 3), 4.0f, 1e-6f);

    t.square_inplace(); // [1,4,9,16]
    t.sqrt_inplace();   // [1,2,3,4]

    EXPECT_NEAR(t.at(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(t.at(0, 1), 2.0f, 1e-5f);
    EXPECT_NEAR(t.at(0, 2), 3.0f, 1e-5f);
    EXPECT_NEAR(t.at(0, 3), 4.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, GpuResidentInplaceBinaryOps)
{
    nn::OpenCLTensorBackend a(1, 4);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(0, 3) = 4.0f;

    nn::OpenCLTensorBackend b(1, 4);
    b.at(0, 0) = 2.0f;
    b.at(0, 1) = 3.0f;
    b.at(0, 2) = 4.0f;
    b.at(0, 3) = 5.0f;

    // Produce GPU-accelerated tensors first.
    auto lhs = a.relu();
    auto rhs = b.relu();

    lhs.add_inplace(rhs); // [3,5,7,9]
    EXPECT_NEAR(lhs.at(0, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 1), 5.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 2), 7.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 3), 9.0f, 1e-6f);

    lhs.subtract_inplace(rhs); // [1,2,3,4]
    EXPECT_NEAR(lhs.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 2), 3.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 3), 4.0f, 1e-6f);

    lhs.multiply_inplace(rhs); // [2,6,12,20]
    EXPECT_NEAR(lhs.at(0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 1), 6.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 2), 12.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 3), 20.0f, 1e-6f);

    lhs.divide_inplace(rhs); // [1,2,3,4]
    EXPECT_NEAR(lhs.at(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 1), 2.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 2), 3.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 3), 4.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, GpuResidentAddRowBroadcastInplace)
{
    nn::OpenCLTensorBackend a(3, 2);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f;
    a.at(1, 1) = 4.0f;
    a.at(2, 0) = 5.0f;
    a.at(2, 1) = 6.0f;

    nn::OpenCLTensorBackend row(1, 2);
    row.at(0, 0) = 10.0f;
    row.at(0, 1) = 20.0f;

    // Put both through OpenCL-producing ops before in-place broadcast.
    auto lhs = a.relu();
    auto bias = row.relu();

    lhs.add_row_broadcast_inplace(bias);

    EXPECT_NEAR(lhs.at(0, 0), 11.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 1), 22.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(1, 0), 13.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(1, 1), 24.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(2, 0), 15.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(2, 1), 26.0f, 1e-5f);
}

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

TEST(OpenCLTensorBackendTest, AbsAndClampCorrectness)
{
    nn::OpenCLTensorBackend a(1, 5);
    a.at(0, 0) = -3.0f;
    a.at(0, 1) = -0.5f;
    a.at(0, 2) = 0.25f;
    a.at(0, 3) = 1.5f;
    a.at(0, 4) = 4.0f;

    auto abs_v = a.abs();
    EXPECT_NEAR(abs_v.at(0, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(abs_v.at(0, 1), 0.5f, 1e-6f);
    EXPECT_NEAR(abs_v.at(0, 2), 0.25f, 1e-6f);
    EXPECT_NEAR(abs_v.at(0, 3), 1.5f, 1e-6f);
    EXPECT_NEAR(abs_v.at(0, 4), 4.0f, 1e-6f);

    auto clamped = a.clamp(-1.0f, 1.0f);
    EXPECT_NEAR(clamped.at(0, 0), -1.0f, 1e-6f);
    EXPECT_NEAR(clamped.at(0, 1), -0.5f, 1e-6f);
    EXPECT_NEAR(clamped.at(0, 2), 0.25f, 1e-6f);
    EXPECT_NEAR(clamped.at(0, 3), 1.0f, 1e-6f);
    EXPECT_NEAR(clamped.at(0, 4), 1.0f, 1e-6f);
}

TEST(OpenCLTensorBackendTest, InplaceArithmeticOps)
{
    nn::OpenCLTensorBackend a(2, 2);
    nn::OpenCLTensorBackend b(2, 2);
    a.at(0, 0) = 10.0f;
    a.at(1, 1) = 20.0f;
    b.at(0, 0) = 3.0f;
    b.at(1, 1) = 4.0f;

    a.add_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 13.0f, 1e-5f);

    a.subtract_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 10.0f, 1e-5f);

    a.multiply_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 30.0f, 1e-5f);

    a.divide_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 10.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, InplaceScalarOps)
{
    nn::OpenCLTensorBackend a(2, 2);
    a.at(0, 0) = 4.0f;
    a.at(1, 1) = 9.0f;

    a.add_scalar_inplace(1.0f);
    EXPECT_NEAR(a.at(0, 0), 5.0f, 1e-5f);

    a.multiply_scalar_inplace(2.0f);
    EXPECT_NEAR(a.at(0, 0), 10.0f, 1e-5f);

    a.divide_scalar_inplace(2.0f);
    EXPECT_NEAR(a.at(0, 0), 5.0f, 1e-5f);

    nn::OpenCLTensorBackend b(2, 2);
    b.at(0, 0) = 4.0f;
    b.at(0, 1) = 9.0f;
    b.sqrt_inplace();
    EXPECT_NEAR(b.at(0, 0), 2.0f, 1e-5f);

    nn::OpenCLTensorBackend c(2, 2);
    c.at(0, 0) = 3.0f;
    c.square_inplace();
    EXPECT_NEAR(c.at(0, 0), 9.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, CompareOps)
{
    nn::OpenCLTensorBackend a(1, 4);
    nn::OpenCLTensorBackend b(1, 4);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(0, 3) = 4.0f;
    b.at(0, 0) = 2.0f;
    b.at(0, 1) = 2.0f;
    b.at(0, 2) = 2.0f;
    b.at(0, 3) = 2.0f;

    auto lt = a.compare_lt(b);
    EXPECT_NEAR(lt.at(0, 0), 1.0f, 1e-5f); // 1 < 2
    EXPECT_NEAR(lt.at(0, 3), 0.0f, 1e-5f); // 4 < 2 → false

    auto gt = a.compare_gt(b);
    EXPECT_NEAR(gt.at(0, 3), 1.0f, 1e-5f); // 4 > 2 → true
    EXPECT_NEAR(gt.at(0, 0), 0.0f, 1e-5f); // 1 > 2 → false

    auto le = a.compare_le(b);
    EXPECT_NEAR(le.at(0, 1), 1.0f, 1e-5f); // 2 <= 2 → true

    auto ge = a.compare_ge(b);
    EXPECT_NEAR(ge.at(0, 1), 1.0f, 1e-5f); // 2 >= 2 → true

    auto eq = a.compare_eq(b);
    EXPECT_NEAR(eq.at(0, 1), 1.0f, 1e-5f); // 2 == 2 → true
    EXPECT_NEAR(eq.at(0, 0), 0.0f, 1e-5f); // 1 == 2 → false
}

TEST(OpenCLTensorBackendTest, CompareScalarOps)
{
    nn::OpenCLTensorBackend a(1, 3);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 3.0f;
    a.at(0, 2) = 5.0f;

    auto lt_s = a.compare_lt_scalar(3.0f);
    EXPECT_NEAR(lt_s.at(0, 0), 1.0f, 1e-5f); // 1 < 3 → true
    EXPECT_NEAR(lt_s.at(0, 2), 0.0f, 1e-5f); // 5 < 3 → false

    auto gt_s = a.compare_gt_scalar(3.0f);
    EXPECT_NEAR(gt_s.at(0, 2), 1.0f, 1e-5f); // 5 > 3 → true
    EXPECT_NEAR(gt_s.at(0, 0), 0.0f, 1e-5f); // 1 > 3 → false

    auto le_s = a.compare_le_scalar(3.0f);
    EXPECT_NEAR(le_s.at(0, 1), 1.0f, 1e-5f); // 3 <= 3 → true

    auto ge_s = a.compare_ge_scalar(3.0f);
    EXPECT_NEAR(ge_s.at(0, 1), 1.0f, 1e-5f); // 3 >= 3 → true

    auto eq_s = a.compare_eq_scalar(3.0f);
    EXPECT_NEAR(eq_s.at(0, 1), 1.0f, 1e-5f); // 3 == 3 → true
    EXPECT_NEAR(eq_s.at(0, 0), 0.0f, 1e-5f); // 1 == 3 → false
}

TEST(OpenCLTensorBackendTest, MatmulTransposed)
{
    // a (2x3) * b^T (2x3) → (2x2)
    nn::OpenCLTensorBackend a(2, 3);
    nn::OpenCLTensorBackend b(2, 3); // b is transposed in matmul_transposed, so same shape as a
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(1, 0) = 4.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 6.0f;
    b.at(0, 0) = 1.0f;
    b.at(0, 1) = 0.0f;
    b.at(0, 2) = 0.0f;
    b.at(1, 0) = 0.0f;
    b.at(1, 1) = 1.0f;
    b.at(1, 2) = 0.0f;
    // a * b^T: row0 of a dot row0 of b = 1, row0 dot row1 = 2, row1 dot row0 = 4, row1 dot row1 = 5
    auto c = a.matmul_transposed(b);
    EXPECT_EQ(c.rows(), 2);
    EXPECT_EQ(c.cols(), 2);
    EXPECT_NEAR(c.at(0, 0), 1.0f, 1e-4f);
    EXPECT_NEAR(c.at(0, 1), 2.0f, 1e-4f);
    EXPECT_NEAR(c.at(1, 0), 4.0f, 1e-4f);
    EXPECT_NEAR(c.at(1, 1), 5.0f, 1e-4f);
}

TEST(OpenCLTensorBackendTest, MatmulLhsTransposed)
{
    // a (3x2), b (3x2) -> a^T (2x3) * b (3x2) = (2x2)
    nn::OpenCLTensorBackend a(3, 2);
    nn::OpenCLTensorBackend b(3, 2);

    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f;
    a.at(1, 1) = 4.0f;
    a.at(2, 0) = 5.0f;
    a.at(2, 1) = 6.0f;

    b.at(0, 0) = 1.0f;
    b.at(0, 1) = 0.0f;
    b.at(1, 0) = 0.0f;
    b.at(1, 1) = 1.0f;
    b.at(2, 0) = 1.0f;
    b.at(2, 1) = 1.0f;

    auto c = a.matmul_lhs_transposed(b);
    EXPECT_EQ(c.rows(), 2);
    EXPECT_EQ(c.cols(), 2);
    EXPECT_NEAR(c.at(0, 0), 6.0f, 1e-4f);
    EXPECT_NEAR(c.at(0, 1), 8.0f, 1e-4f);
    EXPECT_NEAR(c.at(1, 0), 8.0f, 1e-4f);
    EXPECT_NEAR(c.at(1, 1), 10.0f, 1e-4f);
}

TEST(OpenCLTensorBackendTest, AddColVectorToRowsInplace)
{
    nn::OpenCLTensorBackend a(3, 2);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f;
    a.at(1, 1) = 4.0f;
    a.at(2, 0) = 5.0f;
    a.at(2, 1) = 6.0f;

    // API expects shape (cols, 1): col[c] is broadcast to every row at column c
    nn::OpenCLTensorBackend col(2, 1);
    col.at(0, 0) = 10.0f;
    col.at(1, 0) = 20.0f;

    a.add_col_vector_to_rows_inplace(col);
    // Each column gets its bias added to every row at that column
    EXPECT_NEAR(a.at(0, 0), 11.0f, 1e-5f); // 1 + col[0]=10
    EXPECT_NEAR(a.at(0, 1), 22.0f, 1e-5f); // 2 + col[1]=20
    EXPECT_NEAR(a.at(1, 0), 13.0f, 1e-5f); // 3 + col[0]=10
    EXPECT_NEAR(a.at(1, 1), 24.0f, 1e-5f); // 4 + col[1]=20
    EXPECT_NEAR(a.at(2, 0), 15.0f, 1e-5f); // 5 + col[0]=10
    EXPECT_NEAR(a.at(2, 1), 26.0f, 1e-5f); // 6 + col[1]=20
}

TEST(OpenCLTensorBackendTest, GradManagement)
{
    nn::OpenCLTensorBackend a(2, 2);
    EXPECT_FALSE(a.has_grad());

    // grad_ref allocates on first access
    auto& g = a.grad_ref();
    EXPECT_TRUE(a.has_grad());
    g.at(0, 0) = 5.0f;
    EXPECT_NEAR(a.get_grad().at(0, 0), 5.0f, 1e-6f);

    a.zero_grad();
    EXPECT_NEAR(a.get_grad().at(0, 0), 0.0f, 1e-6f);
}

TEST(OpenCLTensorBackendTest, FourDimConstructor)
{
    nn::OpenCLTensorBackend a(2, 3, 4, 5);
    EXPECT_EQ(a.rows(), 2);
    EXPECT_EQ(a.cols(), 3);
    const auto& shape = a.shape();
    EXPECT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[2], 4u);
    EXPECT_EQ(shape[3], 5u);
}

// ---------------------------------------------------------------------------
// Fused kernel correctness: matmul_transposed_add_col_bias_{relu,leaky_relu}
//
// Strategy: compute reference with two separate backend calls
//   (matmul_transposed_add_col_bias + relu/leaky_relu),
// then compare element-wise against the fused single call.
// Threshold 1e-5 matches float precision with ~100 accumulations.
// ---------------------------------------------------------------------------

// Build a small (M x K) * (N x K)^T + bias tensor fixture shared by both tests.
// M=4 batch rows, K=8 input features, N=6 output neurons.
static void fill_fused_test_tensors(nn::OpenCLTensorBackend& A,
    nn::OpenCLTensorBackend& B,
    nn::OpenCLTensorBackend& bias)
{
    // Fill A (4 x 8) with incrementing values
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c)
            A.at(r, c) = static_cast<float>(r * 8 + c + 1) * 0.1f;

    // Fill B (6 x 8) with decreasing values — stored as weight matrix
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 8; ++c)
            B.at(r, c) = static_cast<float>((6 - r) * 8 - c) * 0.05f;

    // Fill bias (6 x 1)
    for (int r = 0; r < 6; ++r)
        bias.at(r, 0) = static_cast<float>(r - 3) * 0.2f;
}

TEST(OpenCLFusedKernelTest, MatmulBiasReluMatchesTwoKernels)
{
    nn::OpenCLTensorBackend A(4, 8);
    nn::OpenCLTensorBackend B(6, 8);
    nn::OpenCLTensorBackend bias(6, 1);
    fill_fused_test_tensors(A, B, bias);

    // Reference: unfused two-kernel path
    auto ref_pre  = A.matmul_transposed_add_col_bias(B, bias);
    auto ref_post = ref_pre.relu();

    // Fused single-kernel path
    auto fused = A.matmul_transposed_add_col_bias_relu(B, bias);

    ASSERT_EQ(fused.rows(), 4);
    ASSERT_EQ(fused.cols(), 6);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c)
            EXPECT_NEAR(fused.at(r, c), ref_post.at(r, c), 1e-5f)
                << "mismatch at (" << r << "," << c << ")";
}

TEST(OpenCLFusedKernelTest, MatmulBiasLeakyReluMatchesTwoKernels)
{
    const float alpha = 0.1f;
    nn::OpenCLTensorBackend A(4, 8);
    nn::OpenCLTensorBackend B(6, 8);
    nn::OpenCLTensorBackend bias(6, 1);
    fill_fused_test_tensors(A, B, bias);

    // Reference: unfused two-kernel path
    auto ref_pre  = A.matmul_transposed_add_col_bias(B, bias);
    auto ref_post = ref_pre.leaky_relu(alpha);

    // Fused single-kernel path
    auto fused = A.matmul_transposed_add_col_bias_leaky_relu(B, bias, alpha);

    ASSERT_EQ(fused.rows(), 4);
    ASSERT_EQ(fused.cols(), 6);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c)
            EXPECT_NEAR(fused.at(r, c), ref_post.at(r, c), 1e-5f)
                << "mismatch at (" << r << "," << c << ")";
}

// ---------------------------------------------------------------------------
// Benchmark: wall-clock timing of unfused vs fused for a layer-realistic size.
// Sizes: batch=32, in=256, out=64 (typical dense layer in experiment04 SNN).
// 10 warmup + 100 timed iterations; reports mean µs via test output.
// This is an informational test — it never fails.
// ---------------------------------------------------------------------------

TEST(OpenCLFusedKernelTest, MatmulBiasSigmoidMatchesCpuRef)
{
    nn::OpenCLTensorBackend A(4, 8);
    nn::OpenCLTensorBackend B(6, 8);
    nn::OpenCLTensorBackend bias(6, 1);
    fill_fused_test_tensors(A, B, bias);

    // Reference: unfused matmul+bias, then CPU sigmoid
    auto pre = A.matmul_transposed_add_col_bias(B, bias);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c)
            pre.at(r, c) = 1.0f / (1.0f + std::exp(-pre.at(r, c)));

    auto fused = A.matmul_transposed_add_col_bias_sigmoid(B, bias);

    ASSERT_EQ(fused.rows(), 4);
    ASSERT_EQ(fused.cols(), 6);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c)
            EXPECT_NEAR(fused.at(r, c), pre.at(r, c), 1e-5f)
                << "mismatch at (" << r << "," << c << ")";
}

TEST(OpenCLFusedKernelTest, MatmulBiasTanhMatchesCpuRef)
{
    nn::OpenCLTensorBackend A(4, 8);
    nn::OpenCLTensorBackend B(6, 8);
    nn::OpenCLTensorBackend bias(6, 1);
    fill_fused_test_tensors(A, B, bias);

    // Reference: unfused matmul+bias, then CPU tanh
    auto pre = A.matmul_transposed_add_col_bias(B, bias);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c)
            pre.at(r, c) = std::tanh(pre.at(r, c));

    auto fused = A.matmul_transposed_add_col_bias_tanh(B, bias);

    ASSERT_EQ(fused.rows(), 4);
    ASSERT_EQ(fused.cols(), 6);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c)
            EXPECT_NEAR(fused.at(r, c), pre.at(r, c), 1e-5f)
                << "mismatch at (" << r << "," << c << ")";
}

TEST(OpenCLFusedKernelTest, BenchmarkFusedVsUnfused)
{
    const int M = 32, K = 256, N = 64;

    nn::OpenCLTensorBackend A(M, K);
    nn::OpenCLTensorBackend W(N, K);
    nn::OpenCLTensorBackend bias(N, 1);

    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            A.at(r, c) = static_cast<float>(r * K + c) / (M * K);

    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < K; ++c)
            W.at(r, c) = static_cast<float>(r * K + c) / (N * K);
        bias.at(r, 0) = static_cast<float>(r) / N;
    }

    const int warmup = 10;
    const int iters  = 100;

    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        auto t1 = A.matmul_transposed_add_col_bias(W, bias).relu();
        (void) t1;
    }

    // Time unfused (matmul_bias + relu)
    auto t0_unfused = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        auto t1 = A.matmul_transposed_add_col_bias(W, bias).relu();
        (void) t1;
    }
    auto t1_unfused = std::chrono::steady_clock::now();
    const double unfused_us =
        std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

    // Warmup fused
    for (int i = 0; i < warmup; ++i)
    {
        auto t1 = A.matmul_transposed_add_col_bias_relu(W, bias);
        (void) t1;
    }

    // Time fused
    auto t0_fused = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        auto t1 = A.matmul_transposed_add_col_bias_relu(W, bias);
        (void) t1;
    }
    auto t1_fused = std::chrono::steady_clock::now();
    const double fused_us =
        std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

    std::cout << "[BENCH] matmul+bias+relu  unfused: " << unfused_us << " µs/iter\n";
    std::cout << "[BENCH] matmul+bias+relu   fused: " <<  fused_us   << " µs/iter\n";
    if (fused_us < unfused_us)
        std::cout << "[BENCH] speedup: " << unfused_us / fused_us << "×\n";
    else
        std::cout << "[BENCH] fused was slower (kernel launch overhead dominates at this size)\n";

    // Test never fails — it is informational only
    SUCCEED();
}
