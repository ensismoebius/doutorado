/**
 * @file opencl_tensor_backend_elementwise_reductions_gtest.cpp
 * @brief Elementwise, scalar, and reduction-op correctness on the OpenCL backend.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/spiking/Lif.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

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

} // namespace
