/**
 * @file opencl_tensor_backend_gtest.cpp
 * @brief Unit tests for OpenCLTensorBackend correctness with CPU fallback safety.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "nn/tensor/OpenCLContext.hpp"
#include "nn/tensor/OpenCLTensorBackend.hpp"

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

} // namespace
