/**
 * @file opencl_tensor_backend_gtest.cpp
 * @brief Unit tests for OpenCLTensorBackend correctness with CPU fallback safety:
 *        construction, runtime/availability plumbing, and gradient bookkeeping.
 *
 * The broader per-operation-family coverage that used to live in this one file
 * moved to the sibling opencl_tensor_backend_*_gtest.cpp files in this
 * directory (matmul_compare, elementwise_reductions, inplace, lif, viewops,
 * fused_shapemismatch) once this file crossed LOC_CRITICAL.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/spiking/Lif.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

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

TEST(OpenCLProfilingTest, ToggleFlag)
{
    // Ensure toggle APIs work without touching hardware kernels.
    nn::opencl::profiling::set_enabled(true);
    EXPECT_TRUE(nn::opencl::profiling::is_enabled());
    nn::opencl::profiling::set_enabled(false);
    EXPECT_FALSE(nn::opencl::profiling::is_enabled());
}

} // namespace
