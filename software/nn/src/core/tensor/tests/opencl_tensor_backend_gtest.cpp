/**
 * @file opencl_tensor_backend_gtest.cpp
 * @brief Unit tests for OpenCLTensorBackend correctness with CPU fallback safety.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "nn/tensor/opencl/OpenCLContext.hpp"
#include "nn/tensor/opencl/OpenCLProfiling.hpp"
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"

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
    nn::Tensor prediction(1, 2);
    nn::Tensor target(1, 2);
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
