/**
 * @file opencl_tensor_backend_matmul_compare_gtest.cpp
 * @brief Matmul family and comparison-op correctness on the OpenCL backend.
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

TEST(OpenCLTensorBackendTest, CompareLtBroadcastsARowOrColumnOfOne)
{
    // Nothing exercised compare_lt's broadcast path before this: every other
    // TEST(...CompareOps) used same-shape operands. Added when the branch was
    // pulled out into compare_lt_broadcast(), to confirm the extraction did
    // not change what it computes.
    nn::OpenCLTensorBackend a(2, 3);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 5.0f;
    a.at(0, 2) = 9.0f;
    a.at(1, 0) = 1.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 9.0f;

    nn::OpenCLTensorBackend row_threshold(1, 3);
    row_threshold.at(0, 0) = 2.0f;
    row_threshold.at(0, 1) = 2.0f;
    row_threshold.at(0, 2) = 2.0f;

    auto lt = a.compare_lt(row_threshold); // (2,3) < (1,3), row broadcasts down
    EXPECT_FLOAT_EQ(lt.at(0, 0), 1.0f);    // 1 < 2
    EXPECT_FLOAT_EQ(lt.at(0, 1), 0.0f);    // 5 < 2 -> false
    EXPECT_FLOAT_EQ(lt.at(1, 0), 1.0f);
}

} // namespace
