/**
 * @file opencl_tensor_backend_fused_shapemismatch_gtest.cpp
 * @brief Fused-kernel correctness and the no-fallback-policy shape-mismatch regression suite (every
 * op refuses bad shapes unconditionally, not just when OpenCL is unavailable).
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/spiking/Lif.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

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
static void fill_fused_test_tensors(
    nn::OpenCLTensorBackend& A, nn::OpenCLTensorBackend& B, nn::OpenCLTensorBackend& bias)
{
    // Fill A (4 x 8) with incrementing values
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c) A.at(r, c) = static_cast<float>(r * 8 + c + 1) * 0.1f;

    // Fill B (6 x 8) with decreasing values — stored as weight matrix
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 8; ++c) B.at(r, c) = static_cast<float>((6 - r) * 8 - c) * 0.05f;

    // Fill bias (6 x 1)
    for (int r = 0; r < 6; ++r) bias.at(r, 0) = static_cast<float>(r - 3) * 0.2f;
}

TEST(OpenCLFusedKernelTest, MatmulBiasReluMatchesTwoKernels)
{
    nn::OpenCLTensorBackend A(4, 8);
    nn::OpenCLTensorBackend B(6, 8);
    nn::OpenCLTensorBackend bias(6, 1);
    fill_fused_test_tensors(A, B, bias);

    // Reference: unfused two-kernel path
    auto ref_pre = A.matmul_transposed_add_col_bias(B, bias);
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
    auto ref_pre = A.matmul_transposed_add_col_bias(B, bias);
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
// Sizes: batch=32, in=256, out=64 (typical dense layer in guayaquil SNN).
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
        for (int c = 0; c < 6; ++c) pre.at(r, c) = 1.0f / (1.0f + std::exp(-pre.at(r, c)));

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
        for (int c = 0; c < 6; ++c) pre.at(r, c) = std::tanh(pre.at(r, c));

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
        for (int c = 0; c < K; ++c) A.at(r, c) = static_cast<float>(r * K + c) / (M * K);

    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < K; ++c) W.at(r, c) = static_cast<float>(r * K + c) / (N * K);
        bias.at(r, 0) = static_cast<float>(r) / N;
    }

    const int warmup = 10;
    const int iters = 100;

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
    std::cout << "[BENCH] matmul+bias+relu   fused: " << fused_us << " µs/iter\n";
    if (fused_us < unfused_us)
        std::cout << "[BENCH] speedup: " << unfused_us / fused_us << "×\n";
    else
        std::cout << "[BENCH] fused was slower (kernel launch overhead dominates at this size)\n";

    // Test never fails — it is informational only
    SUCCEED();
}

// ─── Shape-mismatch contract ────────────────────────────────────────────────
// Feeding two differently shaped tensors to an elementwise op is a caller
// error, not a runtime condition: 2x3 + 3x2 has no meaning to compute. The
// backend must refuse it. These tests exist because the refusal used to be
// SPLIT — add/multiply raised std::invalid_argument naming the real cause,
// while subtract/divide/compare_* raised std::runtime_error saying "OpenCL
// runtime unavailable or tensor shape mismatch", which forces the reader to
// guess which of two unrelated causes fired. Nothing pinned either shape, so
// the split survived unnoticed. One contract now: std::invalid_argument,
// message naming the operation and the cause.
//
// Note these run identically with or without an OpenCL device: the guard
// fires before any CL call.

TEST(OpenCLTensorBackendShapeMismatch, BinaryOpsRefuseMismatchedShapes)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend b(3, 2); // same element count, different shape

    EXPECT_THROW((void) a.add(b), std::invalid_argument);
    EXPECT_THROW((void) a.subtract(b), std::invalid_argument);
    EXPECT_THROW((void) a.multiply(b), std::invalid_argument);
    EXPECT_THROW((void) a.divide(b), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, MessageNamesTheOperationAndTheCause)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend b(3, 2);

    // The message is the whole point of the contract: a reader must be able to
    // tell WHICH op refused and WHY without attaching a debugger.
    try
    {
        (void) a.subtract(b);
        FAIL() << "subtract accepted mismatched shapes";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("subtract"), std::string::npos) << message;
        EXPECT_NE(message.find("shape"), std::string::npos) << message;
    }
}

TEST(OpenCLTensorBackendShapeMismatch, MatchingShapesAreNotRefused)
{
    // Guards the guard: a same-shape pair must reach the kernel, so a future
    // tightening of the check cannot pass these tests by refusing everything.
    nn::OpenCLTensorBackend a(2, 3);
    nn::OpenCLTensorBackend b(2, 3);
    a.fill(2.0f);
    b.fill(3.0f);

    EXPECT_NO_THROW({
        const nn::OpenCLTensorBackend sum = a.add(b);
        EXPECT_FLOAT_EQ(sum.at(0, 0), 5.0f);
    });
}

TEST(OpenCLTensorBackendShapeMismatch, InPlaceOpsRefuseMismatchedShapes)
{
    // Same contract as the out-of-place ops. These used to reach
    // throw_opencl_only_failure with "OpenCL runtime unavailable or tensor
    // shape mismatch", which is two causes in one message.
    nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend b(3, 2);

    EXPECT_THROW(a.add_inplace(b), std::invalid_argument);
    EXPECT_THROW(a.subtract_inplace(b), std::invalid_argument);
    EXPECT_THROW(a.multiply_inplace(b), std::invalid_argument);
    EXPECT_THROW(a.divide_inplace(b), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, InPlaceOpsWithMatchingShapesStillCompute)
{
    // Guards the guard, as above: refusing everything must not pass.
    nn::OpenCLTensorBackend a(2, 3);
    nn::OpenCLTensorBackend b(2, 3);
    a.fill(10.0f);
    b.fill(4.0f);

    a.subtract_inplace(b);
    EXPECT_FLOAT_EQ(a.at(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(a.at(1, 2), 6.0f);
}

// ─── Matmul family shape-mismatch contract ──────────────────────────────────
// Same story as the elementwise family: matmul() refused a dimension
// mismatch with std::invalid_argument naming the real cause, while
// matmul_transposed()/matmul_lhs_transposed() refused the identical class of
// caller error with std::runtime_error saying "OpenCL runtime unavailable or
// matrix dimensions are invalid" -- forcing the reader to guess which of two
// unrelated causes fired. Nothing pinned either shape, so the split survived
// unnoticed. One contract now: std::invalid_argument, unconditionally (the
// check runs before any OpenCL availability probe), message naming the
// operation.

TEST(OpenCLTensorBackendShapeMismatch, MatmulFamilyRefusesIncompatibleDimensions)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend incompatible(4, 5); // no interpretation of a*x makes this work

    EXPECT_THROW((void) a.matmul(incompatible), std::invalid_argument);
    EXPECT_THROW((void) a.matmul_transposed(incompatible), std::invalid_argument);
    EXPECT_THROW((void) a.matmul_lhs_transposed(incompatible), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, MatmulFamilyMessageNamesTheOperation)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend incompatible(4, 5);

    try
    {
        (void) a.matmul_transposed(incompatible);
        FAIL() << "matmul_transposed accepted incompatible dimensions";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("matmul_transposed"), std::string::npos) << message;
    }

    try
    {
        (void) a.matmul_lhs_transposed(incompatible);
        FAIL() << "matmul_lhs_transposed accepted incompatible dimensions";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("matmul_lhs_transposed"), std::string::npos) << message;
    }
}

TEST(OpenCLTensorBackendShapeMismatch, MatmulFamilyRefusesUnconditionallyNotJustWithoutOpenCL)
{
    // The shape guard must fire before any device-availability check, so a
    // caller sees the SAME exception whether or not OpenCL happens to be
    // present. Guards against "fixing" the type by moving the check inside
    // the can_use_opencl() branch, which would silently reintroduce a
    // runtime_error path when a device is unavailable.
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend incompatible(4, 5);
    ASSERT_TRUE(nn::opencl::OpenCLContext::instance().is_available())
        << "this test assumes the suite runs on a real (possibly software) OpenCL device";

    EXPECT_THROW((void) a.matmul_transposed(incompatible), std::invalid_argument);
    EXPECT_THROW((void) a.matmul_lhs_transposed(incompatible), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, CompareFamilyRefusesMismatchedShapes)
{
    // compare_gt/le/ge/eq shared the same split as add/subtract: refused a
    // shape mismatch with std::runtime_error ("OpenCL runtime unavailable or
    // tensor shape mismatch") instead of std::invalid_argument, conflating a
    // caller error with device unavailability. compare_lt is unaffected --
    // it already threw invalid_argument for its own (broadcast) shape rule.
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend b(3, 2);

    EXPECT_THROW((void) a.compare_gt(b), std::invalid_argument);
    EXPECT_THROW((void) a.compare_le(b), std::invalid_argument);
    EXPECT_THROW((void) a.compare_ge(b), std::invalid_argument);
    EXPECT_THROW((void) a.compare_eq(b), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, CompareFamilyMatchingShapesStillCompute)
{
    // Guards the guard: refusing everything must not pass.
    nn::OpenCLTensorBackend a(2, 2);
    nn::OpenCLTensorBackend b(2, 2);
    a.fill(1.0f);
    b.fill(2.0f);

    auto gt = a.compare_gt(b);
    EXPECT_FLOAT_EQ(gt.at(0, 0), 0.0f);
    auto le = a.compare_le(b);
    EXPECT_FLOAT_EQ(le.at(0, 0), 1.0f);
}

TEST(OpenCLTensorBackendShapeMismatch, RowwiseSumRefusesNonRank2)
{
    // rowwise_sum() previously refused a rank != 2 tensor with
    // warn_opencl_cpu_fallback_once + an unconditional std::runtime_error
    // ("OpenCL runtime unavailable or tensor rank is invalid") -- the same
    // ambiguous-cause split fixed for transpose()/matmul_transposed() above.
    nn::OpenCLTensorBackend rank1(std::vector<nn::Index>{5});

    EXPECT_THROW((void) rank1.rowwise_sum(), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, RowwiseSumRank2StillComputes)
{
    nn::OpenCLTensorBackend a(2, 3);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(1, 0) = 4.0f;
    a.at(1, 1) = 5.0f;
    a.at(1, 2) = 6.0f;

    auto sums = a.rowwise_sum();
    EXPECT_FLOAT_EQ(sums.at(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(sums.at(1, 0), 15.0f);
}

TEST(OpenCLTensorBackendShapeMismatch, CompareLtRefusesNonBroadcastableShapes)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend incompatible(2, 5); // neither side is 1, and 3 != 5

    EXPECT_THROW((void) a.compare_lt(incompatible), std::invalid_argument);
}

// matmul_transposed_add_col_bias{,_relu,_sigmoid,_tanh,_leaky_relu} all
// funnel through the shared matmul_transposed_bias_activated(); its shape
// guard used to throw the ambiguous runtime_error under the literal name
// "matmul_transposed_add_col_bias_leaky_relu" no matter which of the 5
// callers hit it. Fixed to std::invalid_argument, named for the actual
// caller.
TEST(OpenCLTensorBackendShapeMismatch, BiasFamilyRefusesMismatchedShapes)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend other(4, 5); // other.cols() != a.cols()
    const nn::OpenCLTensorBackend bias(4, 1);

    EXPECT_THROW((void) a.matmul_transposed_add_col_bias(other, bias), std::invalid_argument);
    EXPECT_THROW((void) a.matmul_transposed_add_col_bias_relu(other, bias), std::invalid_argument);
    EXPECT_THROW(
        (void) a.matmul_transposed_add_col_bias_sigmoid(other, bias), std::invalid_argument);
    EXPECT_THROW((void) a.matmul_transposed_add_col_bias_tanh(other, bias), std::invalid_argument);
    EXPECT_THROW((void) a.matmul_transposed_add_col_bias_leaky_relu(other, bias, 0.1F),
        std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, BiasFamilyRefusesMismatchedBiasShape)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend other(4, 3);
    const nn::OpenCLTensorBackend wrong_bias(5, 1); // wrong_bias.rows() != other.rows()

    EXPECT_THROW((void) a.matmul_transposed_add_col_bias(other, wrong_bias), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, BiasFamilyMessageNamesTheActualCaller)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend other(4, 5);
    const nn::OpenCLTensorBackend bias(4, 1);

    try
    {
        (void) a.matmul_transposed_add_col_bias_sigmoid(other, bias);
        FAIL() << "matmul_transposed_add_col_bias_sigmoid accepted incompatible dimensions";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("matmul_bias_sigmoid"), std::string::npos) << message;
        EXPECT_EQ(message.find("leaky_relu"), std::string::npos) << message;
    }
}

TEST(OpenCLTensorBackendShapeMismatch, BiasFamilyRefusesUnconditionallyNotJustWithoutOpenCL)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend other(4, 5);
    const nn::OpenCLTensorBackend bias(4, 1);
    ASSERT_TRUE(nn::opencl::OpenCLContext::instance().is_available())
        << "this test assumes the suite runs on a real (possibly software) OpenCL device";

    EXPECT_THROW((void) a.matmul_transposed_add_col_bias(other, bias), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, BiasFamilyMatchingShapesStillCompute)
{
    const nn::OpenCLTensorBackend a(2, 3);
    const nn::OpenCLTensorBackend other(4, 3);
    const nn::OpenCLTensorBackend bias(4, 1);

    EXPECT_NO_THROW((void) a.matmul_transposed_add_col_bias(other, bias));
    EXPECT_NO_THROW((void) a.matmul_transposed_add_col_bias_relu(other, bias));
}

// add_row_broadcast_inplace() and add_col_vector_to_rows_inplace() are now
// unified behind run_broadcast_vector_stages() -- both raise identically on
// shape mismatch and on device unavailability. add_row_broadcast_inplace
// previously fell back to a CPU loop instead of raising; that fallback is
// gone.
TEST(OpenCLTensorBackendShapeMismatch, BroadcastVectorFamilyRefusesMismatchedShapes)
{
    nn::OpenCLTensorBackend a(3, 2);
    const nn::OpenCLTensorBackend wrong_row(1, 5); // wrong_row.cols() != a.cols()
    const nn::OpenCLTensorBackend wrong_col(5, 1); // wrong_col.rows() != a.cols()

    EXPECT_THROW(a.add_row_broadcast_inplace(wrong_row), std::invalid_argument);
    EXPECT_THROW(a.add_col_vector_to_rows_inplace(wrong_col), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch,
    BroadcastVectorFamilyRefusesUnconditionallyNotJustWithoutOpenCL)
{
    nn::OpenCLTensorBackend a(3, 2);
    const nn::OpenCLTensorBackend wrong_row(1, 5);
    ASSERT_TRUE(nn::opencl::OpenCLContext::instance().is_available())
        << "this test assumes the suite runs on a real (possibly software) OpenCL device";

    EXPECT_THROW(a.add_row_broadcast_inplace(wrong_row), std::invalid_argument);
}

TEST(OpenCLTensorBackendShapeMismatch, BroadcastVectorFamilyMatchingShapesStillCompute)
{
    nn::OpenCLTensorBackend a(3, 2);
    const nn::OpenCLTensorBackend row(1, 2);
    const nn::OpenCLTensorBackend col(2, 1);

    EXPECT_NO_THROW(a.add_row_broadcast_inplace(row));
    EXPECT_NO_THROW(a.add_col_vector_to_rows_inplace(col));
}

} // namespace
