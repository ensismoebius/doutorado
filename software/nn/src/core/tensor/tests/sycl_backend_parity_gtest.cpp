/**
 * @file sycl_backend_parity_gtest.cpp
 * @brief Numerical parity tests: XTensor (CPU reference) vs SYCL backend.
 *
 * Mirrors backend_parity_gtest.cpp (XTensor vs OpenCL): identical deterministic
 * inputs on both backends, same op, element-by-element comparison. Guards the
 * SYCL kernels against drifting from the reference implementation.
 *
 * The suite runs even without a SYCL device: SYCLTensorBackend falls back to
 * host execution, in which case parity is trivially exact and the suite
 * degenerates to a smoke test of the wrapper plumbing. The first test logs
 * which mode is active.
 */

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "tensor/sycl/SYCLTensorBackend.hpp"
#include "tensor/xtensor/XTensorBackend.hpp"

namespace
{

using XT = nn::XTensorBackend;
using SY = nn::SYCLTensorBackend;

// Device fp32 kernels round differently from the host serial loops; chained
// reductions accumulate that. Same tolerances as the OpenCL parity suite.
constexpr float kOpTol = 2e-4f;

/// Fill both tensors with the same deterministic values in [lo, hi].
void fill_pair(XT& a, SY& b, unsigned seed, float lo = -1.0f, float hi = 1.0f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (nn::Index i = 0; i < a.rows(); ++i)
        for (nn::Index j = 0; j < a.cols(); ++j)
        {
            const float v = dist(rng);
            a.at(i, j) = v;
            b.at(i, j) = v;
        }
}

void expect_parity(const SY& got, const XT& want, float tol, const char* what)
{
    ASSERT_EQ(got.shape(), want.shape()) << what;
    for (nn::Index i = 0; i < want.size(); ++i)
        EXPECT_NEAR(got.at(i), want.at(i), tol) << what << " at flat index " << i;
}

TEST(SyclBackendParityTest, ReportsRuntimeMode)
{
    // Informational: which device (or host fallback) the suite exercises.
    std::cout << "[SYCL] runtime available: " << std::boolalpha << SY::sycl_runtime_available()
              << ", device: " << SY::device_description() << "\n";
    SUCCEED();
}

TEST(SyclBackendParityTest, ElementwiseBinaryOps)
{
    XT ax(5, 7), bx(5, 7);
    SY as(5, 7), bs(5, 7);
    fill_pair(ax, as, 11);
    fill_pair(bx, bs, 22, 0.5f, 2.0f); // divisor away from zero

    expect_parity(as.add(bs), ax.add(bx), kOpTol, "add");
    expect_parity(as.subtract(bs), ax.subtract(bx), kOpTol, "subtract");
    expect_parity(as.multiply(bs), ax.multiply(bx), kOpTol, "multiply");
    expect_parity(as.divide(bs), ax.divide(bx), kOpTol, "divide");

    SY inpl = as;
    XT inplx = ax;
    inpl.add_inplace(bs);
    inplx.add_inplace(bx);
    expect_parity(inpl, inplx, kOpTol, "add_inplace");
}

TEST(SyclBackendParityTest, ScalarOps)
{
    XT ax(6, 4);
    SY as(6, 4);
    fill_pair(ax, as, 31);

    expect_parity(as.add_scalar(1.5f), ax.add_scalar(1.5f), kOpTol, "add_scalar");
    expect_parity(
        as.multiply_scalar(-2.25f), ax.multiply_scalar(-2.25f), kOpTol, "multiply_scalar");
    expect_parity(as.divide_scalar(3.0f), ax.divide_scalar(3.0f), kOpTol, "divide_scalar");
}

TEST(SyclBackendParityTest, UnaryOps)
{
    XT ax(6, 9);
    SY as(6, 9);
    fill_pair(ax, as, 41, 0.1f, 2.0f); // positive domain for sqrt

    expect_parity(as.exp(), ax.exp(), kOpTol, "exp");
    expect_parity(as.sqrt(), ax.sqrt(), kOpTol, "sqrt");
    expect_parity(as.square(), ax.square(), kOpTol, "square");

    XT sx(6, 9);
    SY ss(6, 9);
    fill_pair(sx, ss, 42); // signed domain for abs/relu/leaky
    expect_parity(ss.abs(), sx.abs(), kOpTol, "abs");
    expect_parity(ss.relu(), sx.relu(), kOpTol, "relu");
    expect_parity(ss.leaky_relu(0.01f), sx.leaky_relu(0.01f), kOpTol, "leaky_relu");
}

TEST(SyclBackendParityTest, MatmulFamily)
{
    XT ax(7, 5), bx(5, 3);
    SY as(7, 5), bs(5, 3);
    fill_pair(ax, as, 51);
    fill_pair(bx, bs, 52);
    expect_parity(as.matmul(bs), ax.matmul(bx), kOpTol, "matmul");

    XT btx(3, 5);
    SY bts(3, 5);
    fill_pair(btx, bts, 53);
    expect_parity(
        as.matmul_transposed(bts), ax.matmul_transposed(btx), kOpTol, "matmul_transposed");

    expect_parity(as.transpose(), ax.transpose(), kOpTol, "transpose");
}

TEST(SyclBackendParityTest, MatmulShapeMismatchThrows)
{
    SY a(4, 3), b(4, 3); // inner dims disagree for plain matmul
    EXPECT_THROW((void) a.matmul(b), std::invalid_argument);

    SY c(4, 3), d(5, 4); // cols differ for matmul_transposed
    EXPECT_THROW((void) c.matmul_transposed(d), std::invalid_argument);
}

TEST(SyclBackendParityTest, Reductions)
{
    XT ax(8, 6), bx(8, 6);
    SY as(8, 6), bs(8, 6);
    fill_pair(ax, as, 61);
    fill_pair(bx, bs, 62);

    EXPECT_NEAR(as.sum(), ax.sum(), kOpTol * static_cast<float>(ax.size())) << "sum";
    EXPECT_NEAR(as.norm(), ax.norm(), kOpTol * 10.0f) << "norm";
    EXPECT_NEAR(as.mean_squared_error(bs), ax.mean_squared_error(bx), kOpTol) << "mse";

    expect_parity(as.sum_rows(), ax.sum_rows(), kOpTol * 10.0f, "sum_rows");
    expect_parity(as.sum_cols(), ax.sum_cols(), kOpTol * 10.0f, "sum_cols");
    expect_parity(as.rowwise_sum(), ax.rowwise_sum(), kOpTol * 10.0f, "rowwise_sum");
}

TEST(SyclBackendParityTest, HostDelegatedOpsStayCoherent)
{
    // Slicing/blocks/grad go through the host mirror; verify the wrapper keeps
    // them coherent with accelerated results.
    XT ax(6, 8);
    SY as(6, 8);
    fill_pair(ax, as, 71);

    expect_parity(as.block(1, 2, 3, 4), ax.block(1, 2, 3, 4), 0.0f, "block");
    expect_parity(as.row(2), ax.row(2), 0.0f, "row");

    // Accelerated result feeding a host-delegated op.
    expect_parity(as.relu().topRows(3), ax.relu().topRows(3), kOpTol, "relu→topRows");

    // Gradient plumbing.
    SY g(6, 8);
    XT gx(6, 8);
    fill_pair(gx, g, 72);
    as.set_grad(g);
    ax.set_grad(gx);
    expect_parity(as.get_grad(), ax.get_grad(), 0.0f, "set/get_grad");
}

} // namespace
