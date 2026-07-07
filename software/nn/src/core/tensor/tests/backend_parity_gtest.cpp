/**
 * @file backend_parity_gtest.cpp
 * @brief Numerical parity tests: XTensor (CPU) vs OpenCL backend.
 *
 * Every test builds the SAME deterministic inputs on both backends, runs the
 * SAME operation through the shared TensorImpl / layer templates, and compares
 * results element-by-element. This guards against the two implementations
 * silently diverging (e.g. layout bugs — OpenCL is column-major internally,
 * XTensor row-major — or fast-path kernels like lif_step_inplace drifting from
 * the generic fallback).
 *
 * Coverage:
 *   - elementwise ops, matmul family, reductions, slicing/reshape
 *   - Linear layer forward/backward (incl. weight/bias gradients)
 *   - Lif spiking layer multi-step state evolution + backward (incl. R/C/V_th
 *     gradients) — batched, which exercises the OpenCL lif_step_inplace fast
 *     path against the XTensor generic path
 *   - LifIntegrator readout layer forward/backward
 *   - an SNN-autoencoder-shaped chain (Linear→Lif→Linear→LifIntegrator)
 *     end-to-end forward + backward
 *
 * The whole suite skips when no OpenCL device is available.
 */

#include <gtest/gtest.h>

#include <random>
#include <utility>
#include <vector>

#include "layers/dense/Linear.hpp"
#include "layers/spiking/Lif.hpp"
#include "layers/spiking/LifIntegrator.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

using XT = nn::XTensorBackend;
using CL = nn::OpenCLTensorBackend;
using TX = nn::TensorImpl<XT>;
using TC = nn::TensorImpl<CL>;

// GPU fp32 kernels round differently from CPU; chained ops accumulate that.
constexpr float kOpTol = 2e-4f;
constexpr float kChainTol = 5e-4f;

/// Fill both tensors with the same deterministic values in [lo, hi].
template <typename A, typename B>
void fill_pair(A& a, B& b, unsigned seed, float lo = -1.0f, float hi = 1.0f)
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

/// Element-by-element comparison through the backend-agnostic at(i, j).
template <typename A, typename B>
void expect_parity(const A& x, const B& c, float tol, const char* what)
{
    ASSERT_EQ(x.rows(), c.rows()) << what << ": row mismatch";
    ASSERT_EQ(x.cols(), c.cols()) << what << ": col mismatch";
    for (nn::Index i = 0; i < x.rows(); ++i)
        for (nn::Index j = 0; j < x.cols(); ++j)
            EXPECT_NEAR(x.at(i, j), c.at(i, j), tol)
                << what << " diverges at (" << i << "," << j << ")";
}

class BackendParityTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!nn::opencl::OpenCLContext::instance().is_available())
            GTEST_SKIP() << "No OpenCL device available — parity suite skipped";
    }
};

// ─── tensor ops ──────────────────────────────────────────────────────────────

TEST_F(BackendParityTest, ElementwiseOps)
{
    // Odd, non-square shape catches row-major/column-major layout bugs.
    TX ax(5, 7), bx(5, 7);
    TC ac(5, 7), bc(5, 7);
    fill_pair(ax, ac, 11);
    fill_pair(bx, bc, 22, 0.5f, 2.0f); // divisor kept away from zero

    expect_parity(ax.add(bx), ac.add(bc), kOpTol, "add");
    expect_parity(ax - bx, ac - bc, kOpTol, "subtract");
    expect_parity(ax.multiply(bx), ac.multiply(bc), kOpTol, "multiply");
    expect_parity(ax.divide(bx), ac.divide(bc), kOpTol, "divide");
    expect_parity(ax.add_scalar(0.37f), ac.add_scalar(0.37f), kOpTol, "add_scalar");
    expect_parity(ax.multiply_scalar(-1.9f), ac.multiply_scalar(-1.9f), kOpTol, "multiply_scalar");
    expect_parity(ax.divide_scalar(0.73f), ac.divide_scalar(0.73f), kOpTol, "divide_scalar");
    expect_parity(ax.abs(), ac.abs(), kOpTol, "abs");
    expect_parity(ax.exp(), ac.exp(), kOpTol, "exp");
    expect_parity(ax.square(), ac.square(), kOpTol, "square");
    expect_parity(ax.abs().sqrt(), ac.abs().sqrt(), kOpTol, "sqrt");
    expect_parity(ax.relu(), ac.relu(), kOpTol, "relu");
    expect_parity(ax.leaky_relu(0.01f), ac.leaky_relu(0.01f), kOpTol, "leaky_relu");
    expect_parity(ax.clamp(-0.5f, 0.5f), ac.clamp(-0.5f, 0.5f), kOpTol, "clamp");
}

TEST_F(BackendParityTest, MatmulFamily)
{
    TX ax(7, 5), bx(5, 3);
    TC ac(7, 5), bc(5, 3);
    fill_pair(ax, ac, 31);
    fill_pair(bx, bc, 32);

    expect_parity(ax.matmul(bx), ac.matmul(bc), kOpTol, "matmul");
    expect_parity(ax.transpose(), ac.transpose(), kOpTol, "transpose");

    // matmul_transposed: A(7x5) · B(3x5)^T → (7x3)
    TX btx(3, 5);
    TC btc(3, 5);
    fill_pair(btx, btc, 33);
    expect_parity(
        ax.matmul_transposed(btx), ac.matmul_transposed(btc), kOpTol, "matmul_transposed");
}

TEST_F(BackendParityTest, Reductions)
{
    TX ax(6, 9), bx(6, 9);
    TC ac(6, 9), bc(6, 9);
    fill_pair(ax, ac, 41);
    fill_pair(bx, bc, 42);

    EXPECT_NEAR(ax.sum(), ac.sum(), kOpTol * ax.size()) << "sum";
    EXPECT_NEAR(ax.mean(), ac.mean(), kOpTol) << "mean";
    EXPECT_NEAR(ax.norm(), ac.norm(), kOpTol * 10) << "norm";
    EXPECT_NEAR(ax.mean_squared_error(bx), ac.mean_squared_error(bc), kOpTol)
        << "mean_squared_error";
    expect_parity(ax.sum_rows(), ac.sum_rows(), kOpTol * ax.rows(), "sum_rows");
    expect_parity(ax.sum_cols(), ac.sum_cols(), kOpTol * ax.cols(), "sum_cols");
    expect_parity(ax.rowwise_sum(), ac.rowwise_sum(), kOpTol * ax.cols(), "rowwise_sum");
}

TEST_F(BackendParityTest, SliceAndReshape)
{
    TX ax(6, 8);
    TC ac(6, 8);
    fill_pair(ax, ac, 51);

    expect_parity(ax.block(1, 2, 3, 4), ac.block(1, 2, 3, 4), kOpTol, "block");
    expect_parity(ax.row(3), ac.row(3), kOpTol, "row");
    expect_parity(ax.col(5), ac.col(5), kOpTol, "col");
    expect_parity(
        std::as_const(ax).reshape({4, 12}), std::as_const(ac).reshape({4, 12}), kOpTol, "reshape");
}

// ─── layers ──────────────────────────────────────────────────────────────────

/// Copy layer weights so both backends start from identical parameters.
template <typename LX, typename LC>
void sync_linear(LX& lx, LC& lc, unsigned seed)
{
    fill_pair(lx.weight, lc.weight, seed, -0.5f, 0.5f);
    fill_pair(lx.bias, lc.bias, seed + 1, -0.1f, 0.1f);
}

TEST_F(BackendParityTest, LinearForwardBackward)
{
    constexpr int kIn = 12, kOut = 5, kBatch = 4;
    LinearImpl<XT> lx(kIn, kOut);
    LinearImpl<CL> lc(kIn, kOut);
    sync_linear(lx, lc, 61);

    TX inx(kBatch, kIn);
    TC inc(kBatch, kIn);
    fill_pair(inx, inc, 62);

    auto outx = lx.forward(inx, true);
    auto outc = lc.forward(inc, true);
    expect_parity(outx, outc, kOpTol, "Linear forward");

    TX gx(kBatch, kOut);
    TC gc(kBatch, kOut);
    fill_pair(gx, gc, 63);

    expect_parity(lx.backward(gx), lc.backward(gc), kOpTol, "Linear input grad");
    expect_parity(lx.weight.grad(), lc.weight.grad(), kOpTol, "Linear weight grad");
    expect_parity(lx.bias.grad(), lc.bias.grad(), kOpTol, "Linear bias grad");
}

TEST_F(BackendParityTest, LifBatchedStateEvolution)
{
    // Batched (B=4) multi-step run: exercises the OpenCL lif_step_inplace fast
    // path against the XTensor generic path, including membrane-state carry.
    constexpr int kBatch = 4, kFeat = 8, kSteps = 3;
    LifImpl<XT> lx(1.0f, 5.0f, 5.0f, 0.25f);
    LifImpl<CL> lc(1.0f, 5.0f, 5.0f, 0.25f);

    for (int t = 0; t < kSteps; ++t)
    {
        TX inx(kBatch, kFeat);
        TC inc(kBatch, kFeat);
        fill_pair(inx, inc, 70 + static_cast<unsigned>(t), 0.0f, 0.4f);

        auto sx = lx.forward(inx, true);
        auto sc = lc.forward(inc, true);
        expect_parity(sx, sc, kOpTol, "Lif spikes");
        expect_parity(lx.v_mem, lc.v_mem, kOpTol, "Lif v_mem");
    }

    TX gx(kBatch, kFeat);
    TC gc(kBatch, kFeat);
    fill_pair(gx, gc, 79);

    expect_parity(lx.backward(gx), lc.backward(gc), kChainTol, "Lif input grad");
    expect_parity(lx.resistance.grad(), lc.resistance.grad(), kChainTol, "Lif R grad");
    expect_parity(lx.capacitance.grad(), lc.capacitance.grad(), kChainTol, "Lif C grad");
    expect_parity(
        lx.voltage_threshold.grad(), lc.voltage_threshold.grad(), kChainTol, "Lif V_th grad");
}

TEST_F(BackendParityTest, LifIntegratorForwardBackward)
{
    constexpr int kBatch = 4, kFeat = 6, kSteps = 3;
    LifIntegratorImpl<XT> lx(1.0f, 5.0f, 5.0f);
    LifIntegratorImpl<CL> lc(1.0f, 5.0f, 5.0f);

    for (int t = 0; t < kSteps; ++t)
    {
        TX inx(kBatch, kFeat);
        TC inc(kBatch, kFeat);
        fill_pair(inx, inc, 90 + static_cast<unsigned>(t));
        expect_parity(lx.forward(inx, true), lc.forward(inc, true), kOpTol, "LifIntegrator v_mem");
    }

    TX gx(kBatch, kFeat);
    TC gc(kBatch, kFeat);
    fill_pair(gx, gc, 99);
    expect_parity(lx.backward(gx), lc.backward(gc), kChainTol, "LifIntegrator input grad");
}

TEST_F(BackendParityTest, SnnAutoencoderChainEndToEnd)
{
    // The SNN-AE shape used by Experiment05 phase00 (Linear→Lif encoder,
    // Linear→LifIntegrator decoder), batched — end-to-end forward + backward.
    constexpr int kBatch = 8, kIn = 16, kLatent = 4;

    LinearImpl<XT> encx(kIn, kLatent);
    LinearImpl<CL> encc(kIn, kLatent);
    LifImpl<XT> lifx(1.0f, 5.0f, 5.0f, 0.25f);
    LifImpl<CL> lifc(1.0f, 5.0f, 5.0f, 0.25f);
    LinearImpl<XT> decx(kLatent, kIn);
    LinearImpl<CL> decc(kLatent, kIn);
    LifIntegratorImpl<XT> intx(1.0f, 5.0f, 5.0f);
    LifIntegratorImpl<CL> intc(1.0f, 5.0f, 5.0f);
    sync_linear(encx, encc, 101);
    sync_linear(decx, decc, 103);

    TX inx(kBatch, kIn);
    TC inc(kBatch, kIn);
    fill_pair(inx, inc, 105, 0.0f, 0.5f);

    // forward
    auto rx = intx.forward(decx.forward(lifx.forward(encx.forward(inx, true), true), true), true);
    auto rc = intc.forward(decc.forward(lifc.forward(encc.forward(inc, true), true), true), true);
    expect_parity(rx, rc, kChainTol, "AE reconstruction");

    // MSE-style gradient: 2(recon − input)/N, identical on both by construction
    auto gx = (rx - inx).multiply_scalar(2.0f / static_cast<float>(rx.size()));
    auto gc = (rc - inc).multiply_scalar(2.0f / static_cast<float>(rc.size()));

    // backward through the whole chain
    auto ginx = encx.backward(lifx.backward(decx.backward(intx.backward(gx))));
    auto ginc = encc.backward(lifc.backward(decc.backward(intc.backward(gc))));
    expect_parity(ginx, ginc, kChainTol, "AE input grad");
    expect_parity(encx.weight.grad(), encc.weight.grad(), kChainTol, "AE encoder weight grad");
    expect_parity(decx.weight.grad(), decc.weight.grad(), kChainTol, "AE decoder weight grad");
}

} // namespace
