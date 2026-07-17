/**
 * @file optimizer_parity_gtest.cpp
 * @brief Ground-truth parity for every optimizer against its reference implementation.
 *
 * `scripts/testing/gen_optimizer_refs.py` (developer step; needs torch + the authors'
 * lion-pytorch / schedulefree packages) drives each *reference* optimizer
 * over a fixed sequence of seeded parameters and gradients and records the parameter after
 * every step into fixtures/optimizer_refs.npz — committed, so this test needs no Python.
 * Here we replay the identical parameters/gradients through our port and compare.
 *
 * Why this test earns its keep: these algorithms are easy to implement *plausibly* and
 * wrong. Lion decays before the update and advances its momentum after it; Schedule-Free
 * keeps three coupled sequences (x/y/z) and evaluates at a different point than it trains
 * at. A from-memory port gets those subtly wrong and still trains "fine" — silently worse.
 * Diffing against the authors' own code is what makes the ports trustworthy.
 *
 * Tolerances: 2e-5 absolute. Ours is float32 and so is the reference; the residual is
 * accumulated op-ordering noise over 5 steps.
 */
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "cnpy.h"
#include "optimizers/Adam.hpp"
#include "optimizers/Lion.hpp"
#include "optimizers/SGD.hpp"
#include "optimizers/ScheduleFreeAdamW.hpp"
#include "tensor/Tensor.hpp"

namespace
{

const cnpy::npz_t& refs()
{
    static const cnpy::npz_t data = []
    {
        const std::filesystem::path p =
            std::filesystem::path(__FILE__).parent_path() / "fixtures" / "optimizer_refs.npz";
        return cnpy::npz_load(p.string());
    }();
    return data;
}

const cnpy::NpyArray& arr(const std::string& key)
{
    const auto it = refs().find(key);
    EXPECT_NE(it, refs().end()) << "missing fixture key: " << key;
    return it->second;
}

// Fixture arrays are numpy C-order. Use only the structured accessor at(i,j) — never the
// linear at(k), which exposes backend storage order and would transpose on OpenCL.
nn::Tensor to_tensor(const std::string& key)
{
    const auto& a = arr(key);
    const auto rows = static_cast<nn::Index>(a.shape[0]);
    const auto cols = static_cast<nn::Index>(a.shape.size() > 1 ? a.shape[1] : 1);
    const float* d = a.data<float>();
    nn::Tensor t(rows, cols);
    for (nn::Index i = 0; i < rows; ++i)
        for (nn::Index j = 0; j < cols; ++j) t.at(i, j) = d[(i * cols) + j];
    return t;
}

int steps_of(const std::string& prefix)
{
    return static_cast<int>(arr(prefix + "_steps").data<float>()[0]);
}

void expect_tensor_near(
    const nn::Tensor& got, const nn::Tensor& want, float tol, const std::string& what)
{
    ASSERT_EQ(got.rows(), want.rows()) << what;
    ASSERT_EQ(got.cols(), want.cols()) << what;
    for (nn::Index i = 0; i < got.rows(); ++i)
        for (nn::Index j = 0; j < got.cols(); ++j)
            EXPECT_NEAR(got.at(i, j), want.at(i, j), tol)
                << what << " at (" << i << "," << j << ")";
}

/// Replay a fixture's parameter/gradient sequence through `opt` and compare every step.
template <typename OptT>
void replay(const std::string& prefix, OptT& opt, float tol)
{
    nn::Tensor p = to_tensor(prefix + "_p0");
    std::vector<nn::Tensor*> params = {&p};
    opt.attach(params);

    const int n = steps_of(prefix);
    for (int k = 0; k < n; ++k)
    {
        p.set_grad(to_tensor(prefix + "_g" + std::to_string(k)));
        opt.step(params);
        expect_tensor_near(p,
            to_tensor(prefix + "_p" + std::to_string(k + 1)),
            tol,
            prefix + " step " + std::to_string(k + 1));
    }
}

constexpr float kTol = 2e-5F;

// ── Adam / SGD vs PyTorch's own implementations ──────────────────────────────

TEST(OptimizerParity, AdamMatchesTorchAdam)
{
    Adam opt(0.01F, 0.9F, 0.999F, 1e-8F);
    replay("adam_2x3", opt, kTol);
}

// torch.optim.AdamW is the decoupled-weight-decay form our Adam::weight_decay implements.
// The 2-D parameter matters: this project applies decay only to 2-D weight matrices (so SNN
// 1x1 scalars keep tau=R*C intact), and on a 2-D matrix the two agree exactly.
TEST(OptimizerParity, AdamWithWeightDecayMatchesTorchAdamW)
{
    Adam opt(0.01F, 0.9F, 0.999F, 1e-8F);
    opt.weight_decay = 0.1F;
    replay("adamw_2x3", opt, kTol);
}

TEST(OptimizerParity, SgdMatchesTorchSgd)
{
    SGD opt(0.05F, 0.0F);
    replay("sgd_2x3", opt, kTol);
}

TEST(OptimizerParity, SgdMomentumMatchesTorchSgd)
{
    SGD opt(0.05F, 0.9F);
    replay("sgdm_2x3", opt, kTol);
}

// ── Lion vs lion-pytorch ─────────────────────────────────────────────────────

TEST(OptimizerParity, LionMatchesLionPytorch)
{
    Lion opt(1e-3F, 0.9F, 0.99F);
    replay("lion_2x3", opt, kTol);
}

TEST(OptimizerParity, LionWithWeightDecayMatchesLionPytorch)
{
    Lion opt(1e-3F, 0.9F, 0.99F);
    opt.weight_decay = 0.1F;
    replay("lion_wd_2x3", opt, kTol);
}

// ── Schedule-Free AdamW vs schedulefree ──────────────────────────────────────

TEST(OptimizerParity, ScheduleFreeAdamWMatchesReference)
{
    ScheduleFreeAdamW opt(0.0025F, 0.9F, 0.999F, 1e-8F);
    replay("sfadamw_2x3", opt, kTol);
}

TEST(OptimizerParity, ScheduleFreeAdamWWithWeightDecayMatchesReference)
{
    ScheduleFreeAdamW opt(0.0025F, 0.9F, 0.999F, 1e-8F);
    opt.weight_decay = 0.1F;
    replay("sfadamw_wd_2x3", opt, kTol);
}

TEST(OptimizerParity, ScheduleFreeAdamWWarmupMatchesReference)
{
    ScheduleFreeAdamW opt(0.0025F, 0.9F, 0.999F, 1e-8F, /*warmup_steps=*/3);
    replay("sfadamw_warm_2x3", opt, kTol);
}

// The train/eval swap is the whole point of schedule-free: it trains at the extrapolated
// iterate y but converges at the averaged iterate x, so validation must read x. This pins
// both iterates against the reference's own train()/eval().
TEST(OptimizerParity, ScheduleFreeAdamWEvalIterateMatchesReference)
{
    nn::Tensor p = to_tensor("sfeval_p0");
    std::vector<nn::Tensor*> params = {&p};
    ScheduleFreeAdamW opt(0.0025F, 0.9F, 0.999F, 1e-8F);
    opt.attach(params);

    const int n = static_cast<int>(arr("sfeval_steps").data<float>()[0]);
    for (int k = 0; k < n; ++k)
    {
        p.set_grad(to_tensor("sfeval_g" + std::to_string(k)));
        opt.step(params);
    }

    // Training iterate (y) — what the parameters hold while training.
    expect_tensor_near(p, to_tensor("sfeval_y"), kTol, "schedule-free train iterate y");

    // Evaluation iterate (x) — the averaged point the method actually converges to.
    opt.train_mode(false);
    expect_tensor_near(p, to_tensor("sfeval_x"), kTol, "schedule-free eval iterate x");

    // ...and switching back restores y, so training can continue (OptimizerEvalScope).
    opt.train_mode(true);
    expect_tensor_near(p, to_tensor("sfeval_y"), kTol, "schedule-free restored y");
}

} // namespace
