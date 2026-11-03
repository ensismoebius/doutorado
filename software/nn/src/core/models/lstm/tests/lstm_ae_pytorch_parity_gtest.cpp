/**
 * @file src/core/models/lstm/tests/lstm_ae_pytorch_parity_gtest.cpp
 * @brief The WHOLE LSTM autoencoder, against an independent PyTorch implementation.
 *
 * `micro_network_parity_gtest` pins one LSTM layer plus a Linear head, forward only.
 * `nn::models::lstm::LSTMAutoencoder` -- what the meeting01 experiment trains -- is more than
 * that: stacked encoder, tanh bottleneck, a decoder fed the SAME expanded latent at every step,
 * stacked decoder, a per-step output projection, and a MASKED mean-squared-error. Each layer
 * can be right while that composition is wrong (gradient chained in the wrong order through the
 * decoder stack, the replicated latent's gradient not summed over the steps, the mask
 * normalisation, the gate permutation only visible in a stack). This test loads, into the C++
 * model, exactly the weights PyTorch ran on and compares
 *
 *   - the reconstruction,
 *   - the masked loss,
 *   - the gradient of EVERY parameter after one backward pass.
 *
 * Fixtures: `fixtures/lstm_ae_refs.npz`, from `scripts/testing/gen_lstm_ae_refs.py` (developer
 * step, needs torch; the twin model is `scripts/testing/lstm_ae_torch.py`). Committed, so CI
 * needs no Python.
 *
 * Adam is pinned separately by `optimizer_parity_gtest`; this file does not re-test it.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "Backend.hpp"
#include "cnpy.h"
#include "layers/losses/MSELoss.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "tensor/Tensor.hpp"

namespace
{
using Tensor = nn::Tensor;

const cnpy::npz_t& refs()
{
    static const cnpy::npz_t data = []
    {
        const std::filesystem::path p =
            std::filesystem::path(__FILE__).parent_path() / "fixtures" / "lstm_ae_refs.npz";
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

// numpy C order; only the structured accessor at(i, j) is used, never the linear at(k).
auto to_tensor(const std::string& key) -> Tensor
{
    const auto& a = arr(key);
    const auto rows = static_cast<nn::Index>(a.shape[0]);
    const auto cols = static_cast<nn::Index>(a.shape.size() > 1 ? a.shape[1] : 1);
    const float* d = a.data<float>();
    Tensor t(rows, cols);
    for (nn::Index i = 0; i < rows; ++i)
        for (nn::Index j = 0; j < cols; ++j) t.at(i, j) = d[(i * cols) + j];
    return t;
}

// Largest |got - want| over a tensor, with a message naming the worst element.
void expect_close(const Tensor& got, const Tensor& want, float tol, const std::string& what)
{
    ASSERT_EQ(got.rows(), want.rows()) << what;
    ASSERT_EQ(got.cols(), want.cols()) << what;
    float worst = 0.0F;
    nn::Index wi = 0;
    nn::Index wj = 0;
    for (nn::Index i = 0; i < got.rows(); ++i)
    {
        for (nn::Index j = 0; j < got.cols(); ++j)
        {
            const float e = std::fabs(got.at(i, j) - want.at(i, j));
            if (e > worst)
            {
                worst = e;
                wi = i;
                wj = j;
            }
        }
    }
    EXPECT_LE(worst, tol) << what << ": worst |got - want| = " << worst << " at (" << wi << ","
                          << wj << "): got " << got.at(wi, wj) << ", want " << want.at(wi, wj);
}

// Parameters in LSTMAutoencoder::params() order: per encoder layer {W, U, b}, then
// enc_proj {weight, bias}, dec_expand, per decoder layer {W, U, b}, out_proj.
auto param_names(int layers) -> std::vector<std::string>
{
    std::vector<std::string> n;
    for (int l = 0; l < layers; ++l)
        for (const char* k : {"W", "U", "b"}) n.push_back("enc_lstm" + std::to_string(l) + "." + k);
    for (const char* t : {"enc_proj", "dec_expand"})
        for (const char* k : {"weight", "bias"}) n.push_back(std::string(t) + "." + k);
    for (int l = 0; l < layers; ++l)
        for (const char* k : {"W", "U", "b"}) n.push_back("dec_lstm" + std::to_string(l) + "." + k);
    for (const char* k : {"weight", "bias"}) n.push_back(std::string("out_proj.") + k);
    return n;
}

struct Case
{
    int d, h, z, layers, s;
    std::string prefix;
};

auto case_of(int k) -> Case
{
    const std::string p = "c" + std::to_string(k) + "_";
    const float* dims = arr(p + "dims").data<float>();
    return {static_cast<int>(dims[0]),
        static_cast<int>(dims[1]),
        static_cast<int>(dims[2]),
        static_cast<int>(dims[3]),
        static_cast<int>(dims[4]),
        p};
}

// Build the C++ model, load the exact weights PyTorch used, run forward + masked-MSE + backward.
struct AeRun
{
    nn::models::lstm::LSTMAutoencoder model;
    Tensor recon;
    float loss = 0.0F;

    explicit AeRun(const Case& c)
        : model(nn::models::lstm::LSTMAutoencoderConfig{c.d, c.s, c.h, c.z, c.layers})
    {
        std::map<std::string, Tensor> sd;
        for (const auto& name : param_names(c.layers)) sd[name] = to_tensor(c.prefix + "p." + name);
        model.load_state_dict(sd);

        model.reset_state();
        recon = model.forward(to_tensor(c.prefix + "x"), true);

        MSELossImpl<nn::Backend> mse;
        mse.set_target(to_tensor(c.prefix + "target"));
        mse.set_mask(to_tensor(c.prefix + "mask"));
        loss = mse.forward(recon, true).at(0, 0);
        model.backward(mse.backward(recon));
    }
};

constexpr int kCases = 2;

// Tolerances: both sides are float32; the residual is op-ordering noise through 2 stacked
// layers and up to 9 steps. Forward values are O(1); gradients reach O(1) too.
constexpr float kForwardTol = 2e-5F;
constexpr float kGradTol = 5e-5F;

TEST(LstmAutoencoderTorchParity, FixtureHasTheCasesTheTestExpects)
{
    ASSERT_EQ(static_cast<int>(arr("n_cases").data<float>()[0]), kCases);
}

TEST(LstmAutoencoderTorchParity, ReconstructionAndMaskedLossMatchTorch)
{
    for (int k = 0; k < kCases; ++k)
    {
        const Case c = case_of(k);
        const AeRun run(c);
        expect_close(run.recon,
            to_tensor(c.prefix + "recon"),
            kForwardTol,
            "case " + std::to_string(k) + " reconstruction");
        EXPECT_NEAR(run.loss, arr(c.prefix + "loss").data<float>()[0], kForwardTol)
            << "case " << k << " masked loss";
    }
}

TEST(LstmAutoencoderTorchParity, EveryParameterGradientMatchesTorch)
{
    for (int k = 0; k < kCases; ++k)
    {
        const Case c = case_of(k);
        AeRun run(c);
        const auto names = param_names(c.layers);
        auto params = run.model.params();
        ASSERT_EQ(params.size(), names.size()) << "case " << k << ": parameter count";
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            expect_close(params[i]->grad(),
                to_tensor(c.prefix + "g." + names[i]),
                kGradTol,
                "case " + std::to_string(k) + " d" + names[i]);
        }
    }
}

// The fixture is only a test if the gradients are not all ~0 (a dead network would "match"
// trivially) -- keep the generator honest.
TEST(LstmAutoencoderTorchParity, FixtureGradientsAreNotDegenerate)
{
    for (int k = 0; k < kCases; ++k)
    {
        const Case c = case_of(k);
        for (const auto& name : param_names(c.layers))
        {
            const Tensor g = to_tensor(c.prefix + "g." + name);
            float biggest = 0.0F;
            for (nn::Index i = 0; i < g.rows(); ++i)
                for (nn::Index j = 0; j < g.cols(); ++j)
                    biggest = std::max(biggest, std::fabs(g.at(i, j)));
            EXPECT_GT(biggest, 1e-4F) << "case " << k << " d" << name << " is ~0 in the fixture";
        }
    }
}

} // namespace
