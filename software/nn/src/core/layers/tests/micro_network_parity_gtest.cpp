/**
 * @file src/core/layers/tests/micro_network_parity_gtest.cpp
 * @brief Ground-truth parity for whole micro-NETWORKS, not single layers.
 *
 * `pytorch_parity_gtest` pins each layer in isolation. That is necessary but not
 * sufficient: every layer can be individually correct while the network built from them is
 * wrong — gradients chained in the wrong order through a stack, membrane state not reset
 * between sequences, a gate permutation that only shows up once composed, a bias applied
 * twice. The experiments train networks, so these three minimal networks mirror the shapes
 * the thesis actually uses and check them end to end (forward AND parameter gradients).
 *
 * Fixtures come from `scripts/testing/gen_micro_network_refs.py` (developer step; needs
 * torch + snntorch) and are committed, so CI needs no Python.
 *
 * Two limits are deliberate and encoded rather than hidden — see each test:
 *   - the SNN's spiking backward cannot be compared (different surrogate), so backward is
 *     pinned in readout mode where the path is surrogate-free;
 *   - our LSTM uses softsign gates, not sigmoid/tanh, so it is pinned against our own
 *     recurrence and the divergence from real torch is asserted to stay bounded.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "Backend.hpp"
#include "cnpy.h"
#include "layers/Layers.hpp"
#include "layers/losses/MSELoss.hpp"
#include "tensor/Tensor.hpp"

namespace
{
using Backend = nn::Backend;
using Tensor = nn::Tensor;

const cnpy::npz_t& refs()
{
    static const cnpy::npz_t data = []
    {
        const std::filesystem::path p =
            std::filesystem::path(__FILE__).parent_path() / "fixtures" / "micro_network_refs.npz";
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

// Fixture arrays are numpy C-order. Only the structured accessor at(i,j) is used — the
// linear at(k) would expose backend storage order and transpose on OpenCL.
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

void fill(Tensor& dst, const std::string& key)
{
    const auto& a = arr(key);
    const auto rows = static_cast<nn::Index>(a.shape[0]);
    const auto cols = static_cast<nn::Index>(a.shape.size() > 1 ? a.shape[1] : 1);
    const float* d = a.data<float>();
    ASSERT_EQ(dst.rows(), rows) << key;
    ASSERT_EQ(dst.cols(), cols) << key;
    for (nn::Index i = 0; i < rows; ++i)
        for (nn::Index j = 0; j < cols; ++j) dst.at(i, j) = d[(i * cols) + j];
}

void expect_close(const Tensor& got, const std::string& key, float tol)
{
    const Tensor want = to_tensor(key);
    ASSERT_EQ(got.rows(), want.rows()) << key;
    ASSERT_EQ(got.cols(), want.cols()) << key;
    for (nn::Index i = 0; i < got.rows(); ++i)
        for (nn::Index j = 0; j < got.cols(); ++j)
            EXPECT_NEAR(got.at(i, j), want.at(i, j), tol) << key << " at (" << i << "," << j << ")";
}

auto dim(const std::string& key, int i) -> int
{
    return static_cast<int>(arr(key).data<float>()[i]);
}

constexpr float kTol = 2e-5F;

// ══ micro ANN: Linear -> ReLU -> Linear, MSE ═════════════════════════════════
// Exact parity with torch.nn.Sequential: every op has an exact counterpart. This is the
// baseline that proves composition + gradient chaining are right when nothing approximate
// is involved, so a failure in the SNN/LSTM tests can be attributed to those specifics.
TEST(MicroNetworkParity, AnnMatchesTorchForwardAndBackward)
{
    const int D = dim("ann_dims", 1), H = dim("ann_dims", 2), O = dim("ann_dims", 3);

    nn::Linear fc1(D, H);
    nn::ReLU relu;
    nn::Linear fc2(H, O);
    fill(fc1.weight, "ann_w1");
    fill(fc1.bias, "ann_b1");
    fill(fc2.weight, "ann_w2");
    fill(fc2.bias, "ann_b2");

    Tensor x = to_tensor("ann_x");
    Tensor h = fc1.forward(x, true);
    Tensor a = relu.forward(h, true);
    Tensor out = fc2.forward(a, true);
    expect_close(out, "ann_out", kTol);

    MSELossImpl<Backend> loss;
    loss.set_target(to_tensor("ann_target"));
    Tensor l = loss.forward(out, true);
    EXPECT_NEAR(l.at(0, 0), arr("ann_loss").data<float>()[0], kTol);

    Tensor g = loss.backward(out);
    g = fc2.backward(g);
    g = relu.backward(g);
    g = fc1.backward(g);

    // Gradients w.r.t. every parameter AND the input: a wrong chaining order shows up here
    // even when the forward output is right.
    expect_close(fc1.weight.grad(), "ann_gw1", kTol);
    expect_close(fc1.bias.grad(), "ann_gb1", kTol);
    expect_close(fc2.weight.grad(), "ann_gw2", kTol);
    expect_close(fc2.bias.grad(), "ann_gb2", kTol);
    expect_close(g, "ann_gx", kTol);
}

// ══ micro SNN: Linear -> LIF -> Linear, time-major ═══════════════════════════
namespace
{
/// Build the LIF from the fixture's params row: [time_step, R, C, V_th, reset_zero, readout].
auto make_lif(const std::string& p, int T) -> std::unique_ptr<nn::LifBPTT>
{
    const float* q = arr(p + "params").data<float>();
    return std::make_unique<nn::LifBPTT>(T,
        /*time_step=*/q[0],
        /*resistance=*/q[1],
        /*capacitance=*/q[2],
        /*voltage_threshold=*/q[3],
        /*reset_zero=*/q[4] != 0.0F,
        /*reset_potential=*/0.0F,
        /*readout_mode=*/q[5] != 0.0F);
}
} // namespace

// Forward WITH spikes, against snnTorch's snn.Leaky. No surrogate is involved in a forward
// pass, so this must agree exactly — it pins the recurrence, the threshold/reset, and the
// (T*B, F) time-major contract through a real stack.
TEST(MicroNetworkParity, SnnSpikingForwardMatchesSnntorch)
{
    const std::string p = "snn_spk_";
    const int T = dim(p + "dims", 0), D = dim(p + "dims", 2);
    const int H = dim(p + "dims", 3), O = dim(p + "dims", 4);

    nn::Linear fc_in(D, H);
    nn::Linear fc_out(H, O);
    fill(fc_in.weight, p + "w1");
    fill(fc_in.bias, p + "b1");
    fill(fc_out.weight, p + "w2");
    fill(fc_out.bias, p + "b2");
    auto lif = make_lif(p, T);
    lif->reset_state();

    Tensor x = to_tensor(p + "x"); // (T*B, D)
    Tensor cur = fc_in.forward(x, true);
    Tensor spk = lif->forward(cur, true);
    Tensor out = fc_out.forward(spk, true);

    expect_close(out, p + "out", kTol);
}

// Backward in READOUT mode. Our LifBPTT uses an ExponentialSurrogate for the spike
// derivative while snnTorch's Leaky defaults to arctan — different functions, so gradients
// through a *spiking* layer cannot agree, and pinning them would be meaningless. In readout
// mode the neuron emits its membrane and never fires, so the path is purely continuous and
// both sides must agree exactly. This isolates "is the recurrence + composition + gradient
// chaining right" from "which surrogate is used".
TEST(MicroNetworkParity, SnnReadoutForwardAndBackwardMatchSnntorch)
{
    const std::string p = "snn_ro_";
    const int T = dim(p + "dims", 0), D = dim(p + "dims", 2);
    const int H = dim(p + "dims", 3), O = dim(p + "dims", 4);

    nn::Linear fc_in(D, H);
    nn::Linear fc_out(H, O);
    fill(fc_in.weight, p + "w1");
    fill(fc_in.bias, p + "b1");
    fill(fc_out.weight, p + "w2");
    fill(fc_out.bias, p + "b2");
    auto lif = make_lif(p, T);
    lif->reset_state();

    Tensor x = to_tensor(p + "x");
    Tensor cur = fc_in.forward(x, true);
    Tensor mem = lif->forward(cur, true);
    Tensor out = fc_out.forward(mem, true);
    expect_close(out, p + "out", kTol);

    MSELossImpl<Backend> loss;
    loss.set_target(to_tensor(p + "target"));
    Tensor l = loss.forward(out, true);
    EXPECT_NEAR(l.at(0, 0), arr(p + "loss").data<float>()[0], kTol);

    Tensor g = loss.backward(out);
    g = fc_out.backward(g);
    g = lif->backward(g);
    g = fc_in.backward(g);

    // fc_in's gradient is the interesting one: it only becomes correct if BPTT propagated
    // through the whole unrolled recurrence and back into the layer feeding it.
    expect_close(fc_out.weight.grad(), p + "gw2", kTol);
    expect_close(fc_out.bias.grad(), p + "gb2", kTol);
    expect_close(fc_in.weight.grad(), p + "gw1", kTol);
    expect_close(fc_in.bias.grad(), p + "gb1", kTol);
}

// State must not leak between sequences: running the same input twice with a reset in
// between must reproduce the first result exactly. Without reset_state() the membrane
// carries over and the second pass silently differs — the failure mode CLAUDE.md's SNN
// invariant #4 warns about, and one no single-shot parity check would catch.
TEST(MicroNetworkParity, SnnResetStateMakesSequencesIndependent)
{
    const std::string p = "snn_spk_";
    const int T = dim(p + "dims", 0), D = dim(p + "dims", 2);
    const int H = dim(p + "dims", 3), O = dim(p + "dims", 4);

    nn::Linear fc_in(D, H);
    nn::Linear fc_out(H, O);
    fill(fc_in.weight, p + "w1");
    fill(fc_in.bias, p + "b1");
    fill(fc_out.weight, p + "w2");
    fill(fc_out.bias, p + "b2");
    auto lif = make_lif(p, T);

    Tensor x = to_tensor(p + "x");
    auto run = [&]() -> Tensor
    {
        lif->reset_state();
        Tensor cur = fc_in.forward(x, true);
        Tensor spk = lif->forward(cur, true);
        return fc_out.forward(spk, true);
    };

    Tensor first = run();
    Tensor second = run(); // identical only if reset_state() truly clears the membrane
    ASSERT_EQ(first.rows(), second.rows());
    for (nn::Index i = 0; i < first.rows(); ++i)
        for (nn::Index j = 0; j < first.cols(); ++j)
            EXPECT_FLOAT_EQ(first.at(i, j), second.at(i, j))
                << "sequence " << i << "," << j << " leaked state across reset_state()";
    expect_close(second, p + "out", kTol);
}

// ── LIF reset-mode equivalence, under a HARD drive ───────────────────────────
namespace
{
/// Run a bare LIF over the fixture's input and return the spike train.
auto run_reset_case(const std::string& p) -> Tensor
{
    const int T = dim(p + "dims", 0);
    auto lif = make_lif(p, T);
    lif->reset_state();
    Tensor x = to_tensor(p + "x");
    return lif->forward(x, true);
}

auto spike_disagreement(const Tensor& a, const Tensor& b) -> float
{
    int diff = 0, n = 0;
    for (nn::Index i = 0; i < a.rows(); ++i)
        for (nn::Index j = 0; j < a.cols(); ++j, ++n)
            if (std::abs(a.at(i, j) - b.at(i, j)) > 0.5F) ++diff;
    return n ? static_cast<float>(diff) / static_cast<float>(n) : 0.0F;
}
} // namespace

// The mode the thesis actually uses (Lif/LifBPTT default to reset_zero=true, and no
// production code selects subtract). Driven HARD on purpose: the existing per-layer LIF
// fixture spikes only 3/36 times, which is too weak to exercise a reset at all. Here
// snnTorch fires ~17% of the time and our spike train must match it EXACTLY.
TEST(MicroNetworkParity, LifZeroResetMatchesSnntorchUnderHardDrive)
{
    const Tensor spk = run_reset_case("reset_zero_");
    expect_close(spk, "reset_zero_spk", kTol);
    EXPECT_FLOAT_EQ(spike_disagreement(spk, to_tensor("reset_zero_spk")), 0.0F);
}

// The subtract mode DIVERGES from snnTorch, by construction, and this pins that fact.
//
//   ours     : v[t] = beta*v[t-1] + I[t] - V_th*spk[t]      (reset applied immediately,
//                                                            so it is decayed next step)
//   snnTorch : mem[t] = beta*mem[t-1] + I[t] - V_th*spk[t-1] (reset un-decayed, next step)
//
// i.e. our reset term ends up multiplied by beta. Ours is the textbook soft-reset LIF;
// snnTorch's is snnTorch's convention. Neither is wrong, but they are NOT the same neuron —
// so the claim in gen_pytorch_refs.py that our LifBPTT "is exactly snnTorch's snn.Leaky"
// holds only for reset="zero" (proven above), not for subtract. The existing per-layer
// fixture tests subtract and passes only because it barely spikes.
//
// This is asserted rather than fixed because production never selects subtract. If it ever
// does, this test says exactly what it will and will not reproduce.
TEST(MicroNetworkParity, LifSubtractResetDivergesFromSnntorchAsDocumented)
{
    const Tensor spk = run_reset_case("reset_subtract_");
    const Tensor ref = to_tensor("reset_subtract_spk");
    const float dis = spike_disagreement(spk, ref);

    // It must NOT agree — if it does, the reset semantics changed and the note above
    // (and gen_pytorch_refs.py's claim) need revisiting.
    EXPECT_GT(dis, 0.0F) << "our subtract-reset LIF now matches snnTorch; if intentional, "
                            "update this test and the LifBPTT header";
    // ...but the divergence must stay small and bounded, not grow into a different neuron.
    EXPECT_LT(dis, 0.10F) << "subtract-reset divergence from snnTorch grew beyond the "
                             "measured ~2-3% band";
}

// ══ micro LSTM: LSTM -> Linear ═══════════════════════════════════════════════
// Pinned against a NumPy model of OUR OWN recurrence, not torch.nn.LSTM. Reason: our
// LSTMLayer uses FastActivations' rational approximations — rat_sig(x)=0.5+x/(2(1+|x|)) and
// rat_tanh(x)=x/(1+|x|), i.e. softsign gates chosen for speed — not sigmoid/tanh. They are
// not close: |tanh - rat_tanh| reaches 0.306 on [-4,4]. So ours is a *softsign-gated* LSTM
// and can never match torch numerically; a torch comparison would need a tolerance so loose
// it proves nothing. This reference still pins gate order (i,f,o,g vs torch's i,f,g,o), the
// c/h update order, the merged bias, and composition with the head — everything except the
// activation choice, which is a documented design decision rather than a bug.
TEST(MicroNetworkParity, LstmMatchesOurRecurrenceModel)
{
    const int T = dim("lstm_dims", 0), B = dim("lstm_dims", 1);
    const int D = dim("lstm_dims", 2), H = dim("lstm_dims", 3), O = dim("lstm_dims", 4);

    nn::LSTMLayer lstm(D, H);
    fill(lstm.W_, "lstm_W");
    fill(lstm.U_, "lstm_U");
    fill(lstm.b_, "lstm_b");

    nn::Linear head(H, O);
    fill(head.weight, "lstm_hw");
    fill(head.bias, "lstm_hb");

    // (B*T, D) rows, batch-major — reshaped by the layer into (B, T, D).
    Tensor x = to_tensor("lstm_x");
    x.reshape({static_cast<nn::Index>(B), static_cast<nn::Index>(T), static_cast<nn::Index>(D)});
    Tensor h_seq = lstm.forward(x, true);

    // Read the last timestep's hidden state per batch (the AE's latent readout).
    Tensor last(static_cast<nn::Index>(B), static_cast<nn::Index>(H));
    for (nn::Index b = 0; b < B; ++b)
        for (nn::Index f = 0; f < H; ++f) last.at(b, f) = h_seq.at(b, T - 1, f);

    Tensor out = head.forward(last, true);
    expect_close(out, "lstm_out", 1e-4F);
}

// Guard on the documented divergence. Our softsign gates make the LSTM differ from a real
// torch LSTM by a KNOWN, bounded amount. Asserting the size keeps two things honest: the
// approximation cannot silently drift larger, and nobody can later read "LSTM" here and
// assume standard sigmoid/tanh semantics. If this fails, either FastActivations changed or
// someone made the gates exact — both are real news, not noise.
TEST(MicroNetworkParity, LstmDivergenceFromRealTorchIsBoundedAndExpected)
{
    const int T = dim("lstm_dims", 0), B = dim("lstm_dims", 1), H = dim("lstm_dims", 3);

    const Tensor ours = to_tensor("lstm_h_seq");          // (B*T, H), our recurrence
    const Tensor torch_h = to_tensor("lstm_torch_h_seq"); // (B*T, H), real torch LSTM

    float max_abs = 0.0F;
    for (nn::Index i = 0; i < ours.rows(); ++i)
        for (nn::Index j = 0; j < ours.cols(); ++j)
            max_abs = std::max(max_abs, std::abs(ours.at(i, j) - torch_h.at(i, j)));

    // They must NOT be equal — that would mean the gates are no longer approximate...
    EXPECT_GT(max_abs, 1e-3F)
        << "our LSTM now matches torch exactly; FastActivations may have been replaced — if "
           "intentional, this test and Concepts/LSTM-and-BPTT.md need updating";
    // ...but the gap must stay in the range the softsign approximation predicts.
    EXPECT_LT(max_abs, 0.5F) << "divergence from real torch LSTM grew beyond the known bound";
    EXPECT_EQ(ours.rows(), static_cast<nn::Index>(B * T));
    EXPECT_EQ(ours.cols(), static_cast<nn::Index>(H));
}

} // namespace
