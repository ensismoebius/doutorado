// parameter_count_gtest.cpp — documented parameter budgets for the three trained
// non-spiking baseline families at the article configuration.
//
// The paper's complexity table reports P (trainable parameter count) and the raw
// operation-count estimate per model. This test pins both to a closed-form recompute
// from the config: a mismatch means either the architecture wiring changed OR the
// formula in the paper is wrong — both need a human. It also checks the two properties
// the text leans on: GRU has strictly fewer recurrent parameters than LSTM (3 gates vs
// 4), and the Transformer's op count grows super-linearly in sequence length (the
// O(S^2 * d_model) self-attention term, S = time_steps * window / frame).
//
// The recurrent estimates are additionally tied to the network the constructor really
// builds, and to hand-computed values. They counted one of the two layer stacks (and the
// frame width for every layer) until 2026-10-07, about 2.8x too low at H = 64, L = 1, while
// the Transformer's estimate always counted both stacks.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../lib/include/Meeting01Config.hpp"
#include "../lib/include/Meeting01Energy.hpp"
#include "../lib/include/Meeting01Metrics.hpp"
#include "../lib/include/Meeting01Stopping.hpp"
#include "../lib/include/Meeting01Training.hpp"
#include "models/autoencoder/ProtocolAutoencoder.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "models/gru/GRUAutoencoder.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "models/transformer/TransformerAutoencoder.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using meeting01::Meeting01Config;

namespace
{

Meeting01Config load_loso()
{
    const fs::path path =
        fs::path(__FILE__).parent_path().parent_path() / "profiles" / "meeting01-loso.json";
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    auto cfg = Meeting01Config::from_nested_json(j);
    cfg.validate();
    return cfg;
}

template <typename Model>
auto count_params(Model& m) -> std::size_t
{
    return meeting01::parameter_count(m.params());
}

// Independent recompute of the param count by walking the raw param span. Must equal
// meeting01::parameter_count — this catches a params() wiring regression.
template <typename Model>
auto manual_sum(Model& m) -> std::size_t
{
    std::size_t n = 0;
    for (auto* p : m.params())
        if (p != nullptr) n += static_cast<std::size_t>(p->size());
    return n;
}

// LSTMLayer / GRULayer: W (gH, in), U (gH, H), b (gH, 1)  — g = 4 (LSTM) / 3 (GRU).
auto recurrent_layer_params(int gates, int in_dim, int h) -> std::size_t
{
    return static_cast<std::size_t>(gates) * h * (in_dim + h + 1);
}
// Linear(in, out): weight (out, in) + bias (out).
auto linear_params(int in_dim, int out_dim) -> std::size_t
{
    return static_cast<std::size_t>(out_dim) * (in_dim + 1);
}

auto recurrent_ae_params(int gates, int D, int H, int Z, int L) -> std::size_t
{
    std::size_t p = 0;
    for (int l = 0; l < L; ++l) p += recurrent_layer_params(gates, (l == 0) ? D : H, H);
    p += linear_params(H, Z);                                             // enc_proj
    p += linear_params(Z, H);                                             // dec_expand
    for (int l = 0; l < L; ++l) p += recurrent_layer_params(gates, H, H); // dec layers
    p += linear_params(H, D);                                             // out_proj
    return p;
}

// Bias elements of a recurrent autoencoder: one vector (gates * H) per recurrent layer in
// the encoder and in the decoder, plus enc_proj (Z), dec_expand (H) and out_proj (D).
auto recurrent_ae_bias_count(int gates, int D, int H, int Z, int L) -> std::size_t
{
    return 2 * static_cast<std::size_t>(L) * gates * H + static_cast<std::size_t>(Z) + H + D;
}

// MACs of one forward pass over a window, recomputed from the BUILT network. Every weight
// multiplies once per sequence position, except the two latent projections (enc_proj on the
// final hidden state, dec_expand on the latent), which run once:
//     MACs = S * (W - 2HZ) + 2HZ,    W = parameters - biases.
// Taking W from the model the constructor really built, rather than from a formula, is what
// ties the estimate to the network: a missing stack, a mis-sized layer or an extra
// projection changes W and breaks the identity.
auto macs_from_built_model(std::size_t params, std::size_t biases, int S, int H, int Z)
    -> std::size_t
{
    const std::size_t once = 2 * static_cast<std::size_t>(H) * Z;
    return static_cast<std::size_t>(S) * (params - biases - once) + once;
}

auto transformer_block_params(int M, int d_ff) -> std::size_t
{
    std::size_t p = 0;
    p += 4 * linear_params(M, M);         // Q / K / V / O projections
    p += 2 * static_cast<std::size_t>(M); // ln1 gamma+beta
    p += linear_params(M, d_ff);          // ff1
    p += linear_params(d_ff, M);          // ff2
    p += 2 * static_cast<std::size_t>(M); // ln2 gamma+beta
    return p;
}

auto transformer_ae_params(int D, int M, int n_layers, int d_ff, int Z) -> std::size_t
{
    std::size_t p = 0;
    p += linear_params(D, M);                                                        // embed
    p += linear_params(M, Z);                                                        // to_latent
    p += linear_params(Z, M);                                                        // from_latent
    p += linear_params(M, D);                                                        // out_proj
    p += 2 * static_cast<std::size_t>(n_layers) * transformer_block_params(M, d_ff); // enc + dec
    return p;
}

} // namespace

TEST(ParameterCount, LstmAeMatchesClosedForm)
{
    const auto cfg = load_loso();
    const auto arch = meeting01::make_lstm_cfg(cfg);
    nn::models::lstm::LSTMAutoencoder model(arch);

    const std::size_t expected = recurrent_ae_params(
        4, arch.input_size, arch.hidden_size, arch.latent_size, arch.num_layers);
    EXPECT_EQ(count_params(model), expected);
    EXPECT_EQ(count_params(model), manual_sum(model));
}

TEST(ParameterCount, GruAeMatchesClosedFormAndIsSmallerThanLstm)
{
    const auto cfg = load_loso();
    const auto g = meeting01::make_gru_cfg(cfg);
    const auto l = meeting01::make_lstm_cfg(cfg);
    nn::models::gru::GRUAutoencoder gru(g);
    nn::models::lstm::LSTMAutoencoder lstm(l);

    const std::size_t expected =
        recurrent_ae_params(3, g.input_size, g.hidden_size, g.latent_size, g.num_layers);
    EXPECT_EQ(count_params(gru), expected);
    EXPECT_EQ(count_params(gru), manual_sum(gru));

    // Same D/H/Z/L → GRU's 3 gates cost strictly less than LSTM's 4.
    ASSERT_EQ(g.input_size, l.input_size);
    ASSERT_EQ(g.hidden_size, l.hidden_size);
    EXPECT_LT(count_params(gru), count_params(lstm));
}

TEST(ParameterCount, TransformerAeMatchesClosedFormAndHasNormParams)
{
    const auto cfg = load_loso();
    const auto t = meeting01::make_transformer_cfg(cfg);
    nn::models::transformer::TransformerAutoencoder model(t);

    const std::size_t expected =
        transformer_ae_params(t.input_size, t.d_model, t.n_layers, t.d_ff, t.latent_size);
    EXPECT_EQ(count_params(model), expected);
    EXPECT_EQ(count_params(model), manual_sum(model));

    // LayerNorm gamma/beta contribute 4*d_model per block, 2*n_layers blocks.
    const std::size_t norm_params =
        2 * static_cast<std::size_t>(t.n_layers) * 4 * static_cast<std::size_t>(t.d_model);
    const std::size_t no_norm =
        transformer_ae_params(t.input_size, t.d_model, t.n_layers, t.d_ff, t.latent_size) -
        norm_params;
    EXPECT_GT(count_params(model), no_norm);
}

TEST(ParameterCount, GruMacEstimateIsBelowLstmForMatchedDims)
{
    const auto cfg = load_loso();
    EXPECT_LT(meeting01::estimate_gru_macs(meeting01::make_gru_cfg(cfg)),
        meeting01::estimate_lstm_macs(meeting01::make_lstm_cfg(cfg)));
}

// Production shape AND a deeper, narrower one, so a wrong stack count cannot hide at L = 1.
TEST(ParameterCount, LstmMacEstimateMatchesTheBuiltNetwork)
{
    auto arch = meeting01::make_lstm_cfg(load_loso());
    for (const auto [hidden, layers] :
        {std::pair<int, int>{arch.hidden_size, arch.num_layers}, std::pair<int, int>{24, 3}})
    {
        arch.hidden_size = hidden;
        arch.num_layers = layers;
        nn::models::lstm::LSTMAutoencoder model(arch);
        const auto biases = recurrent_ae_bias_count(
            4, arch.input_size, arch.hidden_size, arch.latent_size, arch.num_layers);
        EXPECT_EQ(meeting01::estimate_lstm_macs(arch),
            macs_from_built_model(
                count_params(model), biases, arch.seq_len, arch.hidden_size, arch.latent_size))
            << "H=" << hidden << " L=" << layers;
    }
}

TEST(ParameterCount, GruMacEstimateMatchesTheBuiltNetwork)
{
    auto arch = meeting01::make_gru_cfg(load_loso());
    for (const auto [hidden, layers] :
        {std::pair<int, int>{arch.hidden_size, arch.num_layers}, std::pair<int, int>{24, 3}})
    {
        arch.hidden_size = hidden;
        arch.num_layers = layers;
        nn::models::gru::GRUAutoencoder model(arch);
        const auto biases = recurrent_ae_bias_count(
            3, arch.input_size, arch.hidden_size, arch.latent_size, arch.num_layers);
        EXPECT_EQ(meeting01::estimate_gru_macs(arch),
            macs_from_built_model(
                count_params(model), biases, arch.seq_len, arch.hidden_size, arch.latent_size))
            << "H=" << hidden << " L=" << layers;
    }
}

// Hand-computed values, not produced by the code under test. Production shape: window 256,
// 16 steps, frame 8, so S = 512 frames of D = 8.
//   LSTM, H = 64, Z = 16, L = 1, per position: encoder 4*64*(8+64) = 18432, decoder
//   4*64*(64+64) = 32768, output head 64*8 = 512 -> 51712; x 512 = 26476544; plus the two
//   latent projections 2*64*16 = 2048 -> 26478592.
//   L = 2 (S = 32, Z = 32): the second layer of EACH stack reads H, not D -> 3756032.
//   GRU, L = 1: 3 gates -> (13824 + 24576 + 512) * 512 + 2048 = 19924992.
TEST(ParameterCount, RecurrentMacEstimatesHitHandComputedValues)
{
    nn::models::lstm::LSTMAutoencoderConfig lstm;
    lstm.seq_len = 512;
    lstm.input_size = 8;
    lstm.hidden_size = 64;
    lstm.latent_size = 16;
    lstm.num_layers = 1;
    EXPECT_EQ(meeting01::estimate_lstm_macs(lstm), 26478592u);

    lstm.seq_len = 32;
    lstm.latent_size = 32;
    lstm.num_layers = 2;
    EXPECT_EQ(meeting01::estimate_lstm_macs(lstm), 3756032u);

    nn::models::gru::GRUAutoencoderConfig gru;
    gru.seq_len = 512;
    gru.input_size = 8;
    gru.hidden_size = 64;
    gru.latent_size = 16;
    gru.num_layers = 1;
    EXPECT_EQ(meeting01::estimate_gru_macs(gru), 19924992u);
}

// make_lstm_cfg yields latent_size -1 when neither the profile nor the layer specs give a
// latent width; cast to size_t that was a silently absurd cost, now it is an error.
TEST(ParameterCount, RecurrentMacEstimatesRefuseAnImpossibleArchitecture)
{
    nn::models::lstm::LSTMAutoencoderConfig lstm;
    lstm.seq_len = 512;
    lstm.input_size = 8;
    lstm.hidden_size = 64;
    lstm.latent_size = 16;
    lstm.num_layers = 1;
    ASSERT_NO_THROW(meeting01::estimate_lstm_macs(lstm));

    auto broken = lstm;
    broken.latent_size = -1;
    EXPECT_THROW(meeting01::estimate_lstm_macs(broken), std::invalid_argument);
    broken = lstm;
    broken.num_layers = 0;
    EXPECT_THROW(meeting01::estimate_lstm_macs(broken), std::invalid_argument);

    nn::models::gru::GRUAutoencoderConfig gru;
    gru.seq_len = 0;
    gru.input_size = 8;
    gru.hidden_size = 64;
    gru.latent_size = 16;
    gru.num_layers = 1;
    EXPECT_THROW(meeting01::estimate_gru_macs(gru), std::invalid_argument);
}

TEST(ParameterCount, TransformerMacEstimateIsSuperlinearInSeqLen)
{
    auto t = meeting01::make_transformer_cfg(load_loso());
    t.seq_len = 16;
    const std::size_t m1 = meeting01::estimate_transformer_macs(t);
    t.seq_len = 32;
    const std::size_t m2 = meeting01::estimate_transformer_macs(t);
    t.seq_len = 64;
    const std::size_t m3 = meeting01::estimate_transformer_macs(t);

    // A purely linear cost would give m2 - m1 == m3 - m2. The O(T^2) attention term
    // makes each doubling add strictly more than the previous one.
    EXPECT_GT(m3 - m2, m2 - m1);
}

// ---------------------------------------------------------------------------------------
// SNN-AE cost (results_format 5): what count_snn_ops counts, on a network small enough to
// count by hand. Window 16, encoder widths {8, 4}, T = 4, latent averaged over time and an
// analog decoder (the production design):
//
//   encoder  Linear 16->8 -> LifBPTT -> Linear 8->4 -> TimeMeanPool     4 rows in
//   decoder  Linear 4->8  -> LeakyReLU -> Linear 8->16                  1 row in
// ---------------------------------------------------------------------------------------

namespace
{

Meeting01Config tiny_snn_config(const std::string& pool, const std::string& neuron)
{
    Meeting01Config cfg = load_loso();
    cfg.dataset.window_size = 16;
    cfg.model.time_steps = 4;
    cfg.model.latent_dim = 4;
    cfg.model.snn_time_pool = pool;
    cfg.model.snn_decoder_neuron = neuron;
    return cfg;
}

using nn::models::autoencoder::ProtocolSpikingAutoencoder;

} // namespace

TEST(SnnCost, RendersTheTimeStageAndDecoderNeuronTheProfileAsks)
{
    using Spec = std::vector<std::string>;
    const auto pooled =
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 1.0F, {8, 4});
    EXPECT_EQ(
        pooled.encoder_layer_spec, (Spec{"linear:8:leaky", "linear:4:identity", "time_mean"}));
    EXPECT_EQ(pooled.decoder_layer_spec, (Spec{"linear:8:leaky_relu", "linear:output:identity"}));

    const auto per_step =
        meeting01::make_snn_cfg(tiny_snn_config("none", "lif"), 0.9F, 1.0F, {8, 4});
    EXPECT_EQ(per_step.encoder_layer_spec, (Spec{"linear:8:leaky", "linear:4:identity"}));
    EXPECT_EQ(per_step.decoder_layer_spec, (Spec{"linear:8:leaky", "linear:output:identity"}));

    // The ANN control never inherits the SNN's time stage or decoder choice.
    const auto twin = meeting01::make_ann_twin_cfg(tiny_snn_config("latent", "analog"), {8, 4});
    EXPECT_EQ(twin.encoder_layer_spec, per_step.encoder_layer_spec);
    EXPECT_EQ(twin.decoder_layer_spec, per_step.decoder_layer_spec);

    EXPECT_EQ(meeting01::snn_output_steps(tiny_snn_config("latent", "analog")), 1);
    EXPECT_EQ(meeting01::snn_output_steps(tiny_snn_config("output", "lif")), 1);
    EXPECT_EQ(meeting01::snn_output_steps(tiny_snn_config("none", "lif")), 4);
}

TEST(SnnCost, CountsAnAnalogInputForwardByHand)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    nn::Tensor x(4, 16); // direct-coded: analog values, never all 0/1
    for (int t = 0; t < 4; ++t)
        for (int d = 0; d < 16; ++d)
            x.at(t, d) = 0.1F * static_cast<float>((t * 16 + d) % 7) - 0.25F;

    model.reset_state();
    const ProtocolSpikingAutoencoder::Tensor input(x);
    const auto y = model.forward(input, false);
    ASSERT_EQ(y.rows(), 1) << "the latent average leaves one row: the decoder runs once";
    ASSERT_EQ(y.cols(), 16);

    const auto ops = meeting01::count_snn_ops(model, input, /*input_is_spikes=*/false, 1);
    const double spikes = model.encoder_.outputs[1].sum(); // the LifBPTT's spike train
    // MACs: Linear 16->8 on 4 rows (512) + LIF 4 x 8 (32) + the 1/T of 4 latents (4)
    //       + Linear 4->8 once (32) + Linear 8->16 once (128)
    EXPECT_DOUBLE_EQ(ops.macs, 512.0 + 32.0 + 4.0 + 32.0 + 128.0);
    // ACs: every spike feeds the 4 latent units; the time sum adds 4 rows x 4 values
    EXPECT_DOUBLE_EQ(ops.acs, 4.0 * spikes + 16.0);
    EXPECT_DOUBLE_EQ(ops.spikes, spikes);
    EXPECT_DOUBLE_EQ(ops.neuron_steps, 32.0);
}

TEST(SnnCost, TheEncoderIsPartOfTheWholeAndTheDecoderIsTheRest)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    nn::Tensor x(4, 16);
    for (int t = 0; t < 4; ++t)
        for (int d = 0; d < 16; ++d)
            x.at(t, d) = 0.1F * static_cast<float>((t * 16 + d) % 7) - 0.25F;
    model.reset_state();
    const ProtocolSpikingAutoencoder::Tensor input(x);
    (void) model.forward(input, false);

    const double whole = meeting01::snn_ae_energy(model, input, false, 1).total_nj();
    const double encoder = meeting01::snn_ae_encoder_energy(model, input, false, 1).total_nj();
    EXPECT_GT(encoder, 0.0);
    EXPECT_LT(encoder, whole);
}

TEST(SnnCost, EachEndOfTheRateIsCountedOnItsOwn)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    auto* lif = dynamic_cast<nn::LifBPTT*>(model.encoder_.layers[1].get());
    ASSERT_NE(lif, nullptr);
    nn::Tensor x(4, 16);
    x.setZero();
    const ProtocolSpikingAutoencoder::Tensor input(x);
    const auto count = [&](float threshold)
    {
        lif->voltage_threshold.at(0, 0) = threshold;
        model.reset_state();
        (void) model.forward(input, false);
        return meeting01::count_snn_ops(model, input, /*input_is_spikes=*/false, 1);
    };

    const auto silent = count(1000.0F);
    EXPECT_DOUBLE_EQ(silent.neuron_windows, 8.0) << "8 neurons, one window";
    EXPECT_DOUBLE_EQ(silent.silent_neuron_windows, 8.0);
    EXPECT_DOUBLE_EQ(silent.always_neuron_windows, 0.0);

    const auto always = count(-1000.0F);
    EXPECT_DOUBLE_EQ(always.silent_neuron_windows, 0.0);
    EXPECT_DOUBLE_EQ(always.always_neuron_windows, 8.0);
}

TEST(SnnCost, TrainedLifParametersAreReadFromTheSpikingLayers)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    auto* lif = dynamic_cast<nn::LifBPTT*>(model.encoder_.layers[1].get());
    ASSERT_NE(lif, nullptr);
    lif->voltage_threshold.at(0, 0) = -0.25F; // nothing bounds V_th, so the minimum must show it
    lif->resistance.at(0, 0) = 2.0F;
    lif->capacitance.at(0, 0) = 1.0F;

    const auto s = meeting01::summarize_lif_parameters(model);
    EXPECT_EQ(s.layers, 1);
    EXPECT_FLOAT_EQ(s.v_th_min, -0.25F);
    EXPECT_FLOAT_EQ(s.v_th_mean, -0.25F);
    EXPECT_FLOAT_EQ(s.beta_mean, std::exp(-1.0F / 2.0F));
}

TEST(SnnCost, CountsABinaryInputAsAccumulates)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    nn::Tensor x(4, 16); // poisson/latency-coded: 0/1 only
    double nonzero = 0.0;
    for (int t = 0; t < 4; ++t)
        for (int d = 0; d < 16; ++d)
        {
            x.at(t, d) = ((t + d) % 3 == 0) ? 1.0F : 0.0F;
            nonzero += x.at(t, d);
        }

    model.reset_state();
    const ProtocolSpikingAutoencoder::Tensor input(x);
    (void) model.forward(input, false);
    const auto ops = meeting01::count_snn_ops(model, input, /*input_is_spikes=*/true, 1);
    const double spikes = model.encoder_.outputs[1].sum();
    // The first layer costs nonzero x 8 ACs instead of 512 MACs.
    EXPECT_DOUBLE_EQ(ops.macs, 32.0 + 4.0 + 32.0 + 128.0);
    EXPECT_DOUBLE_EQ(ops.acs, nonzero * 8.0 + 4.0 * spikes + 16.0);
}

TEST(SnnCost, CountingBeforeAForwardIsRefused)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    nn::Tensor x(4, 16);
    x.setZero();
    EXPECT_THROW(
        (void) meeting01::count_snn_ops(model, ProtocolSpikingAutoencoder::Tensor(x), false, 1),
        std::logic_error);
}

// An encoder that fires nothing leaves an all-zero latent (zero-initialised biases), which
// looks binary. It is still an analog tensor, and the decoder still runs its dense passes:
// deciding "spikes" by value instead of by origin priced those passes at 0 (and failed this
// test whenever a random init happened to stay silent).
TEST(SnnCost, ASilentEncoderStillPaysForItsAnalogLayers)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 1000.0F, {8, 4}));
    nn::Tensor x(4, 16);
    for (int t = 0; t < 4; ++t)
        for (int d = 0; d < 16; ++d) x.at(t, d) = 0.01F * static_cast<float>(d);

    model.reset_state();
    const ProtocolSpikingAutoencoder::Tensor input(x);
    (void) model.forward(input, false);
    const auto ops = meeting01::count_snn_ops(model, input, false, 1);
    ASSERT_EQ(ops.spikes, 0.0) << "v_th = 1000 must keep the encoder silent";
    EXPECT_DOUBLE_EQ(ops.macs, 512.0 + 32.0 + 4.0 + 32.0 + 128.0);
    EXPECT_DOUBLE_EQ(ops.acs, 16.0); // the time sum only
}

TEST(SnnCost, AnInputTakenForSpikesMustBeBinary)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    nn::Tensor x(4, 16);
    x.setZero();
    x.at(0, 0) = 0.5F;
    model.reset_state();
    const ProtocolSpikingAutoencoder::Tensor input(x);
    (void) model.forward(input, false);
    EXPECT_THROW((void) meeting01::count_snn_ops(model, input, true, 1), std::logic_error);
}

TEST(SnnCost, InputIsSpikesFollowsEncodingAndArchitecture)
{
    EXPECT_FALSE(meeting01::snn_input_is_spikes("direct", "dense"));
    EXPECT_TRUE(meeting01::snn_input_is_spikes("poisson", "dense"));
    EXPECT_TRUE(meeting01::snn_input_is_spikes("latency", "dense"));
    EXPECT_FALSE(meeting01::snn_input_is_spikes("poisson", "conv1d"));  // smoothed: analog
    EXPECT_TRUE(meeting01::snn_input_is_spikes("direct", "recurrent")); // a LIF's spikes
    EXPECT_THROW((void) meeting01::snn_input_is_spikes("rate", "dense"), std::invalid_argument);
    EXPECT_THROW((void) meeting01::snn_input_is_spikes("direct", "cnn"), std::invalid_argument);
}

TEST(SnnCost, AnalyticEstimateRunsTheDecoderOnceAfterALatentAverage)
{
    // encoder 16*8 + 8*4 = 160 per step, decoder 4*8 + 8*16 = 160 per run.
    EXPECT_EQ(meeting01::estimate_snn_macs(16, std::vector<int>{8, 4}, 4), 4u * 160u + 4u * 160u);
    EXPECT_EQ(meeting01::estimate_snn_macs(16, std::vector<int>{8, 4}, 4, 1), 4u * 160u + 160u);
    // The first layer (16 -> 8 = 128 MACs) once instead of 4 times: 3 x 128 fewer.
    EXPECT_EQ(meeting01::estimate_snn_macs(16, std::vector<int>{8, 4}, 4, 1, 1),
        4u * 160u + 160u - 3u * 128u);
    EXPECT_EQ(meeting01::estimate_snn_macs(16, std::vector<int>{8, 4}, 4, 1, 4),
        meeting01::estimate_snn_macs(16, std::vector<int>{8, 4}, 4, 1));
    EXPECT_THROW(
        meeting01::estimate_snn_macs(16, std::vector<int>{8, 4}, 4, 1, 0), std::invalid_argument);
    EXPECT_THROW(
        meeting01::estimate_snn_macs(16, std::vector<int>{8, 4}, 4, 1, 5), std::invalid_argument);
}

TEST(SnnCost, EnergyAndTransformCountsUseTheDeclaredConstants)
{
    EXPECT_FLOAT_EQ(meeting01::arith_energy_nj(1000.0, 1000.0), 5.5F); // 4.6 + 0.9 nJ
    EXPECT_DOUBLE_EQ(meeting01::count_architecture_transform_macs("dense", 64), 0.0);
    EXPECT_DOUBLE_EQ(meeting01::count_architecture_transform_macs("conv1d", 64), 192.0);
    EXPECT_DOUBLE_EQ(meeting01::count_architecture_transform_macs("recurrent", 64), 64.0);
    EXPECT_THROW(
        (void) meeting01::count_architecture_transform_macs("cnn", 64), std::invalid_argument);
}

// --- memory-aware energy (Meeting01Energy.hpp, results_format 7) -----------------------------

// Horowitz 2014, Fig. 1.1.9: 10 / 20 / 100 pJ at 8 KB / 32 KB / 1 MB, straight lines between,
// 10 pJ below 8 KB, the last line extended to 2 MB and refused past it.
TEST(MemoryAwareEnergy, SramPriceFollowsHorowitzsThreePoints)
{
    EXPECT_DOUBLE_EQ(meeting01::sram_access_pj(8.0 * 1024.0), 10.0);
    EXPECT_DOUBLE_EQ(meeting01::sram_access_pj(32.0 * 1024.0), 20.0);
    EXPECT_DOUBLE_EQ(meeting01::sram_access_pj(1024.0 * 1024.0), 100.0);
    EXPECT_DOUBLE_EQ(meeting01::sram_access_pj(20.0 * 1024.0), 15.0); // halfway 8 -> 32 KB
    EXPECT_DOUBLE_EQ(meeting01::sram_access_pj(64.0), 10.0);          // a 16-word memory
    EXPECT_GT(meeting01::sram_access_pj(1.5 * 1024.0 * 1024.0), 100.0);
    EXPECT_THROW((void) meeting01::sram_access_pj(3.0 * 1024.0 * 1024.0), std::out_of_range);
    EXPECT_THROW((void) meeting01::sram_access_pj(0.0), std::invalid_argument);
}

// Lemaire et al. 2022, Eqs. 2/4/6/12/17, for a 3 -> 4 layer on 2 rows, every memory < 8 KB.
TEST(MemoryAwareEnergy, DenseLayerCountsLemairesFnnTerms)
{
    meeting01::EnergyLedger e;
    meeting01::add_dense_fc(e, 2.0, 3.0, 4.0, /*writes_output=*/true);
    EXPECT_DOUBLE_EQ(e.fp_macs, 24.0);  // 2 x 3 x 4
    EXPECT_DOUBLE_EQ(e.fp_adds, 8.0);   // a bias per output and row
    EXPECT_DOUBLE_EQ(e.int_adds, 14.0); // 2 x (3 + 4)
    EXPECT_DOUBLE_EQ(e.rd_in, 6.0);     // 2 x 3
    EXPECT_DOUBLE_EQ(e.rd_param, 32.0); // 2 x (3 + 1) x 4
    EXPECT_DOUBLE_EQ(e.wr_out, 8.0);
    EXPECT_DOUBLE_EQ(e.mem_pj, 46.0 * 10.0);
    EXPECT_DOUBLE_EQ(e.ops_pj(), 24.0 * 4.6 + 8.0 * 0.9 + 14.0 * 0.1);
    EXPECT_DOUBLE_EQ(e.total_nj_at(1024.0 * 1024.0), (e.ops_pj() + 46.0 * 100.0) / 1000.0);
}

// Lemaire et al. 2022, Eqs. 2/5/8/10/15/17: 5 spikes into a 3 -> 4 layer over 2 steps. Each
// spike updates the 4 potentials it reaches; every step adds a bias to each potential.
TEST(MemoryAwareEnergy, SpikingLayerPaysPerEventAndPerStep)
{
    meeting01::EnergyLedger e;
    meeting01::add_spiking_fc(e, 5.0, 2.0, 3.0, 4.0, /*writes_output=*/false);
    EXPECT_DOUBLE_EQ(e.fp_macs, 0.0);
    EXPECT_DOUBLE_EQ(e.fp_adds, 20.0 + 8.0);
    EXPECT_DOUBLE_EQ(e.int_adds, 20.0);
    EXPECT_DOUBLE_EQ(e.rd_in, 5.0);
    EXPECT_DOUBLE_EQ(e.rd_param, 28.0);
    EXPECT_DOUBLE_EQ(e.rd_pot, 28.0);
    EXPECT_DOUBLE_EQ(e.wr_pot, 28.0);
    EXPECT_DOUBLE_EQ(e.wr_out, 0.0);

    // No spike still costs the per-step bias and its potential traffic: silence is not free.
    meeting01::EnergyLedger silent;
    meeting01::add_spiking_fc(silent, 0.0, 2.0, 3.0, 4.0, false);
    EXPECT_DOUBLE_EQ(silent.fp_adds, 8.0);
    EXPECT_DOUBLE_EQ(silent.rd_pot + silent.wr_pot, 16.0);
}

// The SNN-AE of SnnCost.CountsAnAnalogInputForwardByHand (window 16, T 4, widths {8, 4},
// latent average, analog decoder), layer by layer with s = the LIF's spikes:
//   Linear 16->8, analog, into the LIF   macs 512, adds 32, int 96, rd_in 64, rd_param 544
//   LifBPTT 4 x 8 (dense input)          macs 32, adds 32 + s, rd/wr_pot 32 each, wr_out s
//   Linear 8->4 on s spikes, 4 steps     adds 4s + 16, int 4s, rd_in s, rd_param/rd_pot/
//                                        wr_pot 4s + 16 each, wr_out 16 (no LIF after it)
//   TimeMeanPool 4 x 4 -> 1 x 4          adds 16, macs 4, rd_in 16, wr_out 4
//   Linear 4->8, LeakyReLU, Linear 8->16 macs 160, adds 8 + 8 + 16, muls 8, int 36,
//                                        rd_in 12, rd_param 40 + 144, wr_out 24
// Every memory is under 8 KB: 10 pJ per access.
TEST(MemoryAwareEnergy, SnnAeWalkMatchesAHandCount)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    nn::Tensor x(4, 16);
    for (int t = 0; t < 4; ++t)
        for (int d = 0; d < 16; ++d)
            x.at(t, d) = 0.1F * static_cast<float>((t * 16 + d) % 7) - 0.25F;
    model.reset_state();
    const ProtocolSpikingAutoencoder::Tensor input(x);
    (void) model.forward(input, false);
    const double s = model.encoder_.outputs[1].sum();

    const auto e = meeting01::snn_ae_energy(model, input, /*input_is_spikes=*/false, 1);
    EXPECT_DOUBLE_EQ(e.fp_macs, 708.0); // = count_snn_ops' macs for the same pass
    EXPECT_DOUBLE_EQ(e.fp_adds, 128.0 + 5.0 * s);
    EXPECT_DOUBLE_EQ(e.fp_muls, 8.0);
    EXPECT_DOUBLE_EQ(e.int_adds, 132.0 + 4.0 * s);
    EXPECT_DOUBLE_EQ(e.rd_in, 92.0 + s);
    EXPECT_DOUBLE_EQ(e.rd_param, 744.0 + 4.0 * s);
    EXPECT_DOUBLE_EQ(e.rd_pot, 48.0 + 4.0 * s);
    EXPECT_DOUBLE_EQ(e.wr_pot, 48.0 + 4.0 * s);
    EXPECT_DOUBLE_EQ(e.wr_out, 44.0 + s);
    EXPECT_DOUBLE_EQ(e.mem_pj, (976.0 + 14.0 * s) * 10.0);
}

// The matched ANN-AE of the same widths: one pass of the window, no potentials.
//   16->8 +leaky, 8->4, 4->8 +leaky, 8->16
TEST(MemoryAwareEnergy, AnnAeWalkMatchesAHandCount)
{
    nn::models::autoencoder::ProtocolAutoencoder model(
        meeting01::make_ann_twin_cfg(tiny_snn_config("latent", "analog"), {8, 4}));
    const auto e = meeting01::ann_ae_energy(model, 16);
    EXPECT_DOUBLE_EQ(e.fp_macs, static_cast<double>(meeting01::estimate_snn_macs(16, {8, 4}, 1)));
    EXPECT_DOUBLE_EQ(e.fp_macs, 320.0);
    EXPECT_DOUBLE_EQ(e.fp_adds, 36.0 + 16.0); // biases + the two leaky ReLUs' comparisons
    EXPECT_DOUBLE_EQ(e.fp_muls, 16.0);
    EXPECT_DOUBLE_EQ(e.int_adds, 72.0);
    EXPECT_DOUBLE_EQ(e.rd_in, 36.0);
    EXPECT_DOUBLE_EQ(e.rd_param, 356.0);
    EXPECT_DOUBLE_EQ(e.rd_pot + e.wr_pot, 0.0);
    EXPECT_DOUBLE_EQ(e.wr_out, 36.0);
}

// results_format 12 (R3 of the round-4 review): the control's silent share counts the ENCODER's
// leaky-ReLU outputs <= 0.01, per (unit, window). Zero weights make each unit's pre-activation
// its bias: {-1, 0, 0.005, 0.01, 0.02, 1, -0.5, 2} -> leaky ReLU -> {-0.01, 0, 0.005, 0.01, 0.02,
// 1, -0.005, 2}, five of them <= 0.01. The latent layer is identity, so it is not counted.
TEST(AnnHiddenShare, CountsEncoderLeakyReluOutputsAtOrBelowTheSilentCut)
{
    nn::models::autoencoder::ProtocolAutoencoder model(
        meeting01::make_ann_twin_cfg(tiny_snn_config("latent", "analog"), {8, 4}));
    auto* first = dynamic_cast<nn::Linear*>(model.encoder_.layers.at(0).get());
    ASSERT_NE(first, nullptr);
    ASSERT_NE(dynamic_cast<const nn::LeakyReLU*>(model.encoder_.layers.at(1).get()), nullptr);
    first->weight.setZero();
    const std::vector<float> biases{-1.0F, 0.0F, 0.005F, 0.01F, 0.02F, 1.0F, -0.5F, 2.0F};
    for (int j = 0; j < 8; ++j) first->bias.at(j, 0) = biases[static_cast<std::size_t>(j)];
    nn::Tensor x(1, 16);
    for (int d = 0; d < 16; ++d) x.at(0, d) = 0.1F * static_cast<float>(d);
    (void) model.forward(nn::models::autoencoder::ProtocolAutoencoder::Tensor(x), false);
    const auto c = meeting01::count_ann_hidden(model);
    EXPECT_DOUBLE_EQ(c.unit_windows, 8.0);
    EXPECT_DOUBLE_EQ(c.silent_unit_windows, 5.0);
}

// Loud, not a wrong share: before any forward pass there is nothing to count, and two rows
// would be two windows (or frames) counted as one.
TEST(AnnHiddenShare, RefusesACallBeforeForwardAndMoreThanOneWindowPerPass)
{
    nn::models::autoencoder::ProtocolAutoencoder model(
        meeting01::make_ann_twin_cfg(tiny_snn_config("latent", "analog"), {8, 4}));
    EXPECT_THROW((void) meeting01::count_ann_hidden(model), std::logic_error);
    nn::Tensor x(2, 16);
    x.setZero();
    (void) model.forward(nn::models::autoencoder::ProtocolAutoencoder::Tensor(x), false);
    EXPECT_THROW((void) meeting01::count_ann_hidden(model), std::logic_error);
}

// The sequence baselines' products are the ones their MAC estimates count (the energy model
// adds the elementwise cell/attention work on top, as multiplies and adds, never as MACs);
// the Transformer adds only the 1/S of its mean over positions.
TEST(MemoryAwareEnergy, SequenceModelsMultiplyWhatTheirMacEstimatesCount)
{
    nn::models::lstm::LSTMAutoencoderConfig lstm;
    lstm.seq_len = 32;
    lstm.input_size = 8;
    lstm.hidden_size = 64;
    lstm.latent_size = 16;
    lstm.num_layers = 2;
    EXPECT_DOUBLE_EQ(meeting01::lstm_ae_energy(lstm).fp_macs,
        static_cast<double>(meeting01::estimate_lstm_macs(lstm)));

    nn::models::gru::GRUAutoencoderConfig gru;
    gru.seq_len = 32;
    gru.input_size = 8;
    gru.hidden_size = 64;
    gru.latent_size = 16;
    gru.num_layers = 2;
    EXPECT_DOUBLE_EQ(meeting01::gru_ae_energy(gru).fp_macs,
        static_cast<double>(meeting01::estimate_gru_macs(gru)));

    nn::models::transformer::TransformerAutoencoderConfig tf;
    tf.seq_len = 32;
    tf.input_size = 8;
    tf.d_model = 32;
    tf.n_heads = 4;
    tf.n_layers = 2;
    tf.d_ff = 64;
    tf.latent_size = 16;
    EXPECT_DOUBLE_EQ(meeting01::transformer_ae_energy(tf).fp_macs,
        static_cast<double>(meeting01::estimate_transformer_macs(tf)) + 32.0);

    // The LSTM's per-step cell state is read and written once per (position, unit, layer).
    const auto e = meeting01::lstm_ae_energy(lstm);
    EXPECT_DOUBLE_EQ(e.rd_pot, 32.0 * 64.0 * 4.0);
    EXPECT_DOUBLE_EQ(e.wr_pot, 32.0 * 64.0 * 4.0);
}

TEST(MemoryAwareEnergy, EncodersCostWhatTheyCompute)
{
    // direct: the first layer reads the window itself.
    EXPECT_DOUBLE_EQ(meeting01::encoding_energy("direct", 4.0, 2, 0.0, true).total_nj(), 0.0);
    // poisson: per (value, step) a read, an LCG step, a comparison; events written per spike.
    const auto p = meeting01::encoding_energy("poisson", 4.0, 2, 3.0, /*event_driven=*/true);
    EXPECT_DOUBLE_EQ(p.rd_in, 8.0);
    EXPECT_DOUBLE_EQ(p.int_muls, 8.0);
    EXPECT_DOUBLE_EQ(p.int_adds, 8.0);
    EXPECT_DOUBLE_EQ(p.fp_adds, 8.0);
    EXPECT_DOUBLE_EQ(p.wr_out, 3.0);
    // A dense consumer reads every encoded value, so all of them are written.
    EXPECT_DOUBLE_EQ(meeting01::encoding_energy("poisson", 4.0, 2, 3.0, false).wr_out, 8.0);
    const auto l = meeting01::encoding_energy("latency", 4.0, 2, 4.0, true);
    EXPECT_DOUBLE_EQ(l.rd_in, 4.0);
    EXPECT_DOUBLE_EQ(l.fp_muls, 4.0);
    EXPECT_DOUBLE_EQ(l.int_adds, 8.0);
    EXPECT_THROW(
        (void) meeting01::encoding_energy("rate", 4.0, 2, 0.0, true), std::invalid_argument);
    EXPECT_THROW(
        (void) meeting01::architecture_transform_energy("cnn", 4.0, 2, 0.0), std::invalid_argument);
}

// The paper's break-even figure is drawn by a Python port of these building blocks
// (first_stage_energy_nj in scripts/pipeline/meeting01/meeting01_results.py). Its self-test
// (09_meeting01_energy_breakeven.py --self-test) pins the SAME three numbers, so a change to
// either side breaks one of the two tests instead of the figure drifting from the harness.
TEST(MemoryAwareEnergy, FirstStageOfTheBreakEvenFigure)
{
    using meeting01::EnergyLedger;
    // The ANN-AE's first layer, 256 -> 59, once: Linear (output written) + leaky ReLU.
    EnergyLedger ann;
    meeting01::add_dense_fc(ann, 1.0, 256.0, 59.0, /*writes_output=*/true);
    meeting01::add_nonlinearity(ann, 59.0, meeting01::Nonlinearity::ComparisonAndMultiply);
    EXPECT_NEAR(ann.total_nj(), 409.5424320690525, 1e-9);

    // A Poisson window, 300 spikes over T = 4, into the dense transform and the same Linear,
    // then its LIF (no output spike charged).
    EnergyLedger poisson = meeting01::encoding_energy("poisson", 256.0, 4, 300.0, true);
    poisson += meeting01::architecture_transform_energy("dense", 256.0, 4, 0.0);
    meeting01::add_spiking_fc(poisson, 300.0, 4.0, 256.0, 59.0, /*writes_output=*/false);
    meeting01::add_lif(poisson, 4.0, 59.0, 0.0, /*readout=*/false, /*dense_input=*/false);
    EXPECT_NEAR(poisson.total_nj(), 796.4763554435484, 1e-9);

    // The direct code through the recurrent transform, which emits the same 300 spikes.
    EnergyLedger recurrent = meeting01::encoding_energy("direct", 256.0, 4, 0.0, false);
    recurrent += meeting01::architecture_transform_energy("recurrent", 256.0, 4, 300.0);
    meeting01::add_spiking_fc(recurrent, 300.0, 4.0, 256.0, 59.0, /*writes_output=*/false);
    meeting01::add_lif(recurrent, 4.0, 59.0, 0.0, /*readout=*/false, /*dense_input=*/false);
    EXPECT_NEAR(recurrent.total_nj(), 818.6599554435484, 1e-9);
}

TEST(MemoryAwareEnergy, SetEnergySplitsOpsAndMemory)
{
    meeting01::EnergyLedger e;
    meeting01::add_dense_fc(e, 2.0, 3.0, 4.0, true);
    meeting01::RunMetrics m;
    meeting01::set_energy(m, e);
    EXPECT_DOUBLE_EQ(m.energy_nj, m.energy_ops_nj + m.energy_mem_nj);
    EXPECT_DOUBLE_EQ(m.mem_accesses, 46.0);
    EXPECT_DOUBLE_EQ(m.energy_nj_sram8k, m.energy_nj); // every memory here is under 8 KB
    EXPECT_GT(m.energy_nj_sram1m, m.energy_nj_sram32k);
    // No repeated input here, so the as-executed check is the same total ...
    EXPECT_DOUBLE_EQ(m.energy_nj_as_executed, m.energy_nj);
    // ... until an SNN-AE evaluation overwrites it with its own every-step ledger.
    meeting01::EnergyLedger twice = e;
    twice += e;
    meeting01::set_energy_as_executed(m, twice);
    EXPECT_DOUBLE_EQ(m.energy_nj_as_executed, 2.0 * m.energy_nj);
}

// --- a repeated input computed once (results_format 8) --------------------------------------

namespace
{
// The SnnCost model (window 16, T 4, widths {8, 4}) on a DIRECT-coded window: the same analog
// row at all 4 steps, as encode_sample builds it.
ProtocolSpikingAutoencoder::Tensor repeated_window(ProtocolSpikingAutoencoder& model)
{
    nn::Tensor x(4, 16);
    for (int t = 0; t < 4; ++t)
        for (int d = 0; d < 16; ++d) x.at(t, d) = 0.1F * static_cast<float>(d % 7) - 0.25F;
    model.reset_state();
    ProtocolSpikingAutoencoder::Tensor input(x);
    (void) model.forward(input, false);
    return input;
}
} // namespace

TEST(RepeatedInput, OnlyTheDirectCodeThroughDenseOrConv1dRepeats)
{
    EXPECT_TRUE(meeting01::snn_input_repeats_every_step("direct", "dense"));
    EXPECT_TRUE(meeting01::snn_input_repeats_every_step("direct", "conv1d"));
    // The recurrent transform's LIF turns the repeated window into spikes that change.
    EXPECT_FALSE(meeting01::snn_input_repeats_every_step("direct", "recurrent"));
    EXPECT_FALSE(meeting01::snn_input_repeats_every_step("poisson", "dense"));
    EXPECT_FALSE(meeting01::snn_input_repeats_every_step("latency", "conv1d"));
    EXPECT_THROW(
        (void) meeting01::snn_input_repeats_every_step("rate", "dense"), std::invalid_argument);
    EXPECT_THROW(
        (void) meeting01::snn_input_repeats_every_step("direct", "cnn"), std::invalid_argument);
}

// The first Linear (16 -> 8 into the LIF) once instead of on each of the 4 rows: 128 MACs
// instead of 512; everything after it is the same pass.
TEST(RepeatedInput, CountsTheFirstLayerOnce)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    const auto input = repeated_window(model);
    const auto as_run = meeting01::count_snn_ops(model, input, false, 1);
    const auto once = meeting01::count_snn_ops(model, input, false, 4);
    EXPECT_DOUBLE_EQ(as_run.macs, 512.0 + 32.0 + 4.0 + 32.0 + 128.0);
    EXPECT_DOUBLE_EQ(once.macs, 128.0 + 32.0 + 4.0 + 32.0 + 128.0);
    EXPECT_DOUBLE_EQ(once.acs, as_run.acs);
    EXPECT_DOUBLE_EQ(once.spikes, as_run.spikes);
}

// The first layer as it runs (4 rows, current straight into the LIF) against once (1 row,
// current written once and read back at each of the 4 steps):
//                 macs  adds  int   rd_in          rd_param        wr_out
//   as executed    512    32   96      64           512 + 32            0
//   once           128     8   24   16 + 4 x 8      128 + 8             8
// Every memory is under 8 KB, so the accesses' difference (-416) is -4160 pJ.
TEST(RepeatedInput, PricesTheFirstLayerOncePlusItsStoredCurrent)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    const auto input = repeated_window(model);
    const auto run = meeting01::snn_ae_energy(model, input, false, 1);
    const auto once = meeting01::snn_ae_energy(model, input, false, 4);
    EXPECT_DOUBLE_EQ(run.fp_macs - once.fp_macs, 384.0);
    EXPECT_DOUBLE_EQ(once.fp_macs, meeting01::count_snn_ops(model, input, false, 4).macs);
    EXPECT_DOUBLE_EQ(run.fp_adds - once.fp_adds, 24.0);
    EXPECT_DOUBLE_EQ(run.int_adds - once.int_adds, 72.0);
    EXPECT_DOUBLE_EQ(run.rd_in - once.rd_in, 16.0);
    EXPECT_DOUBLE_EQ(run.rd_param - once.rd_param, 408.0);
    EXPECT_DOUBLE_EQ(once.wr_out - run.wr_out, 8.0);
    EXPECT_DOUBLE_EQ(once.rd_pot, run.rd_pot); // the LIF still integrates at every step
    EXPECT_DOUBLE_EQ(once.wr_pot, run.wr_pot);
    EXPECT_DOUBLE_EQ(run.mem_pj - once.mem_pj, 4160.0);
}

TEST(RepeatedInput, RefusesWhatItCannotPriceOnce)
{
    ProtocolSpikingAutoencoder model(
        meeting01::make_snn_cfg(tiny_snn_config("latent", "analog"), 0.9F, 0.5F, {8, 4}));
    const auto input = repeated_window(model);
    // 4 rows do not split into 3 steps.
    EXPECT_THROW((void) meeting01::snn_ae_energy(model, input, false, 3), std::logic_error);
    EXPECT_THROW((void) meeting01::count_snn_ops(model, input, false, 3), std::logic_error);
    // Spikes are not an analog repeated input.
    EXPECT_THROW((void) meeting01::snn_ae_energy(model, input, true, 4), std::logic_error);
    EXPECT_THROW((void) meeting01::snn_ae_energy(model, input, false, 0), std::invalid_argument);
    EXPECT_THROW((void) meeting01::count_snn_ops(model, input, false, 0), std::invalid_argument);
}

// --- the stopping rule's mean predictor (Meeting01Stopping.hpp) ------------------------------

namespace
{
using Triple = std::tuple<nn::Tensor, nn::Tensor, nn::Tensor>;
Triple window(const std::vector<float>& target, const std::vector<float>& mask)
{
    nn::Tensor t(1, static_cast<nn::Index>(target.size()));
    nn::Tensor m(1, static_cast<nn::Index>(mask.size()));
    for (std::size_t k = 0; k < target.size(); ++k)
    {
        t.at(static_cast<nn::Index>(k)) = target[k];
        m.at(static_cast<nn::Index>(k)) = mask[k];
    }
    return {t, t, m};
}
} // namespace

// Fit targets [1, 3] and [3, 5] -> the mean predicts [2, 4]. Monitor A = [2, 6] with its second
// value masked out (error 0), B = [0, 4] (error (4 + 0) / 2 = 2). Chunked like Trainer's
// validation: one window per chunk -> (0 + 2) / 2 = 1; both in one chunk -> the masked mean
// over the chunk, 4 / 3.
TEST(StoppingRule, MeanPredictorLossIsScoredLikeTheTrainersValidation)
{
    const std::vector<Triple> fit{window({1, 3}, {1, 1}), window({3, 5}, {1, 1})};
    const std::vector<Triple> monitor{window({2, 6}, {1, 0}), window({0, 4}, {1, 1})};
    EXPECT_DOUBLE_EQ(meeting01::input_independent_monitor_loss(fit, monitor, 1), 1.0);
    EXPECT_DOUBLE_EQ(meeting01::input_independent_monitor_loss(fit, monitor, 2), 4.0 / 3.0);
    EXPECT_THROW(
        (void) meeting01::input_independent_monitor_loss(fit, {}, 1), std::invalid_argument);
    EXPECT_THROW(
        (void) meeting01::input_independent_monitor_loss(fit, {window({1, 2, 3}, {1, 1, 1})}, 1),
        std::invalid_argument);
}

TEST(StoppingRule, ArmsAtTheConfiguredShareBelowTheMeanPredictor)
{
    Meeting01Config cfg;
    cfg.training.early_stop_min_skill = 0.1F;
    const auto rule = meeting01::make_stopping_rule(cfg, 2.0);
    EXPECT_FLOAT_EQ(rule.trivial_loss, 2.0F);
    EXPECT_FLOAT_EQ(rule.arm_below, 1.8F);
    cfg.training.early_stop_min_skill = -1.0F; // unset: a config built in code must set it
    EXPECT_THROW((void) meeting01::make_stopping_rule(cfg, 2.0), std::invalid_argument);
    cfg.training.early_stop_min_skill = 0.1F;
    EXPECT_THROW((void) meeting01::make_stopping_rule(cfg, 0.0), std::logic_error);
}
