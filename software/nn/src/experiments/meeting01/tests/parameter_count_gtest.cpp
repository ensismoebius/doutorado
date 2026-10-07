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

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "../lib/include/Meeting01Config.hpp"
#include "../lib/include/Meeting01Metrics.hpp"
#include "../lib/include/Meeting01Training.hpp"
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
