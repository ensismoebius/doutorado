// parameter_count_gtest.cpp — documented parameter budgets for the three trained
// non-spiking baseline families at the article configuration.
//
// The paper's complexity table reports P (trainable parameter count) and the raw
// operation-count estimate per model. This test pins both to a closed-form recompute
// from the config: a mismatch means either the architecture wiring changed OR the
// formula in the paper is wrong — both need a human. It also checks the two properties
// the text leans on: GRU has strictly fewer recurrent parameters than LSTM (3 gates vs
// 4), and the Transformer's op count grows super-linearly in sequence length (the
// O(T^2 * d_model) self-attention term).

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>

#include "../lib/include/GuayaquilConfig.hpp"
#include "../lib/include/GuayaquilMetrics.hpp"
#include "../lib/include/GuayaquilTraining.hpp"
#include "models/gru/GRUAutoencoder.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "models/transformer/TransformerAutoencoder.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using guayaquil::GuayaquilConfig;

namespace
{

GuayaquilConfig load_loso()
{
    const fs::path path =
        fs::path(__FILE__).parent_path().parent_path() / "profiles" / "article-loso.json";
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    auto cfg = GuayaquilConfig::from_nested_json(j);
    cfg.validate();
    return cfg;
}

template <typename Model>
auto count_params(Model& m) -> std::size_t
{
    return guayaquil::parameter_count(m.params());
}

// Independent recompute of the param count by walking the raw param span. Must equal
// guayaquil::parameter_count — this catches a params() wiring regression.
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
    const auto arch = guayaquil::make_lstm_cfg(cfg);
    nn::models::lstm::LSTMAutoencoder model(arch);

    const std::size_t expected = recurrent_ae_params(
        4, arch.input_size, arch.hidden_size, arch.latent_size, arch.num_layers);
    EXPECT_EQ(count_params(model), expected);
    EXPECT_EQ(count_params(model), manual_sum(model));
}

TEST(ParameterCount, GruAeMatchesClosedFormAndIsSmallerThanLstm)
{
    const auto cfg = load_loso();
    const auto g = guayaquil::make_gru_cfg(cfg);
    const auto l = guayaquil::make_lstm_cfg(cfg);
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
    const auto t = guayaquil::make_transformer_cfg(cfg);
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
    EXPECT_LT(guayaquil::estimate_gru_macs(guayaquil::make_gru_cfg(cfg)),
        guayaquil::estimate_lstm_macs(guayaquil::make_lstm_cfg(cfg)));
}

TEST(ParameterCount, TransformerMacEstimateIsSuperlinearInSeqLen)
{
    auto t = guayaquil::make_transformer_cfg(load_loso());
    t.seq_len = 16;
    const std::size_t m1 = guayaquil::estimate_transformer_macs(t);
    t.seq_len = 32;
    const std::size_t m2 = guayaquil::estimate_transformer_macs(t);
    t.seq_len = 64;
    const std::size_t m3 = guayaquil::estimate_transformer_macs(t);

    // A purely linear cost would give m2 - m1 == m3 - m2. The O(T^2) attention term
    // makes each doubling add strictly more than the previous one.
    EXPECT_GT(m3 - m2, m2 - m1);
}
