#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01DatasetSplit.hpp" // WindowMetadata
#include "Meeting01EpochHistory.hpp"
#include "Meeting01RunMetrics.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "models/gru/GRUAutoencoder.hpp"
#include "models/gru/GRUAutoencoderConfig.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "models/transformer/TransformerAutoencoder.hpp"
#include "models/transformer/TransformerAutoencoderConfig.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"

namespace meeting01
{

using Tensor = nn::Tensor;

// `hidden_size_override`/`num_layers_override`, when > 0, override the profile's fixed
// lstm_hidden_size / derived layer count with a genome-driven value — the bridge the
// recurrent architecture search (Meeting01RecurrentGaGenome::to_lstm_cfg /
// to_gru_cfg) uses to make a genome's hidden_size/num_layers take effect. 0 (the
// default) reproduces today's behavior exactly: every existing caller that never
// passes these arguments is unaffected. latent_dim is deliberately NOT overridable
// here — it stays fixed at cfg.model.latent_dim for every family (user decision,
// 2026-09-22: same compression ratio across SNN/LSTM/GRU/Transformer).
auto make_lstm_cfg(
    const Meeting01Config& cfg, int hidden_size_override = 0, int num_layers_override = 0)
    -> nn::models::lstm::LSTMAutoencoderConfig;
auto make_gru_cfg(
    const Meeting01Config& cfg, int hidden_size_override = 0, int num_layers_override = 0)
    -> nn::models::gru::GRUAutoencoderConfig;
// `d_model_override`/`n_heads_override`/`n_layers_override`/`d_ff_override`: same
// 0-means-"use the profile field" convention as above, for the Transformer
// architecture search (Meeting01TransformerGaGenome::to_transformer_cfg).
auto make_transformer_cfg(const Meeting01Config& cfg,
    int d_model_override = 0,
    int n_heads_override = 0,
    int n_layers_override = 0,
    int d_ff_override = 0) -> nn::models::transformer::TransformerAutoencoderConfig;
// `encoder_widths`, when non-empty, overrides the profile's fixed
// encoder_layer_spec/decoder_layer_spec with a free-form stack rendered from these
// widths (last = latent; decoder mirrors in reverse then projects to output) — the
// bridge the GA architecture search (Meeting01GaGenome::to_ae_config) uses to make a
// genome's widths take effect. Empty (the default) reproduces today's behavior
// exactly: every existing caller that never passes this argument is unaffected.
auto make_snn_cfg(const Meeting01Config& cfg,
    float alpha,
    float v_th,
    const std::vector<int>& encoder_widths = {}) -> nn::models::autoencoder::AutoencoderConfig;

struct TrainResult
{
    RunMetrics metrics;
    EpochHistory history;
};

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult;

auto train_with_early_stopping_gru(nn::models::gru::GRUAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult;

auto train_with_early_stopping_transformer(nn::models::transformer::TransformerAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult;

auto train_with_early_stopping_snn(nn::models::autoencoder::ProtocolSpikingAutoencoder& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<WindowMetadata>& train_meta,
    const std::vector<WindowMetadata>& val_meta,
    const std::vector<int>& val_labels,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult;

} // namespace meeting01
