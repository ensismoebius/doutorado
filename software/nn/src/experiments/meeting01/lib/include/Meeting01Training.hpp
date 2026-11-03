#pragma once

#include <cstdint>
#include <string>
#include <utility>
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
//
// The rendered stack follows model.snn_decoder_neuron (the decoder's hidden neuron, "lif" or
// "analog") and model.snn_time_pool (a `time_mean` stage after the latent for "latent",
// after the output for "output"). The fixed-spec path (no widths) refuses both non-default
// values: it would have to rewrite free-form profile specs, which nothing in meeting01 uses.
auto make_snn_cfg(const Meeting01Config& cfg,
    float alpha,
    float v_th,
    const std::vector<int>& encoder_widths = {}) -> nn::models::autoencoder::AutoencoderConfig;

// The matched ANN control's builder config (family "ann-ae"): the SNN winner's widths with a
// leaky ReLU wherever the default SNN stack has a LIF, and no time stage (it reads the window
// once). Independent of snn_time_pool / snn_decoder_neuron, which shape only the SNN.
auto make_ann_twin_cfg(const Meeting01Config& cfg, const std::vector<int>& encoder_widths)
    -> nn::models::autoencoder::AutoencoderConfig;

// Rows the SNN-AE emits per window: 1 when its time axis is averaged (model.snn_time_pool
// "output" or "latent"), else model.time_steps (one reconstruction per step). The target and
// mask it is trained and scored against have this many rows.
auto snn_output_steps(const Meeting01Config& cfg) -> int;

// The SNN's T range [min, max]: evaluation.ga.snn.time_steps_min/max, or
// [model.time_steps, model.time_steps] when both are 0 (T not searched).
auto snn_time_steps_range(const Meeting01Config& cfg) -> std::pair<int, int>;

// `cfg` with model.time_steps set to one genome's T. Every SNN path (build, encode,
// train, evaluate, cost) reads T from model.time_steps, so evaluating a genome on this
// copy is what makes T a gene; the profile's own value is only the fixed-T default.
auto with_snn_time_steps(const Meeting01Config& cfg, int time_steps) -> Meeting01Config;

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
