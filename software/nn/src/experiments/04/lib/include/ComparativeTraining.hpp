#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "ComparativeConfig.hpp"
#include "EpochHistory.hpp"
#include "RunMetrics.hpp"
#include "autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "nn/models/lstm/LSTMAutoencoder.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"

namespace comparative_autoencoder_experiment
{

using Tensor = nn::Tensor;

auto make_lstm_cfg(const ComparativeConfig& cfg) -> nn::models::lstm::LSTMAutoencoderConfig;
auto make_snn_cfg(const ComparativeConfig& cfg, float alpha, float v_th) -> AutoencoderConfig;

struct TrainResult
{
    RunMetrics metrics;
    EpochHistory history;
};

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult;

auto train_with_early_stopping_snn(ProtocolSpikingAutoencoder& model,
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
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

} // namespace comparative_autoencoder_experiment
