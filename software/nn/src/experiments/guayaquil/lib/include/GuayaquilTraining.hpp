#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "GuayaquilConfig.hpp"
#include "GuayaquilEpochHistory.hpp"
#include "GuayaquilRunMetrics.hpp"
#include "autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"

namespace guayaquil
{

using Tensor = nn::Tensor;

auto make_lstm_cfg(const GuayaquilConfig& cfg) -> nn::models::lstm::LSTMAutoencoderConfig;
auto make_snn_cfg(const GuayaquilConfig& cfg, float alpha, float v_th) -> AutoencoderConfig;

struct TrainResult
{
    RunMetrics metrics;
    EpochHistory history;
};

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const GuayaquilConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    float& train_ms,
    float& infer_ms) -> TrainResult;

auto train_with_early_stopping_snn(ProtocolSpikingAutoencoder& model,
    const GuayaquilConfig& cfg,
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

} // namespace guayaquil
