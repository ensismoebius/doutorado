#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "RunMetrics.hpp"
#include "autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "nn/models/lstm/LSTMAutoencoder.hpp"
#include "nn/tensor/Tensor.hpp"

namespace comparative_autoencoder_experiment
{

using Tensor = nn::Tensor;

auto evaluate_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float anomaly_tau,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    std::uint32_t seed,
    float infer_ms) -> RunMetrics;

auto evaluate_snn(ProtocolSpikingAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float anomaly_tau,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    float infer_ms) -> RunMetrics;

} // namespace comparative_autoencoder_experiment
