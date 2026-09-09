#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Meeting01DatasetSplit.hpp"
#include "Meeting01PerWindow.hpp"
#include "Meeting01RunMetrics.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "tensor/Tensor.hpp"

namespace meeting01
{

using Tensor = nn::Tensor;

auto evaluate_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float max_reconstruct_mean_deviation,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    std::uint32_t seed,
    float infer_ms,
    int lstm_frame_size) -> RunMetrics;

auto evaluate_snn(nn::models::autoencoder::ProtocolSpikingAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float max_reconstruct_mean_deviation,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    float infer_ms) -> RunMetrics;

// Per-window reconstruction error for the SNN-AE (arch transform + flatten + forward +
// unflatten, mse/mae in encoded space). `proto` carries the shared identity fields;
// `meta` is parallel to `samples`.
auto per_window_errors_snn(nn::models::autoencoder::ProtocolSpikingAutoencoder& model,
    const std::vector<Tensor>& samples,
    const std::vector<WindowMetadata>& meta,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    PerWindowError proto) -> std::vector<PerWindowError>;

} // namespace meeting01
