#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "models/gru/GRUAutoencoderConfig.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "models/transformer/TransformerAutoencoderConfig.hpp"
#include "tensor/Tensor.hpp"

namespace meeting01
{

using Tensor = nn::Tensor;

auto mse_between(const Tensor& a, const Tensor& b) -> float;
auto mae_between(const Tensor& a, const Tensor& b) -> float;

void compute_precision_recall_f1(const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    float& precision,
    float& recall,
    float& f1);

auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t;
auto estimate_snn_macs(std::size_t input_features, int hidden_size, int layers) -> std::size_t;

/// Raw multiply-accumulate estimate for the GRU autoencoder. Mirrors
/// estimate_lstm_macs (one stack, T steps, + projections) with 3 gates instead of 4,
/// so the two are directly comparable.
auto estimate_gru_macs(const nn::models::gru::GRUAutoencoderConfig& cfg) -> std::size_t;

/// Raw multiply-accumulate estimate for the bottlenecked Transformer autoencoder.
/// Includes the O(T^2 * d_model) self-attention term explicitly (scores + A*V),
/// counted once per encoder block in both the encoder and the decoder stack.
auto estimate_transformer_macs(const nn::models::transformer::TransformerAutoencoderConfig& cfg)
    -> std::size_t;

template <typename T>
auto parameter_count(std::span<T*> params) -> std::size_t
{
    std::size_t count = 0;
    for (T* p : params)
    {
        if (!p) continue;
        count += static_cast<std::size_t>(p->size());
    }
    return count;
}

} // namespace meeting01
