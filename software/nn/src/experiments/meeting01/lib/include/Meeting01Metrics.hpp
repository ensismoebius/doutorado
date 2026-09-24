#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "models/gru/GRUAutoencoderConfig.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "models/transformer/TransformerAutoencoderConfig.hpp"
#include "statistics/binary_classification_metrics.hpp"
#include "statistics/reconstruction_metrics.hpp"
#include "tensor/Tensor.hpp"
#include "utility/ParameterCount.hpp"

namespace meeting01
{

using Tensor = nn::Tensor;

// Moved to core 2026-09-23 (generic, zero meeting01-specific coupling): mse_between/
// mae_between now live in statistics/reconstruction_metrics.hpp, parameter_count in
// utility/ParameterCount.hpp, and the old compute_precision_recall_f1 in
// statistics/binary_classification_metrics.hpp as binary_precision_recall_f1 (renamed,
// same computation -- distinct from statistics::compute_classification_metrics's
// macro-averaged numbers in multi_class_metrics.hpp, see that header's doc comment).
// Re-exported here so existing call sites throughout meeting01 stay unqualified.
using nn::utils::parameter_count;
using statistics::binary_precision_recall_f1;
using statistics::mae_between;
using statistics::mae_between_masked;
using statistics::mse_between;
using statistics::mse_between_masked;

auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t;
auto estimate_snn_macs(std::size_t input_features, int hidden_size, int layers) -> std::size_t;

/// Exact per-layer MAC count for a free-form encoder (decoder mirrored), multiplied by
/// the simulation steps. Prefer this over the (hidden_size, layers) overload for genome
/// costing: that one assumes a uniform hidden width and cannot distinguish {128, 8} from
/// {128, 120}.
auto estimate_snn_macs(
    std::size_t input_features, const std::vector<int>& encoder_widths, int time_steps)
    -> std::size_t;

/// Raw multiply-accumulate estimate for the GRU autoencoder. Mirrors
/// estimate_lstm_macs (one stack, T steps, + projections) with 3 gates instead of 4,
/// so the two are directly comparable.
auto estimate_gru_macs(const nn::models::gru::GRUAutoencoderConfig& cfg) -> std::size_t;

/// Raw multiply-accumulate estimate for the bottlenecked Transformer autoencoder.
/// Includes the O(T^2 * d_model) self-attention term explicitly (scores + A*V),
/// counted once per encoder block in both the encoder and the decoder stack.
auto estimate_transformer_macs(const nn::models::transformer::TransformerAutoencoderConfig& cfg)
    -> std::size_t;

} // namespace meeting01
