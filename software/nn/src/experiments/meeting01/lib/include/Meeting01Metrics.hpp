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

/// Raw multiply-accumulate estimate for the LSTM autoencoder: the encoder AND decoder stacks
/// (L layers each, 4 gates) over the S = seq_len frames, the output head on every frame, and
/// the two latent projections once. Layers after the first read the H-wide state, not the
/// D-wide frame. Throws std::invalid_argument for a configuration that is not a buildable
/// network (any of seq_len / input_size / hidden_size / latent_size / num_layers < 1).
auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t;
auto estimate_snn_macs(std::size_t input_features, int hidden_size, int layers) -> std::size_t;

/// Exact per-layer MAC count for a free-form encoder (decoder mirrored), multiplied by
/// the simulation steps. Prefer this over the (hidden_size, layers) overload for genome
/// costing: that one assumes a uniform hidden width and cannot distinguish {128, 8} from
/// {128, 120}.
auto estimate_snn_macs(
    std::size_t input_features, const std::vector<int>& encoder_widths, int time_steps)
    -> std::size_t;

/// Raw multiply-accumulate estimate for the GRU autoencoder. Same counting as
/// estimate_lstm_macs (both stacks, the head per frame, the latent projections once) with
/// 3 gates instead of 4, so the two are directly comparable. Throws like it.
auto estimate_gru_macs(const nn::models::gru::GRUAutoencoderConfig& cfg) -> std::size_t;

/// Raw multiply-accumulate estimate for the bottlenecked Transformer autoencoder.
/// Includes the O(S^2 * d_model) self-attention term explicitly (scores + A*V), S = seq_len,
/// counted once per block in both the encoder and the decoder stack.
auto estimate_transformer_macs(const nn::models::transformer::TransformerAutoencoderConfig& cfg)
    -> std::size_t;

} // namespace meeting01
