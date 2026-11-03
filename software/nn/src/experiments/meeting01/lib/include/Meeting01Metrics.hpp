#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"
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

/// Exact per-layer MAC count for a free-form encoder (decoder mirrored), the encoder run
/// `encoder_steps` times and the decoder `decoder_steps` times (1 when the latent is averaged
/// over time before it, model.snn_time_pool = "latent"). An analytic, spike-blind estimate:
/// it is the SNN GA's cost objective, which has to be known before a genome is trained.
/// The measured count of a trained network is count_snn_ops.
auto estimate_snn_macs(std::size_t input_features,
    const std::vector<int>& encoder_widths,
    int encoder_steps,
    int decoder_steps) -> std::size_t;
/// The same with the decoder run as many times as the encoder.
auto estimate_snn_macs(
    std::size_t input_features, const std::vector<int>& encoder_widths, int time_steps)
    -> std::size_t;

/// Energy per operation, 45 nm, 32-bit float (Horowitz, "Computing's energy problem (and
/// what we can do about it)", ISSCC 2014): a multiply 3.7 pJ plus an add 0.9 pJ is one MAC,
/// an add alone is one AC. These are the constants the SNN literature uses for this
/// estimate (e.g. Zhou et al., Spikformer, ICLR 2023). Arithmetic only: memory access,
/// which dominates the energy of real hardware, is not in them.
inline constexpr double kMacEnergyPj = 4.6;
inline constexpr double kAcEnergyPj = 0.9;

/// kMacEnergyPj * macs + kAcEnergyPj * acs, in nJ.
auto arith_energy_nj(double macs, double acs) -> float;

/// The arithmetic of one SNN-AE forward pass, counted from what it really computed.
///
///   Linear, spike input     each spike feeds every output: nnz(input) x out ACs
///   Linear, analog input    rows x in x out MACs (no credit for zeros, as for the ANNs)
///   LifBPTT                 one MAC per (neuron, step): v = beta * v + I
///   TimeMeanPool            one AC per input value (the sum) + one MAC per output (the 1/T)
///   activations, biases     not counted (the convention of every estimate_*_macs)
///
/// Whether a Linear's input is spikes is decided by where it comes from, not by its values: the
/// output of a spiking (non-readout) LifBPTT is, and the model input is when `input_is_spikes`
/// says so (snn_input_is_spikes). Deciding by value would price an analog tensor that happens
/// to be all zeros -- the latent of an encoder that fired nothing, with zero biases -- as zero
/// ACs instead of its MACs. A tensor taken for spikes that holds anything but 0/1 throws
/// std::logic_error. Spikes and (neuron, step) pairs are summed over the spiking LifBPTT
/// layers: spike_rate's numerator and denominator. Reads Sequential::outputs, which every
/// forward() fills, so call it right after model.forward(input). Also throws std::logic_error
/// for the dual-branch model (not built by meeting01) and when the cached outputs do not match
/// the layers (no forward yet).
struct SnnOpCounts
{
    double macs = 0.0;
    double acs = 0.0;
    double spikes = 0.0;
    double neuron_steps = 0.0;
};
auto count_snn_ops(const nn::models::autoencoder::ProtocolSpikingAutoencoder& model,
    const Tensor& input,
    bool input_is_spikes) -> SnnOpCounts;

/// Whether the tensor the SNN-AE receives is a spike train: `recurrent` emits the spikes of a
/// LIF, `dense` passes the encoding through (poisson/latency spikes, direct analog), `conv1d`
/// smooths it into analog values. Throws on an unknown token.
auto snn_input_is_spikes(const std::string& encoding, const std::string& architecture) -> bool;

/// MACs of apply_snn_architecture_transform over `values` input values, which the SNN
/// pipeline runs before the model: "dense" 0, "conv1d" 3 (a 3-tap filter), "recurrent" 1
/// (a LIF update; its reset, at most one subtraction per spike, is not counted). Throws on
/// any other token.
auto count_architecture_transform_macs(const std::string& architecture, std::size_t values)
    -> double;

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
