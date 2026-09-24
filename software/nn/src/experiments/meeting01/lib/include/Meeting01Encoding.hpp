#pragma once

#include <cstdint>
#include <string>

#include "tensor/Tensor.hpp"

namespace meeting01
{

using Tensor = nn::Tensor;

/// Encodes one window into a time-major spike/analog tensor of shape
/// (`time_steps`, window_size): row = simulation step, column = one window sample.
/// Each window sample is an input neuron; the rows are the axis a spike code uses.
/// @throws std::invalid_argument if `time_steps < 2` — a single step leaves rate and
///         latency coding nothing to encode on, and silently produces a plausible but
///         meaningless result rather than failing.
auto encode_sample(
    const Tensor& sample, const std::string& encoding, std::uint32_t seed, int time_steps)
    -> Tensor;

/// Repeats one window across `time_steps` rows to form the reconstruction target that
/// matches `encode_sample`'s output shape. The target is always the ORIGINAL analog
/// window, never the encoded one, so MSE stays comparable across encodings.
auto make_reconstruction_target(const Tensor& sample, int time_steps) -> Tensor;

/// Builds a (window_size, 1) activity mask: 1.0 for the first `valid_length` rows
/// (real signal), 0.0 for the rest (zero-padded tail — see WindowMetadata::valid_length).
/// The result is a plain (window_size, 1) tensor with no notion of encoding or time
/// steps, so it composes with make_reconstruction_target/to_lstm_frames exactly like the
/// window itself does — pass it through the SAME calls used to shape the target to get a
/// mask of matching shape, then feed both to MSELossImpl::set_mask.
/// @throws std::invalid_argument if valid_length is outside [0, window_size].
auto make_activity_mask(int valid_length, int window_size) -> Tensor;

/// Collapses a (T, F) model output into the single (1, F) reconstruction of the window
/// by averaging over the simulation steps — the standard temporal-averaging readout.
auto reduce_time_major_output(const Tensor& output, int time_steps) -> Tensor;

auto flatten_time_series(const Tensor& sample) -> Tensor;
auto unflatten_time_series(const Tensor& flat, nn::Index rows, nn::Index cols) -> Tensor;

auto apply_snn_architecture_transform(
    const Tensor& encoded, const std::string& architecture, float alpha, float v_th) -> Tensor;

/// Spike train + per-step pre-reset membrane value for the `recurrent` transform.
/// `spikes` and `v_mem` are both (time_steps, features) with the same layout the
/// transform's output uses; here the window's samples ARE the time steps.
struct RecurrentLifTrace
{
    Tensor spikes;
    Tensor v_mem;
};

/// Same computation as `apply_snn_architecture_transform(encoded, "recurrent", ...)`
/// but also returns the membrane trajectory the transform otherwise discards.
auto recurrent_lif_trace(const Tensor& encoded, float alpha, float v_th) -> RecurrentLifTrace;

/**
 * @brief Reshape a (window_size, 1) sample into (window_size/frame_size, frame_size) frames.
 *
 * The LSTM previously consumed the window one scalar per timestep (D=1,
 * T=window_size). That makes the recurrent term — the dominant cost, h·Uᵀ with
 * U of shape (4H, H) — get paid `window_size` times. Grouping `frame_size`
 * consecutive samples per timestep cuts both the sequential depth and the total
 * MAC count by `frame_size`, with no information discarded.
 *
 * Frame t holds the consecutive samples [t*frame_size, (t+1)*frame_size).
 * Storage is column-major, so a direct reshape to (T, frame_size) would instead
 * interleave (frame t would get samples t, t+T, t+2T, …). Reshaping to
 * (frame_size, T) and transposing produces the intended consecutive framing.
 *
 * @throws std::invalid_argument if frame_size <= 0 or does not divide the sample length.
 */
auto to_lstm_frames(const Tensor& sample, int frame_size) -> Tensor;

} // namespace meeting01
