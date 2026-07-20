#pragma once

#include <cstdint>
#include <string>

#include "tensor/Tensor.hpp"

namespace e04
{

using Tensor = nn::Tensor;

auto encode_sample(const Tensor& sample, const std::string& encoding, std::uint32_t seed) -> Tensor;

auto flatten_time_series(const Tensor& sample) -> Tensor;
auto unflatten_time_series(const Tensor& flat, nn::Index rows, nn::Index cols) -> Tensor;

auto apply_snn_architecture_transform(
    const Tensor& encoded, const std::string& architecture, float alpha, float v_th) -> Tensor;

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

} // namespace e04
