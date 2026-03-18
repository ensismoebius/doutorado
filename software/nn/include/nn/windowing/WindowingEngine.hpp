/**
 * @file WindowingEngine.hpp
 * @brief Header-only utility to enumerate sliding windows over a fixed-length signal.
 *
 * Usage:
 * @code
 *   WindowSpec spec{ .window_size = 256, .overlap = 0.5f, .sample_rate = 1024 };
 *   auto windows = nn::windowing::compute_windows(4096, spec);
 *   for (const auto& w : windows) {
 *       // access signal[w.start .. w.end)
 *   }
 * @endcode
 *
 * Complexity: O(N_windows) time and space.
 *   N_windows = 1 + (signal_length - W) / H
 *   where H = W * (1 - overlap).
 *
 * Notes:
 *  - Only *complete* windows are produced (no padding of the final partial window).
 *  - Window endpoint `end == start + window_size` is exclusive (STL convention).
 *  - The function is stateless and thread-safe.
 */

#pragma once

#include <vector>

#include "nn/windowing/WindowSpec.hpp"

namespace nn::windowing
{

/**
 * @brief Compute sliding-window descriptors for a signal of `signal_length` samples.
 *
 * @param signal_length  Total number of samples in the signal (> 0).
 * @param spec           Windowing configuration (window_size, overlap, sample_rate).
 * @return Ordered vector of `WindowInfo` structs, one per complete window.
 *         Returns an empty vector if `signal_length < spec.window_size`.
 */
[[nodiscard]] inline auto compute_windows(int signal_length, const WindowSpec& spec)
    -> std::vector<WindowInfo>
{
    if (signal_length <= 0 || spec.window_size <= 0 || signal_length < spec.window_size)
    {
        return {};
    }

    const int hop = spec.hop_size();
    const int n_windows = spec.num_windows(signal_length);

    std::vector<WindowInfo> windows;
    windows.reserve(static_cast<std::size_t>(n_windows));

    const double inv_sr = 1.0 / static_cast<double>(spec.sample_rate);
    const double half_w = static_cast<double>(spec.window_size) * 0.5;

    for (int k = 0; k < n_windows; ++k)
    {
        const int start = k * hop;
        const int end = start + spec.window_size;
        const double center_time_s = (static_cast<double>(start) + half_w) * inv_sr;
        windows.push_back({.start = start, .end = end, .center_time_s = center_time_s});
    }

    return windows;
}

} // namespace nn::windowing
