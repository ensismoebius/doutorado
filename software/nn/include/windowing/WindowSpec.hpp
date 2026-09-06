/**
 * @file WindowSpec.hpp
 * @brief Core structs for signal windowing configuration.
 *
 * Lives in `include/windowing/`, independent of any dataset or backend.
 * Consumer code (datasets, preprocessors) depends on this; nothing here
 * depends on datasets or tensor types.
 */

#pragma once

#include <algorithm>
#include <stdexcept>

#include "windowing/WindowInfo.hpp"

namespace nn::windowing
{

/**
 * @brief Windowing configuration for one modality.
 *
 * Parameters:
 *  - window_size  : number of samples per window.
 *  - overlap      : fraction of window_size that consecutive windows share,
 *                   in [0, 1).  Default = 0.5 (50 % overlap).
 *  - sample_rate  : samples per second; used only for computing
 *                   `WindowInfo::center_time_s`.  Does not affect indexing.
 *
 * Derived quantity:
 *   hop_size = window_size * (1 - overlap)   [rounded towards 1]
 *
 * Example with window_size = 256, overlap = 0.5, sample_rate = 1024 Hz:
 *   hop_size = 128 samples  ->  every ~125 ms there is a new window.
 */
struct WindowSpec
{
    int window_size{};   ///< Samples per window (> 0).
    float overlap{0.5f}; ///< Fractional overlap in [0, 1).
    int sample_rate{1};  ///< Hz (used for timestamp computation only).

    /// Derived hop size: minimum 1 to prevent infinite loops.
    [[nodiscard]] constexpr int hop_size() const noexcept
    {
        return std::max(1, static_cast<int>(static_cast<float>(window_size) * (1.0f - overlap)));
    }

    /// Number of complete windows that fit in a signal of `signal_length` samples.
    [[nodiscard]] constexpr int num_windows(int signal_length) const noexcept
    {
        if (signal_length < window_size || window_size <= 0)
        {
            return 0;
        }
        return 1 + (signal_length - window_size) / hop_size();
    }

    /// Validate that the spec is well-formed.  Throws `std::invalid_argument` on error.
    void validate() const
    {
        if (window_size <= 0)
        {
            throw std::invalid_argument("WindowSpec: window_size must be > 0");
        }
        if (overlap < 0.0f || overlap >= 1.0f)
        {
            throw std::invalid_argument("WindowSpec: overlap must be in [0, 1)");
        }
        if (sample_rate <= 0)
        {
            throw std::invalid_argument("WindowSpec: sample_rate must be > 0");
        }
    }
};

} // namespace nn::windowing
