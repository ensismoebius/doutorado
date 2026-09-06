/**
 * @file WindowInfo.hpp
 * @brief WindowInfo struct (extracted from WindowSpec.hpp).
 */

#pragma once

namespace nn::windowing
{

/**
 * @brief Describes the position and timestamp of a single window.
 *
 * `start` and `end` are sample indices into the original signal:
 *   [start, end)  (end is exclusive, end == start + window_size).
 * `center_time_s` is the wall-clock time of the window centre in seconds,
 *   computed as  (start + window_size / 2) / sample_rate.
 */
struct WindowInfo
{
    int start;            ///< Inclusive start sample index.
    int end;              ///< Exclusive end sample index (start + window_size).
    double center_time_s; ///< Centre time in seconds w.r.t. signal start.
};

} // namespace nn::windowing
