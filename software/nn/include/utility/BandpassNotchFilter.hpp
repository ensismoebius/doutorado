#ifndef BANDPASS_NOTCH_FILTER_HPP
#define BANDPASS_NOTCH_FILTER_HPP

#include <vector>

namespace nn::utility
{

// Bandpass [low_hz, high_hz] + optional mains-notch (notch_hz, width notch_width_hz) FIR
// filter for a FULL, un-windowed recording -- this MUST run before windowing, not per-window:
// a sharp low-frequency cutoff (e.g. 0.5 Hz) needs hundreds of FIR taps to reach a usable
// transition width, which would dwarf or exceed a short (e.g. 256-sample) window and produce
// edge-dominated garbage if applied to each already-sliced window separately instead of once
// on the full signal.
//
// Windowed-sinc design (Hamming window, each lowpass stage normalized to unity passband
// gain by construction -- NOT the min-max-to-[0,1] normalization in wave/filter_operations.hpp,
// which corrupts a sinc kernel's negative side-lobes and does not produce a working filter;
// see .wiki/Core/DataLoaders.md for that finding). transition_hz sets the tap count via the
// standard Hamming-window heuristic (taps ~= 3.3 * sampling_rate / transition_hz, rounded up
// to odd) -- smaller transition_hz means a sharper cutoff, more taps, and a slower (one-time,
// load-time-only) filtering pass. notch_hz <= 0 disables the notch (bandpass only).
//
// Throws std::invalid_argument if sampling_rate/low_hz/high_hz/notch parameters don't form a
// valid, ascending, positive, sub-Nyquist configuration.
auto bandpass_notch(const std::vector<float>& signal,
    double sampling_rate,
    double low_hz,
    double high_hz,
    double notch_hz,
    double notch_width_hz = 2.0,
    double transition_hz = 2.0) -> std::vector<float>;

} // namespace nn::utility

#endif // BANDPASS_NOTCH_FILTER_HPP
