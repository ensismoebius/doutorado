#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "utility/BandpassNotchFilter.hpp"

namespace nn::utility
{
namespace
{

// Odd tap count from the standard Hamming-window main-lobe-width heuristic: a Hamming window
// needs roughly 3.3 sampling-rate-normalized cycles to resolve a transition of transition_hz.
// Odd count -> a true integer center tap (no removable 0/0 singularity to special-case).
auto tap_count(double sampling_rate, double transition_hz) -> int
{
    auto n = static_cast<int>(std::ceil(3.3 * sampling_rate / transition_hz));
    if (n < 3) n = 3;
    if (n % 2 == 0) n += 1;
    return n;
}

// Windowed-sinc lowpass, Hamming-windowed, normalized so the DC/passband gain is exactly 1
// (sum of taps == 1) -- NOT min-max-to-[0,1], which would erase the negative side-lobes a
// sinc kernel needs to cancel stopband frequencies.
auto sinc_lowpass_kernel(int num_taps, double sampling_rate, double cutoff_hz)
    -> std::vector<double>
{
    const double wc = 2.0 * std::numbers::pi * cutoff_hz / sampling_rate;
    const int center = num_taps / 2;

    std::vector<double> h(static_cast<std::size_t>(num_taps));
    double sum = 0.0;
    for (int n = 0; n < num_taps; ++n)
    {
        double v = 0.0;
        if (n == center)
        {
            v = wc / std::numbers::pi;
        }
        else
        {
            const double x = static_cast<double>(n - center);
            v = std::sin(wc * x) / (std::numbers::pi * x);
        }
        const double hamming =
            0.54 - 0.46 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) /
                                   static_cast<double>(num_taps - 1));
        h[static_cast<std::size_t>(n)] = v * hamming;
        sum += h[static_cast<std::size_t>(n)];
    }
    for (double& v : h) v /= sum;
    return h;
}

// Difference of two unity-gain lowpasses: within [low_hz, high_hz], the high-cutoff lowpass
// is ~1 (in its passband) and the low-cutoff lowpass is ~0 (in its stopband), so the
// difference is ~1 across the intended passband -- the standard bandpass-via-lowpass-
// subtraction construction, valid because each lowpass was correctly unity-normalized first.
auto bandpass_kernel(int num_taps, double sampling_rate, double low_hz, double high_hz)
    -> std::vector<double>
{
    auto hi = sinc_lowpass_kernel(num_taps, sampling_rate, high_hz);
    const auto lo = sinc_lowpass_kernel(num_taps, sampling_rate, low_hz);
    for (std::size_t i = 0; i < hi.size(); ++i) hi[i] -= lo[i];
    return hi;
}

// Identity minus a narrow bandpass around notch_hz: passes everything except that band.
//
// Computes its OWN tap count from notch_width_hz rather than reusing the (generally much
// coarser) bandpass transition_hz: a notch only 2 Hz wide needs its two component lowpasses
// resolved much more sharply than a 0.5-40 Hz bandpass does, or their transition slopes
// overlap and the notch never reaches deep attenuation (caught by
// BandpassNotchFilter.MainsNotchAttenuatesOnlyItsOwnBand -- 0.1 Hz-wide notch measured 0.36x
// attenuation, not the near-total attenuation a real notch needs, before this fix).
auto notch_kernel(
    double sampling_rate, double notch_hz, double notch_width_hz, double transition_hz)
    -> std::vector<double>
{
    const int num_taps = tap_count(sampling_rate, std::min(transition_hz, notch_width_hz / 2.0));
    auto h = bandpass_kernel(
        num_taps, sampling_rate, notch_hz - notch_width_hz / 2.0, notch_hz + notch_width_hz / 2.0);
    for (double& v : h) v = -v;
    h[static_cast<std::size_t>(num_taps / 2)] += 1.0;
    return h;
}

// "Same"-length convolution with the constant group delay of a symmetric FIR kernel
// pre-compensated (output index i uses the kernel centered ON i, not delayed by center
// samples) -- zero-padded at the signal's own boundaries, which is a negligible fraction of
// error given this runs on a full, many-thousand-sample recording, never a short window.
auto convolve_same(const std::vector<float>& x, const std::vector<double>& h) -> std::vector<float>
{
    const auto n = static_cast<int>(x.size());
    const auto m = static_cast<int>(h.size());
    const int center = m / 2;

    std::vector<float> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        double acc = 0.0;
        for (int k = 0; k < m; ++k)
        {
            const int j = i + center - k;
            if (j >= 0 && j < n)
                acc += h[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] = static_cast<float>(acc);
    }
    return y;
}

} // namespace

auto bandpass_notch(const std::vector<float>& signal,
    double sampling_rate,
    double low_hz,
    double high_hz,
    double notch_hz,
    double notch_width_hz,
    double transition_hz) -> std::vector<float>
{
    if (sampling_rate <= 0.0)
        throw std::invalid_argument("bandpass_notch: sampling_rate must be > 0");
    if (transition_hz <= 0.0)
        throw std::invalid_argument("bandpass_notch: transition_hz must be > 0");
    const double nyquist = sampling_rate / 2.0;
    if (!(low_hz > 0.0 && low_hz < high_hz && high_hz < nyquist))
        throw std::invalid_argument(
            "bandpass_notch: require 0 < low_hz < high_hz < sampling_rate/2 (Nyquist)");
    if (notch_hz > 0.0)
    {
        if (notch_width_hz <= 0.0)
            throw std::invalid_argument(
                "bandpass_notch: notch_width_hz must be > 0 when notch_hz > 0");
        if (!(notch_hz - notch_width_hz / 2.0 > 0.0 && notch_hz + notch_width_hz / 2.0 < nyquist))
            throw std::invalid_argument(
                "bandpass_notch: notch band [notch_hz - width/2, notch_hz + width/2] must lie "
                "strictly within (0, sampling_rate/2)");
    }

    const int num_taps = tap_count(sampling_rate, transition_hz);
    auto out = convolve_same(signal, bandpass_kernel(num_taps, sampling_rate, low_hz, high_hz));
    if (notch_hz > 0.0)
        out = convolve_same(
            out, notch_kernel(sampling_rate, notch_hz, notch_width_hz, transition_hz));
    return out;
}

} // namespace nn::utility
