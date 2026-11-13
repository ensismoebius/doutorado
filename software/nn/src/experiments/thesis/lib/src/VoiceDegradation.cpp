#include "VoiceDegradation.hpp"

#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

namespace thesis
{

namespace
{

// Levinson-Durbin on r[0..p]: fills a[0..p] (a[0] = 1) of A(z) = sum a_k z^-k.
// The autocorrelation method yields a stable 1/A(z) for a positive-definite r.
void levinson(const std::vector<double>& r, std::vector<double>& a)
{
    const std::size_t p = r.size() - 1;
    a.assign(p + 1, 0.0);
    a[0] = 1.0;
    double err = r[0];
    std::vector<double> prev;
    for (std::size_t i = 1; i <= p; ++i)
    {
        double acc = r[i];
        for (std::size_t j = 1; j < i; ++j) acc += a[j] * r[i - j];
        const double k = -acc / err;
        prev = a;
        for (std::size_t j = 1; j < i; ++j) a[j] = prev[j] + k * prev[i - j];
        a[i] = k;
        err *= (1.0 - k * k);
        if (!(err > 0.0))
            throw std::runtime_error(
                "VoiceDegradation: LPC prediction error became non-positive (order " +
                std::to_string(i) + ")");
    }
}

} // namespace

auto whisper(const std::vector<float>& x, double sample_rate_hz, std::uint32_t seed)
    -> std::vector<float>
{
    if (!(sample_rate_hz > 0.0))
        throw std::invalid_argument("VoiceDegradation: the sample rate must be positive");
    const auto frame = static_cast<std::size_t>(std::lround(0.025 * sample_rate_hz / 2.0)) * 2;
    const std::size_t hop = frame / 2;
    const auto order = static_cast<std::size_t>(2 + std::lround(sample_rate_hz / 1000.0));
    if (frame <= order)
        throw std::invalid_argument(
            "VoiceDegradation: the sample rate is too low for the LPC order");

    // Periodic Hann `win` (shifted copies at 50 % overlap sum to exactly 1) for the LPC
    // analysis, and its square root `syn` for the synthesis. The whisper frames are
    // uncorrelated noise, so their ENERGIES add where they overlap: with syn, whose squares
    // are win and sum to 1, frame f carrying the energy of x under win_f makes the expected
    // power of the whisper exactly that of x. (A Hann synthesis window would give
    // sin^4 + cos^4, 75 % of it on average.)
    std::vector<double> win(frame), syn(frame);
    for (std::size_t n = 0; n < frame; ++n)
    {
        win[n] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) /
                                      static_cast<double>(frame));
        syn[n] = std::sqrt(win[n]);
    }

    const auto len = static_cast<long long>(x.size());
    std::vector<double> out(x.size(), 0.0);
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<double> seg(frame), r(order + 1), a, y(frame);
    // Frames start half a frame before the signal so every sample is covered twice.
    for (long long start = -static_cast<long long>(hop); start < len;
        start += static_cast<long long>(hop))
    {
        double energy = 0.0;
        for (std::size_t n = 0; n < frame; ++n)
        {
            const long long t = start + static_cast<long long>(n);
            const double v = (t >= 0 && t < len) ? x[static_cast<std::size_t>(t)] : 0.0;
            seg[n] = v * win[n];
            energy += v * v * win[n]; // = sum (v * syn)^2, this frame's share of x's energy
        }
        // Draw the excitation whatever the frame holds, so frame k always uses the same
        // stretch of the random stream.
        for (std::size_t n = 0; n < frame; ++n) y[n] = gauss(rng);
        if (energy == 0.0) continue; // silence stays silence

        for (std::size_t k = 0; k <= order; ++k)
        {
            double acc = 0.0;
            for (std::size_t n = k; n < frame; ++n) acc += seg[n] * seg[n - k];
            r[k] = acc;
        }
        r[0] *= 1.0 + 1e-9; // white-noise correction: keeps r positive definite
        levinson(r, a);

        // All-pole synthesis 1/A(z) of the white excitation, then the window.
        double synth_energy = 0.0;
        for (std::size_t n = 0; n < frame; ++n)
        {
            double v = y[n];
            for (std::size_t k = 1; k <= order && k <= n; ++k) v -= a[k] * y[n - k];
            y[n] = v;
        }
        for (std::size_t n = 0; n < frame; ++n)
        {
            y[n] *= syn[n];
            synth_energy += y[n] * y[n];
        }
        if (!(synth_energy > 0.0) || !std::isfinite(synth_energy))
            throw std::runtime_error(
                "VoiceDegradation: the LPC synthesis of a frame is not finite");
        const double gain = std::sqrt(energy / synth_energy);
        for (std::size_t n = 0; n < frame; ++n)
        {
            const long long t = start + static_cast<long long>(n);
            if (t >= 0 && t < len) out[static_cast<std::size_t>(t)] += gain * y[n];
        }
    }
    return {out.begin(), out.end()};
}

auto degrade_voice(
    const std::vector<float>& x, double sample_rate_hz, float level, std::uint32_t seed)
    -> std::vector<float>
{
    if (!(level > 0.0f && level <= 1.0f))
        throw std::invalid_argument(
            "VoiceDegradation: level must be in (0, 1], got " + std::to_string(level));
    const std::vector<float> w = whisper(x, sample_rate_hz, seed);
    const double keep = std::sqrt(1.0 - static_cast<double>(level));
    const double add = std::sqrt(static_cast<double>(level));
    std::vector<float> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        out[i] = static_cast<float>(keep * x[i] + add * w[i]);
    return out;
}

} // namespace thesis
