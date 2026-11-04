#include "utility/Resample.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn::utility
{

auto resample_bandlimited(
    const std::vector<float>& signal, double from_hz, double to_hz, int half_width_zero_crossings)
    -> std::vector<float>
{
    if (from_hz <= 0.0 || to_hz <= 0.0)
        throw std::invalid_argument("resample_bandlimited: rates must be positive (got " +
                                    std::to_string(from_hz) + " -> " + std::to_string(to_hz) + ")");
    if (signal.empty()) throw std::invalid_argument("resample_bandlimited: empty signal");
    if (half_width_zero_crossings < 1)
        throw std::invalid_argument("resample_bandlimited: half_width_zero_crossings must be >= 1");
    if (from_hz == to_hz) return signal;

    const double step = from_hz / to_hz; // input samples per output sample
    const double cutoff = std::min(1.0, to_hz / from_hz);
    const double half_width = static_cast<double>(half_width_zero_crossings) / cutoff;
    const auto last_input = static_cast<double>(signal.size() - 1U);
    const auto n_out = static_cast<std::size_t>(std::floor(last_input / step)) + 1U;

    std::vector<float> out(n_out);
    for (std::size_t n = 0; n < n_out; ++n)
    {
        const double t = static_cast<double>(n) * step;
        const auto first = static_cast<std::size_t>(std::max(0.0, std::ceil(t - half_width)));
        const auto last =
            static_cast<std::size_t>(std::min(last_input, std::floor(t + half_width)));
        double sum = 0.0;
        double weight_sum = 0.0;
        for (std::size_t k = first; k <= last; ++k)
        {
            const double d = t - static_cast<double>(k);
            const double x = std::numbers::pi * cutoff * d;
            const double sinc = (d == 0.0) ? 1.0 : std::sin(x) / x;
            const double hann = 0.5 * (1.0 + std::cos(std::numbers::pi * d / half_width));
            const double w = cutoff * sinc * hann;
            sum += w * static_cast<double>(signal[k]);
            weight_sum += w;
        }
        out[n] = static_cast<float>(sum / weight_sum);
    }
    return out;
}

} // namespace nn::utility
