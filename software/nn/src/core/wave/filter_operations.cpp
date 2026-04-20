/**
 * @file filter_operations.cpp
 * @brief FIR filter construction helpers (low-pass, high-pass, alpha computation).
 */

/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 2 de abr de 2020
 *
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{

void validateOddOrder(int order)
{
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
    }
}

auto buildSincLowPassKernel(int order, double alpha) -> std::vector<double>
{
    std::vector<double> filter(order + 1);
    const auto halfOrderSize = static_cast<double>(order) / 2.0;

    for (int n = 0; n <= order; ++n)
    {
        filter[n] = sin(alpha * (n - halfOrderSize)) / (M_PI * (n - halfOrderSize));
    }

    const auto [min_it, max_it] = std::minmax_element(filter.begin(), filter.end());
    const double minVal = *min_it;
    const double maxVal = *max_it;

    const double range = (maxVal == minVal) ? 1.0 : (maxVal - minVal);
    std::transform(filter.begin(),
        filter.end(),
        filter.begin(),
        [minVal, range](double value) { return (value - minVal) / range; });
    return filter;
}

auto calc_orthogonal_vector_local(const std::vector<double>& vector) -> std::vector<double>
{
    std::vector<double> result(vector.size());
    double multiplier = 1.0;

    for (size_t i = 0; i < vector.size(); ++i)
    {
        result[i] = vector[vector.size() - 1 - i] * multiplier;
        multiplier *= -1.0;
    }

    return result;
}

} // namespace

auto createAlpha(double samplingRate, double filterMaxFrequency, bool highPass = false) -> double
{
    double alpha = M_PI * filterMaxFrequency / (samplingRate / 2);

    if (highPass)
    {
        return M_PI - alpha;
    }

    return alpha;
}

auto createLowPassFilter(int order, double samplingRate, double filterMaxFrequency)
    -> std::vector<double>
{
    validateOddOrder(order);
    const auto alpha = createAlpha(samplingRate, filterMaxFrequency);
    return buildSincLowPassKernel(order, alpha);
}

auto createHighPassFilter(int order, double samplingRate, double filterStartFrequency)
    -> std::vector<double>
{
    validateOddOrder(order);

    // Calculating the alpha for high pass filter
    const double alpha = createAlpha(samplingRate, filterStartFrequency, true);
    const auto filter = buildSincLowPassKernel(order, alpha);

    // Builds the orthogonal vector
    // and return the final result (high pass filter)
    return calc_orthogonal_vector_local(filter);
}

auto createStopBandFilter(
    int order, double samplingRate, double startFrequency, double finalFrequency)
    -> std::vector<double>
{
    validateOddOrder(order);

    auto lowPassMax = createLowPassFilter(order, samplingRate, finalFrequency);
    auto lowPassMin = createLowPassFilter(order, samplingRate, startFrequency);

    for (auto i = 0; i < order + 1; ++i)
    {
        lowPassMax[i] = lowPassMax[i] - lowPassMin[i];
    }

    return lowPassMax;
}

auto bandStopFilter(int order, double samplingRate, double startFrequency, double finalFrequency)
    -> std::vector<double>
{
    validateOddOrder(order);

    auto highPass = createHighPassFilter(order, samplingRate, startFrequency);
    auto lowPass = createLowPassFilter(order, samplingRate, finalFrequency);

    for (auto i = 0; i < order + 1; ++i)
    {
        lowPass[i] = lowPass[i] + highPass[i];
    }

    return lowPass;
}

auto createTriangularWindow(int order) -> std::vector<double>
{
    // order plus 1 is the amount of items
    std::vector<double> w(order + 1);

    // The reference point is amount of items divided by 2
    double referencePoint = order / 2.0;

    auto n = 0;
    for (; n <= referencePoint; ++n)
    {
        w[n] = 2.0 * n / order;
    }

    for (; n <= order; ++n)
    {
        w[n] = 2.0 - (2.0 * n / order);
    }
    return w;
}

void applyWindow(std::vector<double>& filter, const std::vector<double>& window)
{
    if (filter.size() != window.size())
    {
        throw std::runtime_error("Filter and window must have the same size!");
    }

    for (auto i = 0; i < static_cast<int>(filter.size()); ++i)
    {
        filter[i] *= window[i];
    }
}
