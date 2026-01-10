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

#include <cmath>
#include <stdexcept>
#include <vector>

#include "nn/linearAlgebra/linear_algebra.hpp"

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
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
    }

    std::vector<double> filter(order + 1);

    // Calculating the alpha
    auto alpha = createAlpha(samplingRate, filterMaxFrequency);

    auto halfOrderSize = (double) (order / 2.0);

    for (int n = 0; n <= order; ++n)
    {
        filter[n] = sin(alpha * (n - halfOrderSize)) / (M_PI * (n - halfOrderSize));
    }

    linearAlgebra::normalizeVectorToRange(filter, 0, 1);

    return filter;
}

auto createHighPassFilter(int order, double samplingRate, double filterStartFrequency)
    -> std::vector<double>
{
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
    }

    // Filter holder
    std::vector<double> filter(order + 1);

    // Calculating the alpha for high pass filter
    double alpha = createAlpha(samplingRate, filterStartFrequency, true);

    auto halfOrderSize = (double) (order / 2.0);

    // Calculate low pass filter
    for (int n = 0; n <= order; ++n)
    {
        filter[n] = sin(alpha * (n - halfOrderSize)) / (M_PI * (n - halfOrderSize));
    }

    // normalizing data
    linearAlgebra::normalizeVectorToRange(filter, 0, 1);

    // Builds the orthogonal vector
    // and return the final result (high pass filter)
    return linearAlgebra::calcOrthogonalVector(filter);
}

auto createStopBandFilter(int order, double samplingRate, double startFrequency,
                          double finalFrequency) -> std::vector<double>
{
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
    }

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
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
    }

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
