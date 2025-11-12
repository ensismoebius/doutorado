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

#include "../linearAlgebra/linearAlgebra.h"

auto createAlpha(double samplingRate, double filterMaxFrequency, bool highPass = false) -> double
{
    double alpha = M_PI * filterMaxFrequency / (samplingRate / 2);

    if (highPass)
    {
        return M_PI - alpha;
    }

    return alpha;
}

auto createLowPassFilter(int order, double samplingRate, double filterMaxFrequency) -> double*
{
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
        return 0;
    }

    auto* filter = new double[order + 1];

    // Calculating the alpha
    auto alpha = createAlpha(samplingRate, filterMaxFrequency);

    auto halfOrderSize = (double) (order / 2.0);

    for (int n = 0; n <= order; ++n)
    {
        filter[n] = sin(alpha * (n - halfOrderSize)) / (M_PI * (n - halfOrderSize));
    }

    linearAlgebra::normalizeVectorToRange(filter, order + 1, 0, 1);

    // The caller is responsible for deleting this memory
    return filter;
}

auto createHighPassFilter(int order, double samplingRate, double filterStartFrequency) -> double*
{
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
        return 0;
    }

    // Filter holder
    auto* filter = new double[order + 1];

    // Calculating the alpha for high pass filter
    double alpha = createAlpha(samplingRate, filterStartFrequency, true);

    auto halfOrderSize = (double) (order / 2.0);

    // Calculate low pass filter
    for (int n = 0; n <= order; ++n)
    {
        filter[n] = sin(alpha * (n - halfOrderSize)) / (M_PI * (n - halfOrderSize));
    }

    // normalizing data
    linearAlgebra::normalizeVectorToRange(filter, order + 1, 0, 1);

    // Builds the orthogonal vector
    // and return the final result (high pass filter)
    return linearAlgebra::calcOrthogonalVector(filter, order + 1);
}

auto createStopBandFi1lter(int order, double samplingRate, double startFrequency, double finalFrequency) -> double*
{
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
        return 0;
    }

    double* lowPassMax = createLowPassFilter(order, samplingRate, finalFrequency);
    double* lowPassMin = createLowPassFilter(order, samplingRate, startFrequency);

    for (int i = 0; i < order + 1; i++)
    {
        lowPassMax[i] = lowPassMax[i] - lowPassMin[i];
    }

    delete[] lowPassMin;

    return lowPassMax;
}

auto bandStopFilter(int order, double samplingRate, double startFrequency, double finalFrequency) -> double*
{
    // Order MUST be odd
    if (order % 2 == 0)
    {
        throw std::runtime_error("Order MUST be an odd number!");
        return 0;
    }

    double* highPass = createHighPassFilter(order, samplingRate, startFrequency);
    double* lowPass = createLowPassFilter(order, samplingRate, finalFrequency);

    for (int i = 0; i < order + 1; i++)
    {
        lowPass[i] = lowPass[i] + highPass[i];
    }

    return lowPass;
}

auto createTriangularWindow(int order) -> double*
{
    // order plus 1 is the amount of items
    auto* w = new double[order + 1];

    // The reference point is amount of items divided by 2
    double referencePoint = order / 2.0;

    int n = 0;
    for (; n <= referencePoint; n++)
    {
        w[n] = 2.0 * n / order;
    }

    for (; n <= order; n++)
    {
        w[n] = 2.0 - 2.0 * n / order;
    }
    return w;
}

void applyWindow(double* filter, double* window, int order)
{
    do
    {
        filter[order] *= window[order];
    } while ((order--) != 0);
}
