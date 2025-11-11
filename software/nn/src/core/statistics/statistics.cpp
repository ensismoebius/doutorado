/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 3 de jul de 2020
 *
 */
#include <cmath>
#include <utility>
#include <vector>

namespace statistics
{
auto variance(const std::vector<double>& data) -> double
{
    double mean = 0;
    double variance = 0;

    // Calculate the mean
    for (auto accuracy : data)
    {
        mean += accuracy;
    }
    mean /= static_cast<double>(data.size());

    // Calculate the variance
    for (auto accuracy : data)
    {
        variance += std::pow(accuracy - mean, 2);
    }
    variance /= data.size();

    return variance;
}

auto variance(const double* data, unsigned int length) -> double
{
    double mean = 0;
    double variance = 0;

    // Calculate the mean
    for (unsigned int i = 0; i < length; i++)
    {
        mean += data[i];
    }
    mean /= length;

    // Calculate the variance
    for (unsigned int i = 0; i < length; i++)
    {
        variance += std::pow(data[i] - mean, 2);
    }
    variance /= length;

    return variance;
}

auto standardDeviation(const std::vector<double>& data) -> double
{
    return std::sqrt(variance(data));
}

auto standardDeviation(const double* data, unsigned int length) -> double
{
    return std::sqrt(variance(data, length));
}
} // namespace statistics
