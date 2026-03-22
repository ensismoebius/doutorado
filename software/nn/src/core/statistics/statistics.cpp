/**
 * @file statistics.cpp
 * @brief Basic descriptive statistics helpers.
 */

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
#include <numeric>
#include <vector>

namespace statistics
{
/**
 * @brief Computes population variance for a vector of values.
 */
auto variance(const std::vector<double>& data) -> double
{
    // Calculate the mean using std::accumulate
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());

    // Calculate the variance using std::accumulate
    double variance_val =
        std::accumulate(data.begin(),
            data.end(),
            0.0,
            [&](double acc, double val) { return acc + std::pow(val - mean, 2); }) /
        static_cast<double>(data.size());

    return variance_val;
}

/**
 * @brief Computes population variance for a raw contiguous array.
 */
auto variance(const double* data, unsigned int length) -> double
{
    // Calculate the mean using std::accumulate
    double mean = std::accumulate(data, data + length, 0.0) / static_cast<double>(length);

    // Calculate the variance using std::accumulate
    double variance_val =
        std::accumulate(data,
            data + length,
            0.0,
            [&](double acc, double val) { return acc + std::pow(val - mean, 2); }) /
        static_cast<double>(length);

    return variance_val;
}

/**
 * @brief Computes standard deviation for a vector of values.
 */
auto standardDeviation(const std::vector<double>& data) -> double
{
    return std::sqrt(variance(data));
}

/**
 * @brief Computes standard deviation for a raw contiguous array.
 */
auto standardDeviation(const double* data, unsigned int length) -> double
{
    return std::sqrt(variance(data, length));
}
} // namespace statistics
