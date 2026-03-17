/**
 * @file linear_algebra.cpp
 * @brief Implementation of miscellaneous linear algebra helpers.
 */

/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 29 de mar de 2020
 *
 */

#include "nn/linearAlgebra/linear_algebra.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace linearAlgebra
{

auto derivative(std::vector<double>& vector, long level) -> std::vector<double>
{
    if (vector.size() < 2)
    {
        return std::vector<double>({vector.empty() ? 0.0 : vector[0]});
    }

    for (unsigned int i = 0; i < vector.size() - 1; ++i)
    {
        vector.at(i) = vector.at(i + 1) - vector.at(i);
    } // TODO: Refactor loop?

    level--;

    vector.resize(vector.size() - 1);

    if (level > 0)
    {
        vector = derivative(vector, level);
    }
    return vector;
}

auto dotProduct(std::span<const double> a, std::span<const double> b) -> double
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }
    return std::transform_reduce(a.begin(), a.end(), b.begin(), 0.0);
}

auto calcOrthogonalVector(std::span<const double> vector) -> std::vector<double>
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

void normalizeVectorToSum1(std::span<double> signal)
{
    if (signal.empty()) return;
    double sum = std::reduce(signal.begin(), signal.end(), 0.0);

    if (std::abs(sum) > 1e-9) // Check if not zero
        std::ranges::for_each(signal, [sum](double& v) { v /= sum; });
}

void normalizeVectorToSum1AllPositive(std::span<double> signal)
{
    if (signal.empty()) return;
    double min_val = *std::min_element(signal.begin(), signal.end());

    if (min_val < 0)
    {
        double offset = std::abs(min_val) + 1.0;
        std::ranges::for_each(signal, [offset](double& v) { v += offset; });
    }

    normalizeVectorToSum1(signal);
}

void normalizeVectorToRange(std::span<double> signal, double lowerLimit, double upperLimit)
{
    if (lowerLimit >= upperLimit)
    {
        throw std::runtime_error("lowerLimit MUST be bigger than upperLimit");
    }
    if (signal.empty()) return;

    auto [minIt, maxIt] = std::minmax_element(signal.begin(), signal.end());
    double minVal = *minIt;
    double maxVal = *maxIt;

    double rangeVal = (maxVal == minVal) ? 1.0 : (maxVal - minVal);
    double rangeLim = upperLimit - lowerLimit;

    std::ranges::for_each(
        signal, [=](double& v) { v = (((v - minVal) / rangeVal) * rangeLim) + lowerLimit; });
}

auto convolution(std::span<double> data, std::span<const double> kernel) -> bool
{
    if (data.empty() || kernel.empty()) return false;

    long dataLength = static_cast<long>(data.size());
    long kernelSize = static_cast<long>(kernel.size());
    std::vector<double> convolutedSignal(dataLength, 0.0);

    // start convolution from out[kernelSize-1] to out[dataSize-1] (last)
    for (long i = kernelSize - 1; i < dataLength; ++i)
    {
        double sum = 0;
        for (long j = i, k = 0; k < kernelSize; --j, ++k)
        {
            sum += data[j] * kernel[k];
        }
        convolutedSignal[i] = sum;
    }

    // convolution from out[0] to out[kernelSize-2]
    for (long i = 0; i < kernelSize - 1; ++i)
    {
        double sum = 0;
        for (long j = i, k = 0; j >= 0; --j, ++k)
        {
            sum += data[j] * kernel[k];
        }
        convolutedSignal[i] = sum;
    }

    std::copy(convolutedSignal.begin(), convolutedSignal.end(), data.begin());
    return true;
}

void discreteCosineTransform(std::span<double> vector)
{
    if (vector.empty()) return;
    size_t N = vector.size();
    std::vector<double> res(N);

    // Naive implementation matching original logic
    for (size_t k = 0; k < N; ++k)
    {
        double sum = 0;
        for (size_t n = 0; n < N; ++n)
        {
            sum += vector[n] * std::cos(((2.0 * n + 1.0) * M_PI * k) / (2.0 * N));
        }

        double alpha = (k == 0) ? std::sqrt(1.0 / N) : std::sqrt(2.0 / N);
        res[k] = alpha * sum;
    }

    std::copy(res.begin(), res.end(), vector.begin());
}

void scaleMatrix(std::vector<std::vector<double>>& matrix)
{
    if (matrix.empty()) return;
    unsigned int size = matrix.size();

    // Selecting the line on which we have the value to nullify
    for (unsigned int lineIndex = 1; lineIndex < size; lineIndex++)
    {
        // Selecting the column on which we have the value to nullify
        for (unsigned columnIndex = 0; columnIndex < lineIndex; columnIndex++)
        {
            if (matrix[lineIndex][columnIndex] == 0) continue;

            // Find best line for subtraction
            unsigned int bestLine = 0;
            bool found = false;

            for (; bestLine < size; bestLine++)
            {
                if (matrix[bestLine][columnIndex] == 0) continue;

                bool zeroedBefore = true;
                for (unsigned int ci = 0; ci < columnIndex; ci++)
                {
                    if (matrix[bestLine][ci] != 0)
                    {
                        zeroedBefore = false;
                        break;
                    }
                }
                if (zeroedBefore)
                {
                    found = true;
                    break;
                }
            }

            if (!found || matrix[bestLine][columnIndex] == 0)
            {
                // Keep exception message from original
                throw std::runtime_error(
                    "Matrix is singular or ill-conditioned: no suitable pivot found for column " +
                    std::to_string(columnIndex));
            }

            double coef = matrix[lineIndex][columnIndex] / matrix[bestLine][columnIndex];

            // Apply subtraction
            matrix[lineIndex][columnIndex] = 0;
            for (unsigned int ci = columnIndex + 1; ci < matrix[lineIndex].size(); ci++)
            {
                matrix[lineIndex][ci] -= coef * matrix[bestLine][ci];
            }
        }
    }
}

auto solveMatrix(const std::vector<std::vector<double>>& matrix) -> std::vector<double>
{
    if (matrix.empty()) return {};
    unsigned int rows = matrix.size();
    unsigned int cols = matrix[0].size();

    std::vector<double> result(rows);

    // Back substitution
    for (int li = static_cast<int>(rows) - 1; li >= 0; li--)
    {
        double temp = 0;
        for (unsigned int ci = li + 1; ci < cols - 1; ci++)
        {
            temp -= matrix[li][ci] * result[ci];
        }

        result[li] = (matrix[li][cols - 1] + temp) / matrix[li][li];
    }

    return result;
}

void resizeCentered(std::vector<double>& vector, long newSize, double defaultValue)
{
    long currentSize = static_cast<long>(vector.size());
    if (currentSize == newSize) return;

    if (newSize < currentSize)
    {
        long removeCount = currentSize - newSize;
        long left = removeCount / 2;
        long right = removeCount - left;

        vector.erase(vector.begin(), vector.begin() + left);
        vector.erase(vector.end() - right, vector.end());
    }
    else
    {
        long addCount = newSize - currentSize;
        long left = addCount / 2;
        long right = addCount - left;

        vector.insert(vector.begin(), left, defaultValue);
        vector.insert(vector.end(), right, defaultValue);
    }
}

void minMaxNormalizeFeatures(
    std::vector<std::vector<double>>& features, const std::vector<double>& range)
{
    if (features.empty() || range.size() != 2) return;

    size_t n_features = features[0].size();
    std::vector<double> min_vals(n_features, std::numeric_limits<double>::max());
    std::vector<double> max_vals(n_features, std::numeric_limits<double>::lowest());

    // Find min/max
    for (const auto& sample : features)
    {
        for (size_t i = 0; i < n_features; ++i)
        {
            min_vals[i] = std::min(min_vals[i], sample[i]);
            max_vals[i] = std::max(max_vals[i], sample[i]);
        }
    }

    double rangeDiff = range[1] - range[0];
    double rangeMin = range[0];

    // Normalize
    for (auto& sample : features)
    {
        for (size_t i = 0; i < n_features; ++i)
        {
            double minV = min_vals[i];
            double maxV = max_vals[i];

            if (maxV != minV)
            {
                double normalized = (sample[i] - minV) / (maxV - minV);
                sample[i] = normalized * rangeDiff + rangeMin;
            }
            else
            {
                sample[i] = rangeMin;
            }
        }
    }
}

} // namespace linearAlgebra
