/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 29 de mar de 2020
 *
 */

#include "linearAlgebra.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace linearAlgebra
{

auto derivative(std::vector<double>& vector, long level) -> std::vector<double>
{
    if (vector.size() < 2)
    {
        return std::vector<double>({vector.at(0)});
    }

    for (unsigned int i = 0; i < vector.size() - 1; ++i)
    {
        vector.at(i) = vector.at(i + 1) - vector.at(i);
    }

    level--;

    vector.resize(vector.size() - 1);

    if (level > 0)
    {
        vector = derivative(vector, level);
    }
    return vector;
}

auto dotProduct(const std::vector<double>& a, const std::vector<double>& b) -> double
{
    double product = 0;

    // Loop for calculate cot product
    for (unsigned int i = 0; i < a.size(); i++)
    {
        product = product + a.at(i) * b.at(i);
    }
    return product;
}

auto calcOrthogonalVector(const double* originalVector, long vectorSize) -> double*
{
    auto* finalResult = new double[vectorSize];

    long middleSignalIndex = vectorSize / 2;
    double tempVar;
    double inverter = 1.0;

    for (long i = middleSignalIndex; i < vectorSize; ++i)
    {
        tempVar = originalVector[i];

        finalResult[i] = originalVector[vectorSize - i - 1] * (-inverter);
        finalResult[vectorSize - i - 1] = tempVar * inverter;
        inverter *= -1.0;
    }

    return finalResult;
}

auto calcOrthogonalVector(const std::vector<double>& vector) -> std::vector<double>
{
    std::vector<double> result(vector.size());
    double multiplier = 1;

    for (int index = static_cast<int>(vector.size()) - 1; index >= 0; index--)
    {
        result.at(vector.size() - (index + 1)) = vector.at(index) * multiplier;
        multiplier *= -1;
    }
    return result;
}

void normalizeVectorToSum1(double* signal, long signalLength)
{
    double sum = 0;

    for (long i = 0; i < signalLength; ++i)
    {
        sum += signal[i];
    }

    for (long i = 0; i < signalLength; ++i)
    {
        signal[i] /= sum;
    }
}

void normalizeVectorToSum1(std::vector<double>& signal)
{
    double sum = 0;

    for (double& v : signal)
    {
        sum += v;
    }

    for (double& v : signal)
    {
        v /= sum;
    }
}

void normalizeVectorToSum1AllPositive(std::vector<double>& signal)
{
    double min = signal[0];
    double sum = 0;

    for (double& v : signal)
    {
        min = std::min(v, min);
        sum += v;
    }

    if (min < 0)
    {
        sum = 0;
        for (double& v : signal)
        {
            sum += v += std::abs(min) + (double) 1; // @suppress("Ambiguous problem")
        }
    }

    for (double& v : signal)
    {
        v /= sum;
    }
}

void normalizeVectorToSum1AllPositive(double* signal, long signalLength)
{
    double min = signal[0];
    double sum = 0;

    for (long i = 0; i < signalLength; ++i)
    {
        min = std::min(signal[i], min);
        sum += signal[i];
    }

    if (min < 0)
    {
        sum = 0;
        for (long i = 0; i < signalLength; ++i)
        {
            sum += signal[i] += std::abs(min) + (double) 1; // @suppress("Ambiguous problem")
        }
    }

    for (long i = 0; i < signalLength; ++i)
    {
        signal[i] /= sum;
    }
}

void normalizeVectorToRange(double* signal, long signalLength, double lowerLimit, double upperLimit)
{
    if (lowerLimit >= upperLimit)
    {
        throw std::runtime_error("lowerLimit MUST be bigger than upperLimit");
    }

    /*
     * Keep this comment!
     *
     *	double min, max;
     *	min = max = signal[0];
     *
     *	for (auto &v : signal)
     *	{
     *		// Minimum
     *		if (v < min) min = v;
     *		// Maximum
     *		if (v > max) max = v;
     *	}
     *
     *	// Normalize between 0 and 1
     *	double rangeVal = max - min;
     *	for (auto &v : signal)
     *		v = (v - min) / rangeVal;
     *
     *	// Then scale to [lowerLimit,upperLimit]:
     *	double rangeLim = upperLimit - lowerLimit;
     *	for (auto &v : signal)
     *		v = (v * rangeLim) + lowerLimit;
     */

    // The code bellow do the same as the commented code above
    double min;
    double max;
    min = max = signal[0];

    for (long i = 0; i < signalLength; ++i)
    {
        // Minimum
        min = std::min(signal[i], min);
        // Maximum
        max = std::max(signal[i], max);
    }

    double rangeVal = max - min;
    double rangeLim = upperLimit - lowerLimit;
    for (long i = 0; i < signalLength; ++i)
    {
        signal[i] = (((signal[i] - min) / rangeVal) * rangeLim) + lowerLimit;
    }
}

void normalizeVectorToRange(std::vector<double>& signal, double lowerLimit, double upperLimit)
{
    if (lowerLimit >= upperLimit)
    {
        throw std::runtime_error("lowerLimit MUST be bigger than upperLimit");
    }
    /*
     * Keep this comment!
     *
     *	double min, max;
     *	min = max = signal[0];
     *
     *	for (auto &v : signal)
     *	{
     *		// Minimum
     *		if (v < min) min = v;
     *		// Maximum
     *		if (v > max) max = v;
     *	}
     *
     *	// Normalize between 0 and 1
     *	double rangeVal = max == min ? 1 : max - min;
     *	for (auto &v : signal)
     *		v = (v - min) / rangeVal;
     *
     *	// Then scale to [lowerLimit,upperLimit]:
     *	double rangeLim = upperLimit - lowerLimit;
     *	for (auto &v : signal)
     *		v = (v * rangeLim) + lowerLimit;
     */

    // The code bellow do the same as the commented code above
    double min;
    double max;
    min = max = signal[0];

    for (auto& v : signal)
    {
        // Minimum
        min = std::min(v, min);
        // Maximum
        max = std::max(v, max);
    }

    double rangeVal = max == min ? 1 : max - min;
    double rangeLim = upperLimit - lowerLimit;
    for (auto& v : signal)
    {
        v = (((v - min) / rangeVal) * rangeLim) + lowerLimit;
    }
}

auto convolution(double* data, long dataLength, double* kernel, long kernelSize) -> bool
{
    long i;
    long j;
    long k;

    auto* convolutedSignal = new double[dataLength];

    // check validity of params
    if ((data == nullptr) || (convolutedSignal == nullptr) || (kernel == nullptr))
    {
        delete[] convolutedSignal;
        return false;
    }

    if (dataLength <= 0 || kernelSize <= 0)
    {
        delete[] convolutedSignal;
        return false;
    }

    // start convolution from out[kernelSize-1] to out[dataSize-1] (last)
    for (i = kernelSize - 1; i < dataLength; ++i)
    {
        convolutedSignal[i] = 0; // init to 0 before accumulate

        for (j = i, k = 0; k < kernelSize; --j, ++k)
        {
            convolutedSignal[i] += data[j] * kernel[k];
        }
    }

    // convolution from out[0] to out[kernelSize-2]
    for (i = 0; i < kernelSize - 1; ++i)
    {
        convolutedSignal[i] = 0; // init to 0 before sum

        for (j = i, k = 0; j >= 0; --j, ++k)
        {
            convolutedSignal[i] += data[j] * kernel[k];
        }
    }

    std::copy(convolutedSignal, convolutedSignal + dataLength, data);

    delete[] convolutedSignal;

    return true;
}

inline auto getAlphaK(unsigned int k, unsigned int N) -> double
{
    return k == 0 ? std::sqrt(1.0 / N) : std::sqrt(2.0 / N);
}

auto discreteCosineTransform(std::vector<double>& vector) -> void
{
    std::vector<double> res(vector.size());
    unsigned int N = vector.size();

    // Create a vector of indices [0, 1, ..., N-1] for std::accumulate
    std::vector<unsigned int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    for (unsigned int k = 0; k < N; ++k)
    {
        double sum = std::accumulate(indices.begin(), indices.end(), 0.0,
                                     [&](double acc, unsigned int n) {
                                         return acc + vector[n] * std::cos(((2.0 * n + 1.0) * M_PI * k) / (2.0 * N));
                                     });

        res[k] = getAlphaK(k, N) * sum;
    }

    vector = res;
}

void discreteCosineTransform(double* vector, long vectorLength)
{
    auto* res = new double[vectorLength];
    unsigned int N = vectorLength;
    double sum = 0;

    for (unsigned int k = 0; k < N; ++k)
    {
        sum = 0;
        for (unsigned int n = 0; n < N; ++n)
        {
            sum += vector[n] * std::cos(((2.0 * n + 1.0) * M_PI * k) / (2.0 * N));
        }

        res[k] = getAlphaK(k, N) * sum;
    }

    for (unsigned int i = 0; i < N; ++i)
    {
        vector[i] = res[i];
    }

    delete[] res;
}

auto scaleMatrix(std::vector<std::vector<double>>& matrix) -> void
{
    // Points to the best line that can nullify our values
    unsigned int bestLineForSubtration = 0;

    // Selecting the line on which we have the value to nullify
    for (unsigned int lineIndex = 1; lineIndex < matrix.size(); lineIndex++)
    {
        // Selecting the column on which we have the value to nullify
        for (unsigned columnIndex = 0; columnIndex < lineIndex; columnIndex++)
        {
            // If this value is already zero then we are ok, move on
            if (matrix[lineIndex][columnIndex] == 0)
            {
                continue;
            }

            // Otherwise we must find the best line for subtraction
            bestLineForSubtration = 0;
            for (; bestLineForSubtration < matrix.size(); bestLineForSubtration++)
            {
                // The line must have an value different of
                // zero at the position we want nullify
                if (matrix[bestLineForSubtration][columnIndex] == 0)
                {
                    continue;
                }

                // The line must have zeros BEFORE the
                // current position we want to nullify
                bool zeroedBefore = true;
                for (unsigned int ci = 0; ci < columnIndex; ci++)
                {
                    if (matrix[bestLineForSubtration][ci] != 0)
                    {
                        zeroedBefore = false;
                        break;
                    }
                }
                if (!zeroedBefore)
                {
                    continue;
                }

                // We got it! Stop now!
                break;
            }

            // Check if a suitable line was actually found
            if (bestLineForSubtration == matrix.size() || matrix[bestLineForSubtration][columnIndex] == 0)
            {
                throw std::runtime_error(
                    "Matrix is singular or ill-conditioned: no suitable pivot found for column " +
                    std::to_string(columnIndex));
            }

            // Ready to calculate the coefficient
            double coef =
                matrix[lineIndex][columnIndex] / matrix[bestLineForSubtration][columnIndex];

            unsigned int ci = columnIndex;
            matrix[lineIndex][ci] = 0;
            ci++;

            for (; ci < matrix[lineIndex].size(); ci++)
            {
                matrix[lineIndex][ci] -= coef * matrix[bestLineForSubtration][ci];
            }
        }
    }
}

auto solveMatrix(std::vector<std::vector<double>>& matrix) -> std::vector<double>
{
    // Used to make the substitutions
    double temp;

    // final result
    std::vector<double> result(matrix.size());

    // The amount of the matrix columns
    unsigned int colums = matrix[0].size();

    // loop over all matrix lines from bottom to up
    for (int li = static_cast<int>(matrix.size()) - 1; li >= 0; li--)
    {
        // Make the substitutions:
        //	-Ignore the incognito variable (all values from main diagonal).-> "ci = li + 1"
        //	-Ignore the right side of the equation.-> "colums - 1"
        temp = 0;
        for (unsigned ci = li + 1; ci < colums - 1; ci++)
        {
            temp -= matrix[li][ci] * result[ci];
        }

        // The result is computed as follows:
        //	-Take the right side of the equation (the number) -> matrix[li][colums - 1]
        //	-Take the value multiplying the incognito variable -> matrix[li][li]
        //	-Make a substitution with previous result (the loop above)
        //	-Then sum the substitutions with the right side of equation and divide by incognito
        result[li] = (matrix[li][colums - 1] + temp) / matrix[li][li];
    }

    return result;
}

void resizeCentered(std::vector<double>& vector, long newSize, double defaultValue)
{
    if (newSize == vector.size())
    {
        return;
    }

    int diff = static_cast<int>(newSize - vector.size());
    long leftPadding = diff / 2;
    long rightPadding = diff - leftPadding;

    if (newSize < vector.size())
    {
        vector.erase(vector.begin(), vector.begin() + leftPadding * -1);
        vector.erase(vector.end() + rightPadding, vector.end());
        return;
    }

    std::vector<double> left(leftPadding, defaultValue);
    std::vector<double> right(rightPadding, defaultValue);

    vector.insert(vector.begin(), left.begin(), left.end());
    vector.insert(vector.end(), right.begin(), right.end());
}

} // namespace linearAlgebra
