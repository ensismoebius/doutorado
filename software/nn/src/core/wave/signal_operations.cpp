/**
 * @file signal_operations.cpp
 * @brief Low-level signal processing helpers (AMDF, period estimation, simple editing).
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

/**
 * @brief Computes the Average Magnitude Difference Function (AMDF) curve.
 *
 * The output has the same length as the input and stores per-lag
 * accumulated absolute differences.
 */
auto amdf(const std::vector<long double>& vector) -> std::vector<long double>
{
    unsigned int resultIndex = -1;
    unsigned int N = vector.size();
    std::vector<long double> result(N);

    for (unsigned int k = 0; k < N; k++)
    {
        resultIndex++;
        for (unsigned int n = 0; n < N - k; n++)
        {
            result[resultIndex] += std::abs(vector[n] - vector[n + k]);
        }
    }
    return result;
}

/**
 * @brief Estimates the fundamental period from the AMDF curve minima.
 *
 * @throws std::invalid_argument when the input vector is empty.
 */
auto findFZeroPeriodSamples(const std::vector<long double>& vector) -> unsigned int
{
    if (vector.empty())
    {
        throw std::invalid_argument("findFZeroPeriodSamples: vector cannot be empty");
    }

    long double m = *std::min_element(vector.begin(), vector.end());
    unsigned int period = 0;
    unsigned int index = 0;

    while (vector[index] != m)
    {
        index++;
    }

    do
    {
        period++;
        index++;
    } while (vector[index] != m);

    return period;
}

/**
 * @brief Scales the signal so the largest absolute sample maps to 32767.
 */
void doAFineAmplification(double* signal, int signalLength)
{
    double highestSignal = 0;

    // find the highest signal
    for (int i = 0; i < signalLength; ++i)
    {
        double value = std::abs(signal[i]);

        highestSignal = std::max(value, highestSignal);
    }

    double multiplicationRatio = 32767 / highestSignal;

    for (int i = 0; i < signalLength; ++i)
    {
        signal[i] *= multiplicationRatio;
    }
}

/**
 * @brief Mutes the second half of the signal buffer in-place.
 */
void silentHalfOfTheSoundTrack(double* signal, int signalLength)
{
    int middleSignalIndex = signalLength / 2;

    for (int i = middleSignalIndex; i < signalLength; ++i)
    {
        signal[i] = 0;
    }
}

/**
 * @brief Applies a constant 0.5 gain to the entire signal buffer.
 */
void halfVolume(double* signal, int signalLength)
{
    for (int i = 0; i < signalLength; ++i)
    {
        signal[i] *= .5;
    }
}

/**
 * @brief Applies a simple delayed echo effect in-place.
 */
void addEchoes(double* signal, int signalLength)
{
    // the "time" sound get to bounce and back
    int bouncingTime = 100000;

    // Iterate over all values
    for (int i = 0; i < signalLength; ++i)
    {
        // while the data starts to end we decrease
        // the bouncing time to avoid access data
        // outside the array
        if (i + bouncingTime - 1 == signalLength)
        {
            bouncingTime--;
        }

        // we have to wait the bouncingTime before start echoing
        if (i > bouncingTime - 1)
        {
            // the resulting signal are going to be
            // the average of the current signal
            // plus 80% of the previous signal
            signal[i] = (signal[i - bouncingTime] * .8 + signal[i]) / 2;
        }
    }
}
