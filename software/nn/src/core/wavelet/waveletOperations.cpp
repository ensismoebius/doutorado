#include "waveletOperations.h"

#include <cmath>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../linearAlgebra/linearAlgebra.h"
#include "WaveletTransformResults.h"

namespace wavelets
{

auto malat(const std::vector<double>& signal, std::span<const double>& lowpassfilter,
           TransformMode mode, unsigned int level, unsigned int maxItens, bool highPassBranch)
    -> WaveletTransformResults
{
    // If maxitems is not informed then get the full signal size
    if (maxItens == 0)
    {
        maxItens = signal.size();
    }
    else
    {
        // The number of items must be equal or less than the signal length
        if (maxItens > signal.size())
        {
            throw std::runtime_error(
                "The number of items must be equal or less than the signal length");
        }

        // The number of items must be equal or less than the half of signal length
        // when in the high pass branch of the signal (used in the wavelet packet
        // starting in the second transformation level)
        if (highPassBranch && (maxItens > signal.size() / 2))
        {
            throw std::runtime_error(
                "The number of items must be equal or less than the half of signal length when in "
                "the high pass branch of the signal");
        }
    }

    /*
     * There is a limit of transformations that can be done, depending
     * on the length of the signal, until we get coefficients with only
     * one number. The transformation levels shall not pass this limit
     * (log2(maxItens))
     */
    if (level > std::log2(maxItens))
    {
        std::stringstream s;
        s << "This signal only supports a maximum of " << static_cast<int>(std::log2(maxItens))
          << " levels.";
        throw std::runtime_error(s.str());
    }

    // Precompute high-pass filter once to eliminate repeated allocations
    // Optimization: Moved outside hot path, reduces dynamic allocations
    std::vector<double> highpassfilter = linearAlgebra::calcOrthogonalVector(lowpassfilter);

    // Create padded signal for circular convolution to eliminate modulo operations
    // Optimization: Pre-pad signal to avoid expensive % in inner loops, improves cache locality
    size_t filter_len = lowpassfilter.size();
    std::vector<double> padded(maxItens + filter_len - 1);
    for (size_t i = 0; i < padded.size(); ++i)
    {
        padded[i] = signal[i % maxItens];
    }

    // Create the storage for the final results with the correct size
    WaveletTransformResults results(maxItens);

    /*
     * The way we apply the filters and store the results for the
     * highpass and lowpass portions of the signal is different
     */
    if (highPassBranch)
    {
        // For high-pass branch (packet wavelet), access second half of signal
        // Keep modulo for correctness, as signal indexing is different
        // Translate the filters over the signal
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (unsigned int translation = 0; translation < maxItens; translation += 2)
        {
            double lowPassSum = 0;
            double highPassSum = 0;
            size_t signalIndex;

            // Make the sums for lowpass and highpass (i.e. apply the filters)
            for (unsigned int filterIndex = 0; filterIndex < filter_len; ++filterIndex)
            {
                // This part corresponds to the "wrap around" part of Mallat's algorithm
                signalIndex = (translation + filterIndex) % maxItens;

                /*
                 * When in highpass branch of the signal we just want the
                 * second half of the signal (signalIndex + maxItens). This
                 * is used only with wavelet packet transformations
                 */
                lowPassSum += signal[signalIndex + maxItens] * lowpassfilter[filterIndex];
                highPassSum += signal[signalIndex + maxItens] * highpassfilter[filterIndex];
            }

            // Stores the values according to Malat's algorithm
            // Optimization: Use [] instead of .at() after bounds validation
            /*
             * If we are decomposing the highpass branch then we need to swap the
             * high pass and low pass filtered signals in order to maintain the
             * signal order in the frequency domain. This is used only with
             * wavelet packet transformations
             */
            results.transformedSignal[translation / 2] = highPassSum;
            results.transformedSignal[(translation / 2) + (maxItens / 2)] = lowPassSum;
        }
    }
    else
    {
        // For regular branch, use padded signal for efficient circular access
        // Optimization: No modulo in inner loop, contiguous memory access
        // Translate the filters over the signal
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (unsigned int translation = 0; translation < maxItens; translation += 2)
        {
            double lowPassSum = 0;
            double highPassSum = 0;

            // Make the sums for lowpass and highpass (i.e. apply the filters)
            for (unsigned int filterIndex = 0; filterIndex < filter_len; ++filterIndex)
            {
                // Use padded signal for circular convolution without modulo
                size_t padded_index = translation + filterIndex;
                lowPassSum += padded[padded_index] * lowpassfilter[filterIndex];
                highPassSum += padded[padded_index] * highpassfilter[filterIndex];
            }

            // Stores the values according to Malat's algorithm
            // Optimization: Use [] instead of .at() for performance
            /*
             * If we are at the lowpass portion of the signal
             * then just store the values as the regular wavelet transform
             */
            results.transformedSignal[translation / 2] = lowPassSum;
            results.transformedSignal[(translation / 2) + (maxItens / 2)] = highPassSum;
        }
    }

    // Iterative level-by-level decomposition to remove recursion
    // Optimization: Eliminates function call overhead and stack usage for deep decompositions
    if (maxItens > 2 && level > 1)
    {
        // Define task structure: start index, size, remaining levels, is high-pass branch
        using Task = std::tuple<size_t, size_t, unsigned int, bool>;
        std::vector<Task> tasks;

        // Add initial sub-tasks based on mode
        if (mode == PACKET_WAVELET)
        {
            tasks.emplace_back(0, maxItens / 2, level - 1, false);           // low-pass branch
            tasks.emplace_back(maxItens / 2, maxItens / 2, level - 1, true); // high-pass branch
        }
        else
        {
            tasks.emplace_back(0, maxItens / 2, level - 1, false); // only low-pass for regular
        }

        // Process tasks iteratively
        while (!tasks.empty())
        {
            auto [start, sz, lev, is_high] = tasks.back();
            tasks.pop_back();

            if (lev == 0 || sz < 2) continue;

            // Extract segment to decompose
            std::vector<double> segment(sz);
            for (size_t i = 0; i < sz; ++i)
            {
                segment[i] = results.transformedSignal[start + i];
            }

            // Create padded segment for circular convolution
            // Optimization: Avoid modulo in inner loops
            std::vector<double> padded_seg(sz + filter_len - 1);
            for (size_t i = 0; i < padded_seg.size(); ++i)
            {
                padded_seg[i] = segment[i % sz];
            }

            // Perform convolution
            // Optimization: Parallel processing, contiguous memory access
            std::vector<double> temp(sz);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (size_t t = 0; t < sz; t += 2)
            {
                double lp = 0.0, hp = 0.0;
                for (size_t f = 0; f < filter_len; ++f)
                {
                    size_t idx = t + f;
                    lp += padded_seg[idx] * lowpassfilter[f];
                    hp += padded_seg[idx] * highpassfilter[f];
                }
                // Swap low/high for high-pass branches to maintain frequency order
                if (is_high)
                {
                    temp[t / 2] = hp;
                    temp[t / 2 + sz / 2] = lp;
                }
                else
                {
                    temp[t / 2] = lp;
                    temp[t / 2 + sz / 2] = hp;
                }
            }

            // Copy results back to main buffer
            // Optimization: Reuse existing buffer, avoid full copies
            for (size_t i = 0; i < sz; ++i)
            {
                results.transformedSignal[start + i] = temp[i];
            }

            // Add sub-tasks for next level
            size_t half = sz / 2;
            if (mode == PACKET_WAVELET)
            {
                tasks.emplace_back(start, half, lev - 1, false);
                tasks.emplace_back(start + half, half, lev - 1, true);
            }
            else
            {
                tasks.emplace_back(start, half, lev - 1, false);
            }
        }
    }

    // Set transformation metadata
    results.levelsOfTransformation = level;
    results.packet = (mode == PACKET_WAVELET);

    // Return the optimized result
    return results;
}

auto getNextPowerOfTwo(double number) -> int
{
    return static_cast<int>(std::pow(2, std::ceil(std::log2(number))));
}
} // namespace wavelets