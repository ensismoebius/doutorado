/**
 * @file waveletOperations.cpp
 * @brief Wavelet transform operations (Mallat algorithm, regular/packet modes).
 */

#include "nn/wavelet/waveletOperations.h"

#include <bit> // For std::bit_width (C++20)
#include <cmath>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nn/linearAlgebra/linear_algebra.hpp"
#include "nn/wavelet/WaveletTransformResults.h"

namespace wavelets
{

auto malat(const std::vector<double>& signal,
    const std::span<const double>& lowpassfilter,
    TransformMode mode,
    unsigned int level) -> WaveletTransformResults
{
    // The total number of items to process is the size of the input signal.
    // This variable will represent the effective size of the signal at the current processing
    // level.
    size_t current_signal_size = signal.size();

    // The signal size must be a power of two for Mallat's algorithm.
    if ((current_signal_size == 0) || ((current_signal_size & (current_signal_size - 1)) != 0))
    {
        throw std::invalid_argument("Signal size must be a power of two and greater than zero.");
    }

    /*
     * There is a limit of transformations that can be done, depending
     * on the length of the signal, until we get coefficients with only
     * one number. The transformation levels shall not pass this limit
     * (log2(current_signal_size))
     */
    if (level == 0)
    { // If level is 0, no transformation is done, just return the signal
        WaveletTransformResults results((long) current_signal_size);
        results.transformedSignal = signal; // Direct copy
        results.levelsOfTransformation = 0;
        results.packet = (mode == PACKET_WAVELET);
        return results;
    } // LCOV_EXCL_LINE
    // Calculate max levels using std::bit_width for power-of-two sizes
    const unsigned int max_levels = std::bit_width(current_signal_size) - 1;
    if (level > max_levels)
    {
        std::stringstream s;
        s << "This signal only supports a maximum of " << max_levels << " levels.";
        throw std::invalid_argument(s.str());
    }

    // Precompute high-pass filter once to eliminate repeated allocations.
    // Optimization: Moved outside hot path, reduces dynamic allocations.
    std::vector<double> highpassfilter = linearAlgebra::calcOrthogonalVector(lowpassfilter);
    size_t filter_len = lowpassfilter.size();

    // Main working buffers.
    // `results.transformedSignal` will hold the current state of the transformed signal.
    // `temp_buffer` will hold intermediate results during convolution before swapping.
    WaveletTransformResults results((long) current_signal_size);
    results.transformedSignal = signal; // Initialize with the input signal
    std::vector<double> temp_buffer(current_signal_size);

    // Define task structure: (start_index, segment_size, is_high_pass_branch)
    // The current_level is implicitly handled by the outer loop iteration.
    struct Task
    {
        size_t start_idx;
        size_t size;
        bool is_high_pass;
    };
    std::vector<Task> tasks; // Queue of segments to process in the current level

    // Initial task for the first level of decomposition
    // Optimization: The initial padding is handled segment-wise within the loop
    tasks.emplace_back(0, current_signal_size, false);

    // Iterative level-by-level decomposition to replace recursion
    // Optimization: Eliminates function call overhead, stack usage for deep decompositions,
    // and repeated vector allocations by reusing main buffers.
    // Loop `level` times for each decomposition level.
    for (unsigned int l = 0; l < level; ++l)
    {
        // `tasks_for_next_level` stores new segments to be processed in the next iteration.
        // Optimization: Pre-allocated and cleared, reducing dynamic allocations.
        std::vector<Task> tasks_for_next_level;
        if (mode == PACKET_WAVELET)
        {
            tasks_for_next_level.reserve(tasks.size() * 2);
        }
        else
        {
            tasks_for_next_level.reserve(tasks.size());
        }

        // Process all segments (`tasks`) for the current level
        for (const auto& task : tasks)
        {
            size_t current_start = task.start_idx;
            size_t current_sz = task.size;
            bool current_is_high_pass = task.is_high_pass;

            // If the segment size is too small for the filter, it cannot be decomposed further.
            // This segment will be carried over to the next level's tasks if it's part of
            // a wavelet packet transform or if it's the approximation in a DWT.
            if (current_sz < filter_len - 1)
            { // -1 because filter_len-1 is the minimum useful size
                // This segment cannot be transformed, but it still contributes to the overall
                // signal. We just pass it through to the next level's tasks if it's relevant.
                tasks_for_next_level.emplace_back(
                    current_start, current_sz, current_is_high_pass); // LCOV_EXCL_LINE
                continue;                                             // LCOV_EXCL_LINE
            }

            // Perform convolution on the current segment
            // Optimization: Parallel processing, contiguous memory access.
            // Using `temp_buffer` for intermediate results to avoid in-place modification issues
            // and ensure correct data for subsequent segment processing in the same level.
            size_t half_sz = current_sz / 2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (size_t t = 0; t < current_sz; t += 2) // Iterate for downsampled output
            {
                double lp_sum = 0.0;
                double hp_sum = 0.0;

                for (size_t f = 0; f < filter_len; ++f)
                {
                    size_t idx_in_segment = t + f;
                    // Calculate circular index without modulo
                    size_t signal_idx = current_start + ((idx_in_segment >= current_sz)
                                                                ? (idx_in_segment - current_sz)
                                                                : idx_in_segment);
                    lp_sum += results.transformedSignal[signal_idx] * lowpassfilter[f];
                    hp_sum += results.transformedSignal[signal_idx] * highpassfilter[f];
                }

                // Store results in `temp_buffer` based on whether it's a high-pass branch
                // Optimization: Efficient write to contiguous memory.
                // The order of lp_sum/hp_sum depends on the `is_high_pass` flag
                // which represents the parent branch. For wavelet packet, if the parent
                // was a high-pass branch, we swap the low-pass and high-pass outputs
                // to maintain frequency order.
                if (mode == PACKET_WAVELET && current_is_high_pass) [[likely]]
                {
                    temp_buffer[current_start + (t / 2)] = hp_sum;
                    temp_buffer[current_start + (t / 2) + half_sz] = lp_sum;
                }
                else [[unlikely]]
                {
                    temp_buffer[current_start + (t / 2)] = lp_sum;
                    temp_buffer[current_start + (t / 2) + half_sz] = hp_sum;
                }
            }

            // After processing, add new tasks for the next level's decomposition.
            // These tasks represent the new low-pass and high-pass bands.
            if (mode == PACKET_WAVELET) [[likely]]
            {
                // Both low-pass and high-pass branches become new tasks for the next level
                tasks_for_next_level.emplace_back(current_start, half_sz, false); // Low-pass child
                tasks_for_next_level.emplace_back(
                    current_start + half_sz, half_sz, true); // High-pass child
            }
            else [[unlikely]]
            {
                // For regular DWT, only the low-pass branch is decomposed further
                tasks_for_next_level.emplace_back(
                    current_start, half_sz, false); // Only low-pass child
            }
        }
        // After all segments of the current level are processed, swap `results.transformedSignal`
        // with `temp_buffer`. This effectively "commits" the changes of the current level
        // to `results.transformedSignal` and prepares `temp_buffer` to be filled in the
        // next iteration with new intermediate results.
        std::swap(results.transformedSignal, temp_buffer);

        // Prepare tasks for the next level
        tasks = std::move(tasks_for_next_level);
        if (tasks.empty()) [[unlikely]]
        {
            break; // No more tasks to process (e.g., all segments are too small) // LCOV_EXCL_LINE
        }
    }

    // Set transformation metadata
    results.levelsOfTransformation = level;
    results.packet = (mode == PACKET_WAVELET);

    // Return the optimized result
    return results;
}

/**
 * @brief Returns the next power of two greater than or equal to the input.
 */
auto get_next_power_of_two(double number) -> int
{
    return static_cast<int>(std::pow(2, std::ceil(std::log2(number))));
}

/**
 * @brief Extracts RMS energies for each subband from a transform result.
 *
 * For packet transforms this returns one value per packet part. For regular
 * transforms this returns one value for approximation and one per detail band.
 */
auto extract_subband_energies(const WaveletTransformResults& transform, int level)
    -> std::vector<double>
{
    std::vector<double> energies;

    if (transform.packet)
    {
        long num_parts = transform.get_wavelet_packet_amount_of_parts();

        for (long i = 0; i < num_parts; ++i)
        {
            auto part = WaveletTransformResults::get_wavelet_packet_transforms(
                transform.transformedSignal, i, transform.levelsOfTransformation);

            if (!part.empty())
            {
                const double energy =
                    std::inner_product(part.begin(), part.end(), part.begin(), 0.0);
                energies.push_back(std::sqrt(energy)); // RMS energy
            }
        }
    }
    else // Regular wavelet transform
    {
        // For regular transform, we have 1 approximation and `level` details
        for (int i = 0; i <= transform.levelsOfTransformation; ++i)
        {
            // The getWaveletTransforms is not const, so I need to create a copy
            WaveletTransformResults temp_transform = transform;
            auto part = temp_transform.get_wavelet_transforms(i);

            if (!part.empty())
            {
                const double energy =
                    std::inner_product(part.begin(), part.end(), part.begin(), 0.0);
                energies.push_back(std::sqrt(energy)); // RMS energy
            }
        }
    }

    return energies; // LCOV_EXCL_LINE
}

} // namespace wavelets
