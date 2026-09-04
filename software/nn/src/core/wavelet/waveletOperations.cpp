/**
 * @file waveletOperations.cpp
 * @brief Wavelet transform operations (Mallat algorithm, regular/packet modes).
 */

#include "wavelet/waveletOperations.hpp"

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

#include "linear_algebra/linear_algebra.hpp"
#include "wavelet/WaveletTransformResults.hpp"

namespace wavelets
{

/// One contiguous piece of the signal still awaiting decomposition.
///
/// `is_high_pass` records which branch of the PARENT produced this segment.
/// It matters only for the packet transform, where a segment descended from
/// a high-pass branch has its two outputs written in swapped order so the
/// final layout stays in ascending frequency.
struct Task
{
    size_t start_idx;
    size_t size;
    bool is_high_pass;
};

/// Circular-convolve one segment with both filters and write the two
/// downsampled halves into `out`.
///
/// Reading from `in` and writing to `out` is not an optimization detail but
/// a correctness requirement: several segments of the same level are
/// processed in sequence, and convolving in place would feed a later
/// segment the values an earlier one had already overwritten.
///
/// Extracted from a 190-line `malat`. The call is per SEGMENT while the work
/// inside is `size * filter_len` multiply-accumulates, so the call cannot
/// show up against it -- and being static in this translation unit, -O3
/// inlines it anyway.
void decompose_segment(const Task& task,
    TransformMode mode,
    const std::vector<double>& in,
    std::vector<double>& out,
    const std::span<const double>& lowpassfilter,
    const std::vector<double>& highpassfilter)
{
    const size_t current_start = task.start_idx;
    const size_t current_sz = task.size;
    const size_t filter_len = lowpassfilter.size();
    const size_t half_sz = current_sz / 2;

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
            size_t signal_idx =
                current_start +
                ((idx_in_segment >= current_sz) ? (idx_in_segment - current_sz) : idx_in_segment);
            lp_sum += in[signal_idx] * lowpassfilter[f];
            hp_sum += in[signal_idx] * highpassfilter[f];
        }

        // Store results in `temp_buffer` based on whether it's a high-pass branch
        // Optimization: Efficient write to contiguous memory.
        // The order of lp_sum/hp_sum depends on the `is_high_pass` flag
        // which represents the parent branch. For wavelet packet, if the parent
        // was a high-pass branch, we swap the low-pass and high-pass outputs
        // to maintain frequency order.
        if (mode == PACKET_WAVELET && task.is_high_pass) [[likely]]
        {
            out[current_start + (t / 2)] = hp_sum;
            out[current_start + (t / 2) + half_sz] = lp_sum;
        }
        else [[unlikely]]
        {
            out[current_start + (t / 2)] = lp_sum;
            out[current_start + (t / 2) + half_sz] = hp_sum;
        }
    }
}

/// Reject a signal Mallat's algorithm cannot decompose, and say why.
///
/// Two separate impossibilities, both fatal and both worth naming:
///
/// * a size that is not a power of two -- every level halves the segment,
///   so a size of 100 would reach 25 and then have no valid split;
/// * more levels than the signal can carry. A signal of 1024 samples
///   supports 10 (2^10 = 1024); asking for 11 would decompose a segment of
///   one sample.
///
/// Returns the maximum level count, which the caller has already paid for
/// computing. Throwing rather than clamping is deliberate: silently
/// transforming to level 8 when 11 was requested produces a result that is
/// perfectly well-formed and answers a different question.
auto validate_and_max_levels(size_t signal_size, unsigned int level) -> unsigned int
{
    // The signal size must be a power of two for Mallat's algorithm.
    if ((signal_size == 0) || ((signal_size & (signal_size - 1)) != 0))
    {
        throw std::invalid_argument("Signal size must be a power of two and greater than zero.");
    }

    // There is a limit to how many transformations a signal can carry: each
    // level halves the segment, so the chain ends when a coefficient is one
    // number wide. That limit is log2(signal_size), which for a power of two
    // is exactly `bit_width - 1`.
    const unsigned int max_levels = std::bit_width(signal_size) - 1;
    if (level > max_levels)
    {
        std::stringstream s;
        s << "This signal only supports a maximum of " << max_levels << " levels.";
        throw std::invalid_argument(s.str());
    }

    return max_levels;
}

/// Decompose every segment of one level, and return the segments the next
/// level will work on.
///
/// `signal` is read and `scratch` is written; the caller swaps them, which
/// is what commits the level. The seeding copy on the first line is not
/// redundant: this function only WRITES the segments it decomposes, while
/// the caller swaps the WHOLE buffers -- so any region left untouched (the
/// detail bands a regular DWT carries along) would resurface stale data
/// from two levels back. At three levels that silently replaced cD2 with
/// the second half of cA1. Packet mode never showed it, because its tasks
/// cover the whole signal at every level.
auto decompose_one_level(const std::vector<Task>& tasks,
    TransformMode mode,
    const std::vector<double>& signal,
    std::vector<double>& scratch,
    const std::span<const double>& lowpassfilter,
    const std::vector<double>& highpassfilter) -> std::vector<Task>
{
    // Seed this level's output with the current state: the task loop only
    // writes the segments it decomposes, and the swap below exchanges the
    // WHOLE buffers — without this copy, every untouched region (e.g. the
    // detail bands regular DWT carries along) resurfaces stale data from
    // two levels ago after the swap. At 3 levels that silently replaced
    // cD2 with the second half of cA1. Found by pywt_parity_gtest; packet
    // mode was never affected (its tasks cover the full signal each level).
    scratch = signal;

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

    const size_t filter_len = lowpassfilter.size();

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
            tasks_for_next_level.push_back(
                Task{current_start, current_sz, current_is_high_pass}); //
            continue;                                                   //
        }

        // Perform convolution on the current segment
        // Optimization: Parallel processing, contiguous memory access.
        // Using `temp_buffer` for intermediate results to avoid in-place modification issues
        // and ensure correct data for subsequent segment processing in the same level.
        const size_t half_sz = current_sz / 2;
        decompose_segment(task, mode, signal, scratch, lowpassfilter, highpassfilter);

        // After processing, add new tasks for the next level's decomposition.
        // These tasks represent the new low-pass and high-pass bands.
        if (mode == PACKET_WAVELET) [[likely]]
        {
            // Both low-pass and high-pass branches become new tasks for the next level
            tasks_for_next_level.push_back(Task{current_start, half_sz, false}); // Low-pass child
            tasks_for_next_level.push_back(
                Task{current_start + half_sz, half_sz, true}); // High-pass child
        }
        else [[unlikely]]
        {
            // For regular DWT, only the low-pass branch is decomposed further
            tasks_for_next_level.push_back(
                Task{current_start, half_sz, false}); // Only low-pass child
        }
    }
    // After all segments of the current level are processed, swap `results.transformedSignal`
    // with `temp_buffer`. This effectively "commits" the changes of the current level
    // to `results.transformedSignal` and prepares `temp_buffer` to be filled in the
    return tasks_for_next_level;
}

auto malat(const std::vector<double>& signal,
    const std::span<const double>& lowpassfilter,
    TransformMode mode,
    unsigned int level) -> WaveletTransformResults
{
    // The total number of items to process is the size of the input signal.
    // This variable will represent the effective size of the signal at the current processing
    // level.
    size_t current_signal_size = signal.size();

    // Throws when the size is not a power of two or the level is beyond what
    // the signal can carry. Runs BEFORE the level == 0 shortcut below: that
    // path returns the signal untouched, and a malformed size has to be
    // rejected whether or not any transform was asked for.
    validate_and_max_levels(current_signal_size, level);

    if (level == 0)
    { // If level is 0, no transformation is done, just return the signal
        WaveletTransformResults results((long) current_signal_size);
        results.transformedSignal = signal; // Direct copy
        results.levelsOfTransformation = 0;
        results.packet = (mode == PACKET_WAVELET);
        return results;
    } //

    // Precompute high-pass filter once to eliminate repeated allocations.
    // Optimization: Moved outside hot path, reduces dynamic allocations.
    std::vector<double> highpassfilter = linearAlgebra::calc_orthogonal_vector(lowpassfilter);

    // Main working buffers.
    // `results.transformedSignal` will hold the current state of the transformed signal.
    // `temp_buffer` will hold intermediate results during convolution before swapping.
    WaveletTransformResults results((long) current_signal_size);
    results.transformedSignal = signal; // Initialize with the input signal
    std::vector<double> temp_buffer(current_signal_size);

    std::vector<Task> tasks; // Queue of segments to process in the current level

    // Initial task for the first level of decomposition
    // Optimization: The initial padding is handled segment-wise within the loop
    tasks.push_back(Task{0, current_signal_size, false});

    // Level-by-level rather than recursive: a deep decomposition would
    // otherwise nest as deep as `level`, and the two buffers below are
    // reused across levels instead of being reallocated per call.
    for (unsigned int l = 0; l < level; ++l)
    {
        std::vector<Task> tasks_for_next_level = decompose_one_level(
            tasks, mode, results.transformedSignal, temp_buffer, lowpassfilter, highpassfilter);

        // The swap is what COMMITS the level: `decompose_one_level` wrote
        // into `temp_buffer`, and after this the two names have traded
        // places, so the next level reads what this one produced.
        std::swap(results.transformedSignal, temp_buffer);

        tasks = std::move(tasks_for_next_level);
        if (tasks.empty()) [[unlikely]]
        {
            break; // No more tasks to process (e.g., all segments are too small) //
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
                energies.push_back(std::sqrt(energy)); // L2 norm = sqrt(Σx²) (root-energy, not RMS)
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
                energies.push_back(std::sqrt(energy)); // L2 norm = sqrt(Σx²) (root-energy, not RMS)
            }
        }
    }

    return energies;
} //

} // namespace wavelets
