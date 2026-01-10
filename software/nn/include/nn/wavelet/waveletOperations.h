/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 1 de abr de 2020
 */

/**
 * @file waveletOperations.h
 * @brief Core wavelet transform entry points (Mallat) and analysis helpers.
 *
 * Main API:
 * - `malat(...)` runs a discrete wavelet transform using Mallat's algorithm.
 *   The low-pass filter coefficients are provided explicitly (see `Types.h`).
 * - `TransformMode` selects regular vs packet decomposition.
 *
 * The additional helpers provide common utilities used by experiments
 * (next power of two, subband energy extraction).
 */
#ifndef SRC_LIB_WAVELET_WAVELETOPERATIONS_H_
#define SRC_LIB_WAVELET_WAVELETOPERATIONS_H_

#include <sys/types.h>

#include <cstdint>
#include <span>
#include <vector>

#include "nn/wavelet/WaveletTransformResults.h"

namespace wavelets
{

/**
 * Indicates what a kind of wavelet
 * transformation that must be done
 */
enum TransformMode : uint8_t
{
    PACKET_WAVELET,
    REGULAR_WAVELET
};

/**
 * Applies a wavelets transform over a signal using the Mallat's algorithm
 * @param signal - signal to be transformed
 * @param lowpassfilter - the wavelet lowpass filter
 * @param level - levels of the signal decomposition
 * @param maxItens - the signal upper limit to be processed
 * @param highPassBranch - true: Do the decomposition in the highpass portion of the signal
 * (wavelet packet transform). false: Do a regular wavelet transform
 * @param mode - PACKET_WAVELET: wavelet packet, REGULAR_WAVELET: regular wavelet
 * @return transformed signal
 */
auto malat(const std::vector<double>& signal, const std::span<const double>& lowpassfilter,
           TransformMode mode, unsigned int level) -> WaveletTransformResults;

/**
 * Return the next power of two based number
 * @param number - The reference number
 * @return - Next power of two
 */
auto get_next_power_of_two(double number) -> int;

/**
 * Extract subband energies from wavelet transform results
 * @param transform - wavelet transform results
 * @param level - decomposition level
 * @return vector of RMS energies for each subband
 */
auto extract_subband_energies(const WaveletTransformResults& transform, int level)
    -> std::vector<double>;

} // namespace wavelets

#endif /* SRC_LIB_WAVELET_WAVELETOPERATIONS_H_ */
