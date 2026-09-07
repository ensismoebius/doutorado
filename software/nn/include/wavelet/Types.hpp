/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 26 de dez de 2025
 *
 * @see http://wavelets.pybytes.com/
 *
 */

/**
 * @file Types.hpp
 * @brief Compile-time wavelet "tags" and coefficient traits.
 *
 * This header defines:
 * - Tag structs (e.g., `Haar`, `Daub4`, ...), used only for template selection.
 *   See WaveletCoefficients.hpp.
 * - `WaveletTraits<Tag>` specializations providing the low-pass decomposition
 *   filter coefficients as `constexpr std::array<double, N>`. See WaveletTraits.hpp.
 *
 * Usage:
 * - Select a wavelet by tag type and access `WaveletTraits<Tag>::coeffs`.
 * - Coefficients live in read-only memory and do not allocate.
 *
 * Kept as a single thin file (not split per-type) so every existing includer
 * keeps working unchanged.
 */

#ifndef SRC_LIB_WAVELET_TYPES_H_
#define SRC_LIB_WAVELET_TYPES_H_

#include "wavelet/WaveletCoefficients.hpp"
#include "wavelet/WaveletTraits.hpp"

#endif /* SRC_LIB_WAVELET_TYPES_H_ */
