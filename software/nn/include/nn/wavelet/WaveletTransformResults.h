/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 12 de abr de 2020
 */

/**
 * @file WaveletTransformResults.h
 * @brief Container for wavelet transform output and helpers to extract subbands.
 *
 * The `transformedSignal` vector stores the concatenated coefficients generated
 * by the transform implementation.
 *
 * Key concepts:
 * - Regular wavelet transform: decomposes only the low-pass (approximation) branch.
 * - Packet wavelet transform: decomposes both low-pass and high-pass branches,
 *   producing a full binary tree of subbands.
 *
 * The helper methods here provide ways to slice out approximation/detail signals
 * or packet partitions from that flattened storage.
 */

#ifndef SRC_LIB_WAVELET_WAVELETTRANSFORMRESULTS_H_
#define SRC_LIB_WAVELET_WAVELETTRANSFORMRESULTS_H_

#include <vector>

namespace wavelets
{
class WaveletTransformResults
{
   private:
    /**
     * Maximum number of items allowed in the transformed signal
     */
    long maxItems = 0;

   public:
    /**
     * If true the results comes from a
     * packet wavelet transform
     */
    bool packet = false;

    /**
     * The levels of transformations
     * we have done to the signal
     */
    long levelsOfTransformation = 0;

    /**
     * The transformed signal
     */
    std::vector<double> transformedSignal;

    explicit WaveletTransformResults(long maxItems = 0);

    /**
     * Extracts the approximation and details from the wavelet transformation
     * @param detailIndex -1: Return the whole transformed signal
     * @param detailIndex 0: Extracts the approximation
     * @param detailIndex 1 or more: Extracts the corresponding detail
     * @return Whole transformed signal, approximation or details
     */
    auto get_wavelet_transforms(int detailIndex = -1) -> std::vector<double>;

    /**
     * Extracts the values of a wavelet packet transformation
     * differently from @get_wavelet_packet_transforms it DO NOT returns
     * the details of transformation, otherwise, returns the
     * generated chunks of the transformed signal
     * @param partIndex : A value from 0 up to @get_wavelet_packet_amount_of_parts
     * @param maxFrequecy
     * @return the requested chunk
     */
    auto get_wavelet_packet_transforms(long startIndex, long endIndex, long maxFrequecy)
        -> std::vector<double>;

    /**
     * Calculate the maximum number of generated
     * parts in a packet wavelet transform
     * @return maximum number of generated parts
     */
    [[nodiscard]] auto get_wavelet_packet_amount_of_parts() const -> long;

    /**
     * Static version of @get_wavelet_packet_transforms(long partIndex)
     * Extracts the values of a wavelet packet transformation
     * differently from @get_wavelet_transforms it DO NOT returns
     * the details of transformation, otherwise, returns the
     * generated chunks of the transformed signal
     * USE ONLY WITH PACKET WAVELETS!!
     * @param transformedSignal : vector with transformed signal
     * @param partIndex : A value from 0 up to
     * @get_wavelet_packet_amount_of_parts
     * @param levelsOfTransformation : levels of transformation of the
     * signal
     * @return the requested chunk
     */
    static auto get_wavelet_packet_transforms(std::vector<double> transformedSignal, long partIndex,
                                              long levelsOfTransformation) -> std::vector<double>;

    /**
     * Static version of @get_wavelet_packet_amount_of_parts()
     * Calculate the maximum number of generated
     * parts in a packet wavelet transform given
     * the levels of transformations performed
     * USE ONLY WITH PACKET WAVELETS!!
     * @param levelsOfTransformation
     * @return maximum number of generated parts
     */
    static auto get_wavelet_packet_amount_of_parts(long _levelsOfTransformation) -> long;
};
} // namespace wavelets

#endif /* SRC_LIB_WAVELET_WAVELETTRANSFORMRESULTS_H_ */
