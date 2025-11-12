/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 12 de abr de 2020
 */

#include "WaveletTransformResults.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace wavelets
{

WaveletTransformResults::WaveletTransformResults(long maxItems)
{
    this->maxItems = maxItems;
    if (maxItems > 0)
    {
        this->transformedSignal.resize(maxItems);
    }
}

/**
 * Extracts the approximation and details from the wavelet transformation
 * @param detailIndex -1: Return the whole transformed signal
 * @param detailIndex 0: Extracts the approximation
 * @param detailIndex 1 or more: Extracts the corresponding detail
 * @return Whole transformed signal, approximation or details
 */
auto WaveletTransformResults::getWaveletTransforms(int detailIndex) -> std::vector<double>
{
    // User is requesting more details then we had produced
    if (detailIndex > (int) this->levelsOfTransformation)
    {
        throw std::runtime_error("There is not a transformations at this level");
    }

    // Check maxItems limit if set
    if (this->maxItems > 0 && this->transformedSignal.size() > this->maxItems)
    {
        throw std::runtime_error("Signal exceeds maximum size limit");
    }

    // Returns the full transformed signal
    if (detailIndex == -1)
    {
        return this->transformedSignal;
    }

    // Creating the indexers witch will point to the start (sstrat)
    // and the end (send) of the signal we want
    unsigned send = 0;
    unsigned sstart = 0;

    // The container of our response
    std::vector<double> levelTransformedSignal;

    // This value is used in the calculations of indexers
    int log = (int) std::log2(this->transformedSignal.size());

    // User is requesting just the approximation
    if (detailIndex == 0)
    {
        // Approximations always starts at index 0
        sstart = 0;
        // The size of approximation depends on size of original
        // signal and the levels of transformations made
        send = (int) std::pow(2, log - this->levelsOfTransformation);
    }
    else
    {
        // If the executions reaches this part the user are interested in
        // details
        sstart = (int) std::pow(2, log - detailIndex);
        send = (int) std::pow(2, log - detailIndex + 1);
    }

    // Assembling the response
    for (long indexRange = sstart; indexRange < send; indexRange++)
    {
        levelTransformedSignal.push_back(this->transformedSignal.at(indexRange));
    }

    return levelTransformedSignal;
}

/**
 * Extracts the values of a wavelet packet transformation
 * differently from @getWaveletTransforms it DO NOT returns
 * the details of transformation, otherwise, returns the
 * generated chunks of the transformed signal
 * @param partIndex : A value from 0 up to @getWaveletPacketAmountOfParts
 * @param maxFrequecy
 * @return the requested chunk
 */
auto WaveletTransformResults::getWaveletPacketTransforms(long startIndex, long endIndex,
                                                         long maxFrequecy) -> std::vector<double>
{
    // Checks if endIndex < startIndex
    if (endIndex < startIndex)
    {
        throw std::runtime_error("EndIndex < startIndex this is WRONG!!!");
    }

    // Checks if this is a wavelet transform
    if (!this->packet)
    {
        throw std::runtime_error("This is not a wavelet packet transformed signal");
    }

    // Calculate the size of the chunks
    auto chunkSize = this->getWaveletPacketAmountOfParts() / maxFrequecy;

    // Get the ranges that must be returned
    long sstart = startIndex * chunkSize;
    long send = endIndex * chunkSize;

    // Returns the data
    return {this->transformedSignal.begin() + sstart, this->transformedSignal.begin() + send};
}

/**
 * Calculate the maximum number of generated
 * parts in a packet wavelet transform
 * @return maximum number of generated parts
 */
auto WaveletTransformResults::getWaveletPacketAmountOfParts() -> long
{
    // Checks if this is a wavelet transform
    if (!this->packet)
    {
        throw std::runtime_error("This is not a wavelet packet transformed signal");
    }

    return (long) std::pow(2, this->levelsOfTransformation);
}

/**
 * Static version of @getWaveletPacketTransforms(long partIndex)
 * Extracts the values of a wavelet packet transformation
 * differently from @getWaveletTransforms it DO NOT returns
 * the details of transformation, otherwise, returns the
 * generated chunks of the transformed signal
 * USE ONLY WITH PACKET WAVELETS!!
 * @param transformedSignal : vector with transformed signal
 * @param partIndex : A value from 0 up to @getWaveletPacketAmountOfParts
 * @param levelsOfTransformation : levels of transformation of the signal
 * @return the requested chunk
 */
auto WaveletTransformResults::getWaveletPacketTransforms(std::vector<double> transformedSignal,
                                                         long partIndex,
                                                         long levelsOfTransformation)
    -> std::vector<double>
{
    // The partIndex must not access non existent parts of the transformation
    if (WaveletTransformResults::getWaveletPacketAmountOfParts(levelsOfTransformation) - 1 <
        partIndex)
    {
        throw std::runtime_error("You are trying to access a non existent part of transformation");
    }

    // Calculate de size of the chuncks
    int chunkSize = transformedSignal.size() /
                    WaveletTransformResults::getWaveletPacketAmountOfParts(levelsOfTransformation);

    // Get the ranges that must be returned
    int sstart = partIndex * chunkSize;
    int send = sstart + chunkSize;

    // Returns the data
    return std::vector<double>(transformedSignal.begin() + sstart,
                               transformedSignal.begin() + send);
}

/**
 * Static version of @getWaveletPacketAmountOfParts()
 * Calculate the maximum number of generated
 * parts in a packet wavelet transform given
 * the levels of transformations performed
 * USE ONLY WITH PACKET WAVELETS!!
 * @param levelsOfTransformation
 * @return maximum number of generated parts
 */
auto WaveletTransformResults::getWaveletPacketAmountOfParts(long _levelsOfTransformation) -> long
{
    return (long) std::pow(2, _levelsOfTransformation);
}
} // namespace wavelets
