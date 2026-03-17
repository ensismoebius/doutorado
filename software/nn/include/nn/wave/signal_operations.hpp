/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 2 de abr de 2020
 *
 */

/**
 * @file signal_operations.hpp
 * @brief Small signal-processing utilities (pitch period heuristics, simple effects).
 *
 * This header collects standalone operations used by older experiments:
 * - AMDF (Average Magnitude Difference Function)
 * - Rough F0 period estimation helpers
 * - In-place simple effects (amplification, silence, half volume, echo)
 *
 * These functions work on raw arrays/vectors and are independent of the NN layers.
 */

#ifndef SRC_LIB_SIMPLESIGNALOPERATIONS_H_
#define SRC_LIB_SIMPLESIGNALOPERATIONS_H_

#include <vector>

/**
 * Average Magnitude Difference Function
 * @param vector
 * @return amdf vector
 */
auto amdf(const std::vector<long double>& vector) -> std::vector<long double>;

/**
 * Returns the amount of samples in order to
 * calculate the first formant of the signal
 * @param vector
 * @return integer
 */
auto findFZeroPeriodSamples(const std::vector<long double>& vector) -> unsigned int;

void doAFineAmplification(double* signal, int signalLength);

void silentHalfOfTheSoundTrack(double* signal, int signalLength);

void halfVolume(double* signal, int signalLength);

void addEchoes(double* signal, int signalLength);

#endif /* SRC_LIB_SIMPLESIGNALOPERATIONS_H_ */
