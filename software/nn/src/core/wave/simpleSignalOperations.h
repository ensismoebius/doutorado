/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 2 de abr de 2020
 *
 */

#ifndef SRC_LIB_SIMPLESIGNALOPERATIONS_H_
#define SRC_LIB_SIMPLESIGNALOPERATIONS_H_

#include <vector>

/**
 * Average Magnitude Difference Function
 * @param vector
 * @return amdf vector
 */
auto amdf(std::vector<long double> vector) -> std::vector<long double>;

/**
 * Returns the amount of samples in order to
 * calculate the first formant of the signal
 * @param vector
 * @return integer
 */
auto findFZeroPeriodSamples(std::vector<long double> vector) -> unsigned int;

auto doAFineAmplification(double* signal, int signalLength) -> void;

auto silentHalfOfTheSoundTrack(double* signal, int signalLength) -> void;

auto xuxasDevilInvocation(double* signal, int signalLength) -> void;

auto halfVolume(double* signal, int signalLength) -> void;

auto addEchoes(double* signal, int signalLength) -> void;

#endif // SRC_LIB_SIMPLESIGNALOPERATIONS_H_
