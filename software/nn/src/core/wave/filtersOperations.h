/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 2 de abr de 2020
 *
 */
#ifndef SRC_LIB_FILTEROPERATIONS_H_
#define SRC_LIB_FILTEROPERATIONS_H_

/**
 * Create alpha value
 * @param samplingRate
 * @param filterMaxFrequency
 * @param highPass
 * @return alpha
 */
auto createAlpha(double samplingRate, double filterMaxFrequency, bool highPass = false) -> long double;

/**
 * Create lowpass filter
 * @param order
 * @param samplingRate
 * @param filterMaxFrequency
 * @return lowpass filter
 */
auto createLowPassFilter(int order, double samplingRate, double filterMaxFrequency) -> long double*;

/**
 * Create highpass filter
 * @param order
 * @param samplingRate
 * @param filterStartFrequency
 * @return highpass filter
 */
auto createHighPassFilter(int order, double samplingRate, double filterStartFrequency) -> long double*;

/**
 * Create bandpass filter
 * @param order
 * @param samplingRate
 * @param startFrequency
 * @param finalFrequency
 * @return bandpass filter
 */
auto createStopBandFilter(int order, double samplingRate, double startFrequency, double finalFrequency) -> long double*;

/**
 * Create bandstop filter
 * @param order
 * @param samplingRate
 * @param startFrequency
 * @param finalFrequency
 * @return bandstop filter
 */
auto bandStopFilter(int order, double samplingRate, double startFrequency, double finalFrequency) -> long double*;

/**
 * Create a window for signals
 * @param order
 * @return window
 */
auto createTriangularWindow(int order) -> long double*;

/**
 * Apply window
 * @param filter
 * @param window
 * @param order
 */
void applyWindow(long double *filter, long double *window, int order);

#endif
