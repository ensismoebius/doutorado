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

#include <vector>

/**
 * Create alpha value
 * @param samplingRate
 * @param filterMaxFrequency
 * @param highPass
 * @return alpha
 */
auto createAlpha(double samplingRate, double filterMaxFrequency, bool highPass = false) -> double;

/**
 * Create lowpass filter
 * @param order
 * @param samplingRate
 * @param filterMaxFrequency
 * @return lowpass filter coefficients
 */
auto createLowPassFilter(int order, double samplingRate, double filterMaxFrequency)
    -> std::vector<double>;

/**
 * Create highpass filter
 * @param order
 * @param samplingRate
 * @param filterStartFrequency
 * @return highpass filter coefficients
 */
auto createHighPassFilter(int order, double samplingRate, double filterStartFrequency)
    -> std::vector<double>;

/**
 * Create bandpass filter
 * @param order
 * @param samplingRate
 * @param startFrequency
 * @param finalFrequency
 * @return bandpass filter coefficients
 */
auto createStopBandFilter(int order, double samplingRate, double startFrequency,
                          double finalFrequency) -> std::vector<double>;

/**
 * Create bandstop filter
 * @param order
 * @param samplingRate
 * @param startFrequency
 * @param finalFrequency
 * @return bandstop filter coefficients
 */
auto bandStopFilter(int order, double samplingRate, double startFrequency, double finalFrequency)
    -> std::vector<double>;

/**
 * Create a window for signals
 * @param order
 * @return window coefficients
 */
auto createTriangularWindow(int order) -> std::vector<double>;

/**
 * Apply window to filter
 * @param filter filter coefficients (modified in-place)
 * @param window window coefficients
 */
void applyWindow(std::vector<double>& filter, const std::vector<double>& window);

#endif
