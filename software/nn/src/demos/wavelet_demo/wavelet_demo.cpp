#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <iostream>

#include "core/tensor/Tensor.hpp"
#include "core/wavelet/Types.h"
#include "core/wavelet/waveletOperations.h"
#include "core/wavelet/WaveletTransformResults.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

/**
 * @brief Generate a composite signal with two sine waves
 *
 * @param freq1
 * @param freq2
 * @param sample_rate
 * @param duration_seconds
 * @return std::vector<double>
 */
auto generateSignal(double freq1, double freq2, size_t sample_rate,
                    size_t duration_seconds) -> std::vector<double> {
    int signal_length = static_cast<int>(duration_seconds * sample_rate);
    std::vector<double> signal(signal_length);

    for (int n = 0; n < signal_length; n++) {
        signal[n] =
            (0.7 * sin((2.0 * M_PI * freq1 * static_cast<double>(n)) /
                       static_cast<double>(sample_rate))) +
            (0.3 * sin((2.0 * M_PI * freq2 * static_cast<double>(n)) /
                       static_cast<double>(sample_rate)));
    }
    return signal;
}

/**
 * @brief Plot the signal using matplotlibcpp
 *
 * @param signal_vector
 * @param title
 * @param show_blocking
 */
void plotSignal(const std::vector<double> &signal_vector, const std::string &title,
                bool show_blocking) {
    plt::plot(signal_vector);
    plt::title(title);
    if(show_blocking) {
        plt::show();
    }
}

void plotDecomposition(const std::vector<double> &approximation, const std::vector<double> &details,
                       const std::string &title) {
    plt::figure();
    plt::subplot(2, 1, 1);
    plt::plot(approximation);
    plt::title("Approximation");

    plt::subplot(2, 1, 2);
    plt::plot(details);
    plt::title("Details");

    plt::suptitle(title);
}

void plotPacketDecomposition(wavelets::WaveletTransformResults& results, const std::string &title) {
    plt::figure();
    long n_parts = results.getWaveletPacketAmountOfParts();
    std::cout << "Number of parts: " << n_parts << std::endl;

    plt::subplot(1, 1, 1);
    auto part = wavelets::WaveletTransformResults::getWaveletPacketTransforms(results.transformedSignal, 0, results.levelsOfTransformation);
    plt::plot(part);
    plt::title("Packet 0");

    plt::suptitle(title);
}


auto main() -> int {
    std::vector<double> x = {1, 2, 3, 4};
    std::vector<double> y = {1, 4, 9, 16};

    plt::figure();
    plt::plot(x, y);
    plt::show(true);

    return 0;
}