#include <cmath>
#include <cstddef>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "core/wavelet/Types.h"
#include "core/wavelet/waveletOperations.h"
#include "core/wavelet/WaveletTransformResults.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

/**
 * @brief Generate a composite signal with two sine waves
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

void plot_signal(const std::vector<double>& signal) {
    try {
        plt::figure();
        plt::plot(signal);
        plt::title("Input Signal");
    } catch (const std::runtime_error& e) {
        std::cerr << "matplotlib-cpp error in plot_signal: " << e.what() << std::endl;
    }
}

void plot_dwt_levels(std::vector<double>& signal, const std::vector<double>& filter, int max_level) {
    for (int level = 1; level <= max_level; ++level) {
        try {
            auto dwt_results = wavelets::malat(signal, const_cast<std::vector<double>&>(filter), wavelets::REGULAR_WAVELET, level);
            auto approximation = dwt_results.getWaveletTransforms(0);
            
            plt::figure();
            plt::suptitle("DWT Level " + std::to_string(level));

            plt::subplot(max_level + 1, 1, 1);
            plt::plot(approximation);
            plt::title("Approximation");

            for (int i = 1; i <= level; ++i) {
                auto details = dwt_results.getWaveletTransforms(i);
                plt::subplot(max_level + 1, 1, i + 1);
                plt::plot(details);
                plt::title("Detail " + std::to_string(i));
            }
        } catch (const std::runtime_error& e) {
            std::cerr << "matplotlib-cpp error in plot_dwt_levels (level " << level << "): " << e.what() << std::endl;
        }
    }
}

void plot_dwpt_levels(std::vector<double>& signal, const std::vector<double>& filter, int max_level) {
    for (int level = 1; level <= max_level; ++level) {
        try {
            auto dwpt_results = wavelets::malat(signal, const_cast<std::vector<double>&>(filter), wavelets::PACKET_WAVELET, level);
            long n_parts = dwpt_results.getWaveletPacketAmountOfParts();

            plt::figure();
            plt::suptitle("DWPT Level " + std::to_string(level));

            for (long i = 0; i < n_parts; ++i) {
                plt::subplot(n_parts, 1, i + 1);
                auto part = wavelets::WaveletTransformResults::getWaveletPacketTransforms(dwpt_results.transformedSignal, i, dwpt_results.levelsOfTransformation);
                plt::plot(part);
                plt::title("Packet " + std::to_string(i));
            }
        } catch (const std::runtime_error& e) {
            std::cerr << "matplotlib-cpp error in plot_dwpt_levels (level " << level << "): " << e.what() << std::endl;
        }
    }
}


auto main() -> int {
    // Signal Generation
    const size_t duration_seconds = 1;
    const size_t sample_rate = 1024;
    const float freq1 = 50.0F;
    const float freq2 = 120.0F;
    const int max_level = 4;

    auto signal = generateSignal(freq1, freq2, sample_rate, duration_seconds);

    // Wavelet initialization
    wavelets::init({"db8"});
    auto db8_filter = wavelets::get("db8");

    // Plotting
    plot_signal(signal);
    plot_dwt_levels(signal, db8_filter, max_level);
    plot_dwpt_levels(signal, db8_filter, max_level);

    try {
        std::cout << "Showing plots..." << std::endl;
        plt::show(true);
    } catch (const std::runtime_error& e) {
        std::cerr << "matplotlib-cpp error on show: " << e.what() << std::endl;
    }
    
    wavelets::resetInitialization();

    return 0;
}
