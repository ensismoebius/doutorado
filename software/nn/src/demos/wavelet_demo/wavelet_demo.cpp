#include <cmath>
#include <cstddef>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm> // Required for std::max_element and std::min_element

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
        plt::plot(signal);
        plt::title("Input Signal");
    } catch (const std::runtime_error& e) {
        std::cerr << "matplotlib-cpp error in plot_signal: " << e.what() << std::endl;
    }
}

void plot_dwt_decomposition_single_plot(const std::vector<double>& signal,
                                        const std::vector<double>& filter,
                                        int level_to_plot,
                                        int num_rows,
                                        int num_cols,
                                        int plot_idx) {
    try {
        auto dwt_results = wavelets::malat(signal, const_cast<std::vector<double>&>(filter), wavelets::REGULAR_WAVELET, level_to_plot);
        
        std::vector<double> combined_coefficients;
        double offset = 0.0; // Initial offset for visual separation

        // Get approximation coefficients
        auto approximation = dwt_results.getWaveletTransforms(0);
        if (!approximation.empty()) {
            for(double val : approximation) {
                combined_coefficients.push_back(val + offset);
            }
            if(approximation.size() > 1) { // Only calculate range if more than one element
                 offset += (*std::max_element(approximation.begin(), approximation.end()) - *std::min_element(approximation.begin(), approximation.end())) + 0.1; // dynamic offset
            }
           
        }

        // Get detail coefficients for each level
        for (int i = 1; i <= level_to_plot; ++i) {
            auto details = dwt_results.getWaveletTransforms(i);
            if (!details.empty()) {
                for(double val : details) {
                    combined_coefficients.push_back(val + offset);
                }
                if(details.size() > 1) { // Only calculate range if more than one element
                    offset += (*std::max_element(details.begin(), details.end()) - *std::min_element(details.begin(), details.end())) + 0.1; // dynamic offset
                }
            }
        }

        plt::subplot(num_rows, num_cols, plot_idx);
        plt::plot(combined_coefficients);
        plt::title("DWT Decomposition (Level " + std::to_string(level_to_plot) + ")");

    } catch (const std::runtime_error& e) {
        std::cerr << "matplotlib-cpp error in plot_dwt_decomposition_single_plot (level " << level_to_plot << "): " << e.what() << std::endl;
    }
}

void plot_dwpt_decomposition_single_plot(const std::vector<double>& signal,
                                         const std::vector<double>& filter,
                                         int level_to_plot,
                                         int num_rows,
                                         int num_cols,
                                         int plot_idx) {
    try {
        auto dwpt_results = wavelets::malat(signal, const_cast<std::vector<double>&>(filter), wavelets::PACKET_WAVELET, level_to_plot);
        long n_parts = dwpt_results.getWaveletPacketAmountOfParts();

        std::vector<double> combined_packets;
        double offset = 0.0; // Initial offset for visual separation

        for (long i = 0; i < n_parts; ++i) {
            auto part = wavelets::WaveletTransformResults::getWaveletPacketTransforms(dwpt_results.transformedSignal, i, dwpt_results.levelsOfTransformation);
            if (!part.empty()) {
                for(double val : part) {
                    combined_packets.push_back(val + offset);
                }
                if(part.size() > 1) { // Only calculate range if more than one element
                    offset += (*std::max_element(part.begin(), part.end()) - *std::min_element(part.begin(), part.end())) + 0.1; // dynamic offset
                }
            }
        }

        plt::subplot(num_rows, num_cols, plot_idx);
        plt::plot(combined_packets);
        plt::title("DWPT Decomposition (Level " + std::to_string(level_to_plot) + ")");

    } catch (const std::runtime_error& e) {
        std::cerr << "matplotlib-cpp error in plot_dwpt_decomposition_single_plot (level " << level_to_plot << "): " << e.what() << std::endl;
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

    // Plotting loop for each level
    for (int level = 1; level <= max_level; ++level) {
        try {
            plt::figure();
            plt::suptitle("Wavelet Decomposition - Level " + std::to_string(level));

            // Plot 1: Original Signal
            plt::subplot(3, 1, 1);
            plot_signal(signal);

            // Plot 2: DWT Decomposition
            plot_dwt_decomposition_single_plot(signal, db8_filter, level, 3, 1, 2);

            // Plot 3: DWPT Decomposition
            plot_dwpt_decomposition_single_plot(signal, db8_filter, level, 3, 1, 3);

        } catch (const std::runtime_error& e) {
            std::cerr << "matplotlib-cpp error in main loop (level " << level << "): " << e.what() << std::endl;
        }
    }

    try {
        std::cout << "Showing plots..." << std::endl;
        plt::show(true);
    } catch (const std::runtime_error& e) {
        std::cerr << "matplotlib-cpp error on show: " << e.what() << std::endl;
    }
    
    wavelets::resetInitialization();

    return 0;
}