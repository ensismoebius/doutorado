/**
 * @file wavelet_demo.cpp
 * @brief Wavelet transform demo (Mallat) with matplotlib visualization.
 *
 * Generates a synthetic signal, computes a discrete wavelet decomposition,
 * and plots approximation/detail coefficients to help validate the wavelet code.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "matplotlibcpp.h"
#include "wavelet/Types.hpp"
#include "wavelet/WaveletTransformResults.hpp"
#include "wavelet/waveletOperations.hpp"

namespace plt = matplotlibcpp;

/**
 * @brief Generate a composite signal with two sine waves
 */
auto generateSignal(double freq1, double freq2, size_t sample_rate, size_t duration_seconds)
    -> std::vector<double>
{
    int signal_length = static_cast<int>(duration_seconds * sample_rate);
    std::vector<double> signal(signal_length);

    for (int n = 0; n < signal_length; n++)
    {
        signal[n] = (0.7 * sin((2.0 * M_PI * freq1 * static_cast<double>(n)) /
                               static_cast<double>(sample_rate))) +
                    (0.3 * sin((2.0 * M_PI * freq2 * static_cast<double>(n)) /
                               static_cast<double>(sample_rate)));
    }
    return signal;
}

void plot_signal(const std::vector<double>& signal)
{
    try
    {
        plt::plot(signal);
        plt::title("Input Signal");
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "matplotlib-cpp error in plot_signal: " << e.what() << '\n';
    }
}

void plot_dwt_decomposition_single_plot(
    const std::vector<double>& signal, std::span<const double> filter, int level_to_plot)
{
    try
    {
        auto dwt_results =
            wavelets::malat(signal, filter, wavelets::REGULAR_WAVELET, level_to_plot);

        std::vector<double> combined_coefficients;
        double offset = 0.0; // Initial offset for visual separation

        // Get approximation coefficients
        auto approximation = dwt_results.get_wavelet_transforms(0);
        if (!approximation.empty())
        {
            std::transform(approximation.begin(),
                approximation.end(),
                std::back_inserter(combined_coefficients),
                [offset](double val) { return val + offset; });
            if (approximation.size() > 1)
            { // Only calculate range if more than one element
                offset +=
                    (*std::ranges::max_element(approximation.begin(), approximation.end()) -
                        *std::ranges::min_element(approximation.begin(), approximation.end())) +
                    0.1; // dynamic offset
            }
        }

        // Get detail coefficients for each level
        for (int i = 1; i <= level_to_plot; ++i)
        {
            auto details = dwt_results.get_wavelet_transforms(i);
            if (!details.empty())
            {
                std::transform(details.begin(),
                    details.end(),
                    std::back_inserter(combined_coefficients),
                    [offset](double val) { return val + offset; });
                if (details.size() > 1)
                {
                    offset += (*std::ranges::max_element(details.begin(), details.end()) -
                                  *std::ranges::min_element(details.begin(), details.end())) +
                              0.1; // dynamic offset
                }
            }
        }

        plt::plot(combined_coefficients);
        plt::title("DWT Decomposition (Level " + std::to_string(level_to_plot) + ")");
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "matplotlib-cpp error in plot_dwt_decomposition_single_plot (level "
                  << level_to_plot << "): " << e.what() << '\n';
    }
}

void plot_dwpt_decomposition_single_plot(
    const std::vector<double>& signal, std::span<const double> filter, int level_to_plot)
{
    try
    {
        auto dwpt_results =
            wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, level_to_plot);
        long n_parts = dwpt_results.get_wavelet_packet_amount_of_parts();

        std::vector<double> combined_packets;
        double offset = 0.0; // Initial offset for visual separation

        for (long i = 0; i < n_parts; ++i)
        {
            auto part = wavelets::WaveletTransformResults::get_wavelet_packet_transforms(
                dwpt_results.transformedSignal, i, dwpt_results.levelsOfTransformation);
            if (!part.empty())
            {
                std::transform(part.begin(),
                    part.end(),
                    std::back_inserter(combined_packets),
                    [offset](double val) { return val + offset; });
                if (part.size() > 1)
                { // Only calculate range if more than one element
                    offset += (*std::ranges::max_element(part.begin(), part.end()) -
                                  *std::ranges::min_element(part.begin(), part.end())) +
                              0.1; // dynamic offset
                }
            }
        }

        plt::plot(combined_packets);
        plt::title("DWPT Decomposition (Level " + std::to_string(level_to_plot) + ")");
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "matplotlib-cpp error in plot_dwpt_decomposition_single_plot (level "
                  << level_to_plot << "): " << e.what() << '\n';
    }
}

auto main() -> int
{
#ifndef NDEBWUG
    // Use the Agg backend for non-interactive plotting when debugging
    // plt::backend("Agg");
#endif

    // Signal Generation
    const size_t duration_seconds = 1;
    const size_t sample_rate = 1024;
    const float freq1 = 50.0F;
    const float freq2 = 120.0F;
    const int max_level = 4;

    auto signal = generateSignal(freq1, freq2, sample_rate, duration_seconds);

    // Wavelet initialization
    constexpr auto db8_filter = wavelets::get_wavelet<wavelets::Daub8>();

    // Debug mode: save plots to files
    // Plot original signal
    plt::figure();
    plot_signal(signal);
    std::cout << "Saving signal plot to wavelet_demo_signal.png" << '\n';
    plt::save("wavelet_demo_signal.png");
    plt::close();

    // Plotting loop for each level
    for (int level = 1; level <= max_level; ++level)
    {
        try
        {
            // DWT
            plt::figure();
            plot_dwt_decomposition_single_plot(signal, db8_filter, level);
            std::string dwt_filename = "wavelet_demo_dwt_" + std::to_string(level) + ".png";
            std::cout << "Saving DWT plot to " << dwt_filename << '\n';
            plt::save(dwt_filename);
            plt::close();

            // DWPT
            plt::figure();
            plot_dwpt_decomposition_single_plot(signal, db8_filter, level);
            std::string dwpt_filename = "wavelet_demo_dwpt_" + std::to_string(level) + ".png";
            std::cout << "Saving DWPT plot to " << dwpt_filename << '\n';
            plt::save(dwpt_filename);
            plt::close();
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << "matplotlib-cpp error in main loop (level " << level << "): " << e.what()
                      << '\n';
        }
    }
    // Release mode: show plots interactively
    // Plot original signal
    plt::figure();
    plot_signal(signal);

    // Plotting loop for each level
    for (int level = 1; level <= max_level; ++level)
    {
        try
        {
            // DWT
            plt::figure();
            plot_dwt_decomposition_single_plot(signal, db8_filter, level);

            // DWPT
            plt::figure();
            plot_dwpt_decomposition_single_plot(signal, db8_filter, level);
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << "matplotlib-cpp error in main loop (level " << level << "): " << e.what()
                      << '\n';
        }
    }

    try
    {
        std::cout << "Showing plots..." << '\n';
        plt::show(true);
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "matplotlib-cpp error on show: " << e.what() << '\n';
    }

    return 0;
}