/**
 * @file fftw3_demo.cpp
 * @brief Small FFTW3 + matplotlib-cpp demo.
 *
 * Generates a synthetic signal, computes an FFT (r2c), converts magnitudes to dB,
 * and plots the result. This is primarily a dependency/sanity check for FFTW.
 */

#include <fftw3.h>

#include <cmath>
#include <cstddef>
#include <memory> // For std::unique_ptr
#include <vector>

#include "matplotlibcpp.h"
#include "tensor/Tensor.hpp"

using std::vector;

namespace plt = matplotlibcpp;

/**
 * Custom deleter for unique_ptr to manage FFTW allocated memory
 */
struct FFTWFreeDeleter
{
    void operator()(void* ptr) const
    {
        fftw_free(ptr);
    }
};

/**
 * @brief Generate a composite signal with two sine waves
 *
 * @param freq1
 * @param freq2
 * @param sample_rate
 * @param duration_seconds
 * @param signal_length_out
 * @return std::unique_ptr<float, FFTWFreeDeleter>
 */
auto generateSignal(
    double freq1, double freq2, size_t sample_rate, size_t duration_seconds, int& signal_length_out)
    -> std::unique_ptr<double, FFTWFreeDeleter>
{
    signal_length_out = static_cast<int>(duration_seconds * sample_rate);
    auto* in_raw = static_cast<double*>(fftw_malloc(sizeof(double) * signal_length_out));

    for (int n = 0; n < signal_length_out; n++)
    {
        in_raw[n] = (0.7 * sin((2.0 * M_PI * freq1 * static_cast<double>(n)) /
                               static_cast<double>(sample_rate))) +
                    (0.3 * sin((2.0 * M_PI * freq2 * static_cast<double>(n)) /
                               static_cast<double>(sample_rate)));
    }
    return std::unique_ptr<double, FFTWFreeDeleter>(in_raw);
}

/**
 * @brief Execute FFT using FFTW3
 *
 * @param signal_length
 * @param in_raw
 * @param out_fftw_ptr
 * @return fftwf_plan
 */
auto executeFFT(                                                 //
    int signal_length,                                           //
    double* in_raw,                                              //
    std::unique_ptr<fftw_complex, FFTWFreeDeleter>& out_fftw_ptr //
    ) -> fftw_plan
{
    auto* out_raw =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (signal_length / 2 + 1)));

    out_fftw_ptr.reset(out_raw);

    fftw_plan p = fftw_plan_dft_r2c_1d(signal_length, in_raw, out_raw, FFTW_ESTIMATE);

    fftw_execute(p);

    return p;
}

/**
 * @brief Calculate the magnitude of the FFT output
 *
 * @param signal_length
 * @param out_fftw_raw
 * @return nn::Tensor
 */
auto calculateFFTMagnitude(int signal_length, const fftw_complex* out_fftw_raw) -> nn::Tensor
{
    // Tensor to hold magnitude values
    nn::Tensor fft_magnitude(1, static_cast<size_t>((signal_length / 2) + 1));

    // Small epsilon to avoid log(0)
    const double epsilon = 1e-300;

    for (int k = 0; k < (signal_length / 2) + 1; k++)
    {
        const double real = out_fftw_raw[k][0];
        const double imag = out_fftw_raw[k][1];

        // Calculate magnitude and convert to dB scale
        const double magnitude = sqrt((real * real) + (imag * imag)) + epsilon;

        // fft_magnitude stores floats; cast from double
        fft_magnitude(0, k) = static_cast<float>(20.0 * log10(magnitude));
    }
    return fft_magnitude;
}

/**
 * @brief Plot the signal using matplotlibcpp
 *
 * @param signal_tensor
 * @param title
 * @param show_blocking
 */
void plotSignal(const nn::Tensor& signal_tensor, const std::string& title, bool show_blocking)
{
    plt::plot(signal_tensor.toVector<float>());
    plt::title(title);
    plt::show(show_blocking);
}

auto main() -> int
{
    // Signal Generation
    const size_t duration_seconds = 4;
    const size_t sample_rate = 1024;
    const float freq1 = 10.0F;
    const float freq2 = 40.0F;
    int signal_length;

    auto in_ptr = generateSignal( //
        freq1,                    //
        freq2,                    //
        sample_rate,              //
        duration_seconds,         //
        signal_length             //
    );

    // FFT Execution
    std::unique_ptr<fftw_complex, FFTWFreeDeleter> out_fftw_ptr;

    auto* p = executeFFT( //
        signal_length,    //
        in_ptr.get(),     //
        out_fftw_ptr      //
    );

    // Magnitude Calculation
    auto fft_magnitude = calculateFFTMagnitude( //
        signal_length,                          //
        out_fftw_ptr.get()                      //
    );

    nn::Tensor in_vec(1, static_cast<size_t>(signal_length));
    for (int i = 0; i < signal_length; i++)
    {
        in_vec(0, i) = static_cast<float>(in_ptr.get()[i]);
    }

    // Plotting
    plotSignal(in_vec, "Input Signal", false);
    plotSignal(fft_magnitude, "FFT Magnitude", true);

    // Cleanup
    fftw_destroy_plan(p);
    return 0;
}