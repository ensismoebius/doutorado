#include <fftw3.h>

#include <cmath>
#include <cstddef>
#include <memory> // For std::unique_ptr
#include <vector>

#include "core/tensor/Tensor.hpp"
#include "matplotlibcpp.h"

using std::vector;

namespace plt = matplotlibcpp;

// Custom deleter for unique_ptr to manage FFTW allocated memory
struct FFTWFreeDeleter
{
    void operator()(void* ptr) const
    {
        fftw_free(ptr);
    }
};



// Function to generate the sample signal
auto generateSignal(float freq1, float freq2, size_t sample_rate, size_t duration_seconds,
                    int& signal_length_out) -> std::unique_ptr<double, FFTWFreeDeleter>
{
    signal_length_out = static_cast<int>(duration_seconds * sample_rate);

    auto* in_raw = static_cast<double*>(fftw_malloc(sizeof(double) * signal_length_out));

    for (int n = 0; n < signal_length_out; n++)
    {
        in_raw[n] = (0.7 * sinf((2.0F * M_PIf * freq1 * static_cast<float>(n)) /
                                static_cast<float>(sample_rate))) +
                    (0.3 * sinf((2.0F * M_PIf * freq2 * static_cast<float>(n)) /
                                static_cast<float>(sample_rate)));
    }
    return std::unique_ptr<double, FFTWFreeDeleter>(in_raw);
}

// Function to execute FFT
auto executeFFT(int signal_length, double* in_raw,
                std::unique_ptr<fftw_complex, FFTWFreeDeleter>& out_fftw_ptr) -> fftw_plan
{
    auto* out_raw = static_cast<fftw_complex*>(                     //
        fftw_malloc(sizeof(fftw_complex) * (signal_length / 2 + 1)) //
    );

    out_fftw_ptr.reset(out_raw);

    fftw_plan p = fftw_plan_dft_r2c_1d(signal_length, in_raw, out_raw, FFTW_ESTIMATE);

    fftw_execute(p);

    return p;
}

// Function to calculate FFT magnitude
auto calculateFFTMagnitude(int signal_length, const fftw_complex* out_fftw_raw) -> nn::Tensor
{
    nn::Tensor fft_magnitude(1, static_cast<Eigen::Index>((signal_length / 2) + 1));
    
    // Small epsilon to avoid log(0)
    const double epsilon = 1e-100;

    for (int k = 0; k < (signal_length / 2) + 1; k++)
    {
        const double real = out_fftw_raw[k][0];
        const double imag = out_fftw_raw[k][1];
        const double magnitude = sqrt((real * real) + (imag * imag)) + epsilon;
        
        fft_magnitude.get_data_ref()(0, k) = static_cast<float>(20 * log10(magnitude));
    }
    return fft_magnitude;
}

// Function to plot the signal
void plotSignal(const nn::Tensor& signal_tensor, const std::string& title, bool show_blocking)
{
    plt::plot(signal_tensor.toVector());
    plt::title(title);
    plt::show(show_blocking);
}

auto main() -> int
{
    // Signal Generation
    const size_t duration_seconds = 4;
    const size_t sample_rate = 1024;
    const float freq1 = 10.0;
    const float freq2 = 40.0;
    int signal_length;

    std::unique_ptr<double, FFTWFreeDeleter> in_ptr =
        generateSignal(freq1, freq2, sample_rate, duration_seconds, signal_length);

    // FFT Execution
    std::unique_ptr<fftw_complex, FFTWFreeDeleter> out_fftw_ptr;
    fftw_plan p = executeFFT(signal_length, in_ptr.get(), out_fftw_ptr);

    // Magnitude Calculation
    nn::Tensor fft_magnitude = calculateFFTMagnitude(signal_length, out_fftw_ptr.get());

    // Prepare input signal for plotting (copy from double* to nn::Tensor)
    nn::Tensor in_vec(1, static_cast<Eigen::Index>(signal_length));
    for (int i = 0; i < signal_length; i++)
    {
        in_vec.get_data_ref()(0, i) = static_cast<float>(in_ptr.get()[i]);
    }

    // Plotting
    plotSignal(in_vec, "Input Signal", false);
    plotSignal(fft_magnitude, "FFT Magnitude", true);

    // Cleanup
    fftw_destroy_plan(p); // fftw_free for in_ptr and out_fftw_ptr handled by unique_ptr
    return 0;
}