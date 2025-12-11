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
        fftwf_free(ptr);
    }
};



// Function to generate the sample signal
auto generateSignal(float freq1, float freq2, size_t sample_rate, size_t duration_seconds,
                    int& signal_length_out) -> std::unique_ptr<float, FFTWFreeDeleter>
{
    signal_length_out = static_cast<int>(duration_seconds * sample_rate);

    auto* in_raw = static_cast<float*>(fftwf_malloc(sizeof(float) * signal_length_out));

    for (int n = 0; n < signal_length_out; n++)
    {
        in_raw[n] = (0.7F * sinf((2.0F * M_PIf * freq1 * static_cast<float>(n)) /
                                static_cast<float>(sample_rate))) +
                    (0.3F * sinf((2.0F * M_PIf * freq2 * static_cast<float>(n)) /
                                static_cast<float>(sample_rate)));
    }
    return std::unique_ptr<float, FFTWFreeDeleter>(in_raw);
}

// Function to execute FFT
auto executeFFT(int signal_length, float* in_raw, // Changed double* to float*
                std::unique_ptr<fftwf_complex, FFTWFreeDeleter>& out_fftw_ptr) -> fftwf_plan // Changed fftw_complex to fftwf_complex, fftw_plan to fftwf_plan
{
    auto* out_raw = static_cast<fftwf_complex*>(                     // Changed fftw_complex to fftwf_complex
        fftwf_malloc(sizeof(fftwf_complex) * (signal_length / 2 + 1)) // Changed fftw_malloc to fftwf_malloc
    );

    out_fftw_ptr.reset(out_raw);

    fftwf_plan p = fftwf_plan_dft_r2c_1d(signal_length, in_raw, out_raw, FFTW_ESTIMATE); // Changed fftw_plan_dft_r2c_1d to fftwf_plan_dft_r2c_1d

    fftwf_execute(p); // Changed fftw_execute to fftwf_execute

    return p;
}

// Function to calculate FFT magnitude
auto calculateFFTMagnitude(int signal_length, const fftwf_complex* out_fftw_raw) -> nn::Tensor // Changed fftw_complex to fftwf_complex
{
    nn::Tensor fft_magnitude(1, static_cast<Eigen::Index>((signal_length / 2) + 1));
    
    // Small epsilon to avoid log(0)
    const float epsilon = 1e-100F;

    for (int k = 0; k < (signal_length / 2) + 1; k++)
    {
        const float real = out_fftw_raw[k][0]; // Changed double to float
        const float imag = out_fftw_raw[k][1]; // Changed double to float
        const float magnitude = sqrtf((real * real) + (imag * imag)) + epsilon; // Changed sqrt to sqrtf
        
        fft_magnitude.get_data_ref()(0, k) = static_cast<float>(20 * log10f(magnitude)); // Changed log10 to log10f
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
    const float freq1 = 10.0F;
    const float freq2 = 40.0F;
    int signal_length;

    std::unique_ptr<float, FFTWFreeDeleter> in_ptr = // Changed double to float
        generateSignal(freq1, freq2, sample_rate, duration_seconds, signal_length);

    // FFT Execution
    std::unique_ptr<fftwf_complex, FFTWFreeDeleter> out_fftw_ptr; // Changed fftw_complex to fftwf_complex
    fftwf_plan p = executeFFT(signal_length, in_ptr.get(), out_fftw_ptr); // Changed fftw_plan to fftwf_plan

    // Magnitude Calculation
    nn::Tensor fft_magnitude = calculateFFTMagnitude(signal_length, out_fftw_ptr.get());

    // Prepare input signal for plotting (copy from float* to nn::Tensor)
    nn::Tensor in_vec(1, static_cast<Eigen::Index>(signal_length));
    for (int i = 0; i < signal_length; i++)
    {
        in_vec.get_data_ref()(0, i) = in_ptr.get()[i]; // Removed static_cast<float> as in_ptr.get() returns float*
    }

    // Plotting
    plotSignal(in_vec, "Input Signal", false);
    plotSignal(fft_magnitude, "FFT Magnitude", true);

    // Cleanup
    fftwf_destroy_plan(p); // Changed fftw_destroy_plan to fftwf_destroy_plan
    return 0;
}