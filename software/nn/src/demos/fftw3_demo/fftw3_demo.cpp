#include <fftw3.h>

#include <cmath>
#include <cstddef>
#include <vector> // Add this back for std::vector

#include "core/tensor/Tensor.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// Helper function to convert Eigen::MatrixXf to std::vector<float>
auto eigenMatrixToVector(const Eigen::MatrixXf& matrix) -> std::vector<float>
{
    std::vector<float> vec(matrix.data(), matrix.data() + matrix.size());
    return vec;
}

// Function to generate the sample signal
auto generateSignal(double freq1, double freq2, double sample_rate, size_t duration_seconds, int& signal_length_out) -> double*
{
    signal_length_out = static_cast<int>(duration_seconds * sample_rate);
    double* in = static_cast<double*>(fftw_malloc(sizeof(double) * signal_length_out));
    for (int n = 0; n < signal_length_out; n++)
    {
        in[n] = (0.7 * sin((2.0 * M_PI * freq1 * n) / sample_rate)) +
                (0.3 * sin((2.0 * M_PI * freq2 * n) / sample_rate));
    }
    return in;
}

// Function to execute FFT
auto executeFFT(int signal_length, double* in, fftw_complex*& out_fftw) -> fftw_plan
{
    out_fftw = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (signal_length / 2 + 1)));
    fftw_plan p = fftw_plan_dft_r2c_1d(signal_length, in, out_fftw, FFTW_ESTIMATE);
    fftw_execute(p);
    return p;
}

// Function to calculate FFT magnitude
auto calculateFFTMagnitude(int signal_length, fftw_complex* out_fftw) -> nn::Tensor
{
    nn::Tensor fft_magnitude(1, static_cast<Eigen::Index>((signal_length / 2) + 1));
    for (int k = 0; k < (signal_length / 2) + 1; k++)
    {
        const double epsilon = 1e-100;
        const double real = out_fftw[k][0];
        const double imag = out_fftw[k][1];
        const double magnitude = sqrt((real * real) + (imag * imag)) + epsilon;
        fft_magnitude.get_data_ref()(0, k) = static_cast<float>(20 * log10(magnitude));
    }
    return fft_magnitude;
}

// Function to plot the signal
void plotSignal(const nn::Tensor& signal_tensor, const std::string& title, bool show_blocking)
{
    plt::plot(eigenMatrixToVector(signal_tensor.get_data_ref()));
    plt::title(title);
    plt::show(show_blocking);
}

auto main() -> int
{
    // Signal Generation
    const size_t duration_seconds = 4;
    const double sample_rate = 1024.0;
    const double freq1 = 10.0;
    const double freq2 = 40.0;
    int signal_length;

    double* in = generateSignal(freq1, freq2, sample_rate, duration_seconds, signal_length);

    // FFT Execution
    fftw_complex* out_fftw;
    fftw_plan p = executeFFT(signal_length, in, out_fftw);

    // Magnitude Calculation
    nn::Tensor fft_magnitude = calculateFFTMagnitude(signal_length, out_fftw);

    // Prepare input signal for plotting (copy from double* to nn::Tensor)
    nn::Tensor in_vec(1, static_cast<Eigen::Index>(signal_length));
    for (int i = 0; i < signal_length; i++)
    {
        in_vec.get_data_ref()(0, i) = static_cast<float>(in[i]);
    }

    // Plotting
    plotSignal(in_vec, "Input Signal", false);
    plotSignal(fft_magnitude, "FFT Magnitude", true);

    // Cleanup
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out_fftw);
    return 0;
}