#include <fftw3.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

auto main() -> int
{
    // Generate a sample signal of 10 Hz and 40 Hz with sample rate of 1024 Hz
    const size_t duration_seconds = 4;
    const double sample_rate = 1024.0;
    const double freq1 = 10.0;
    const double freq2 = 40.0;
    const int signal_length = static_cast<int>(duration_seconds * sample_rate);

    auto* in = static_cast<double*>(fftw_malloc(sizeof(double) * signal_length));
    auto* out =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (signal_length / 2 + 1)));

    for (int n = 0; n < signal_length; n++)
    {
        in[n] = (0.7 * sin((2.0 * M_PI * freq1 * n) / sample_rate)) +
                (0.3 * sin((2.0 * M_PI * freq2 * n) / sample_rate));
    }

    fftw_plan p = fftw_plan_dft_r2c_1d(signal_length, in, out, FFTW_ESTIMATE);

    fftw_execute(p);

    std::vector<double> in_vec(signal_length);
    for (int i = 0; i < signal_length; i++)
    {
        in_vec[i] = in[i];
    }

    std::vector<double> fft_magnitude((signal_length / 2) + 1);
    for (int k = 0; k < (signal_length / 2) + 1; k++)
    {
        // Compute magnitude in dB
        // To avoid log(0), we can add a small epsilon if needed
        const double epsilon = 1e-100;
        const double real = out[k][0];
        const double imag = out[k][1];

        // Magnitude calculation:
        // magnitude = sqrt(real^2 + imag^2)
        // dB calculation:
        // dB = 20 * log10(magnitude)
        const double magnitude = sqrt((real * real) + (imag * imag)) + epsilon;
        fft_magnitude[k] = 20 * log10(magnitude);
    }

    // plt::plot(in_vec);
    // plt::title("Input Signal");
    // plt::show(false);

    plt::plot(fft_magnitude);
    plt::title("FFT Magnitude");
    plt::show(true); // Block until closed

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return 0;
}