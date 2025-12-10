#include <fftw3.h>

#include <cmath>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

auto main() -> int
{
    int N = 2048;
    auto* in = static_cast<double*>(fftw_malloc(sizeof(double) * N));
    auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1)));

    // Example signal
    for (int i = 0; i < N; i++)
    {
        in[i] = sin(2 * M_PI * i / N);
    }

    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    fftw_execute(p);

    std::vector<double> in_vec(N);
    for (int i = 0; i < N; i++)
    {
        in_vec[i] = in[i];
    }
    plt::plot(in_vec);
    plt::title("Input Signal");
    plt::show(false); // Don't block, allow multiple plots

    std::vector<double> fft_magnitude((N / 2) + 1);
    for (int k = 0; k < (N / 2) + 1; k++)
    {
        // printf("%d: %f + %fi\n", k, out[k][0], out[k][1]);
        fft_magnitude[k] = sqrt((out[k][0] * out[k][0]) + (out[k][1] * out[k][1]));
    }
    plt::plot(fft_magnitude);
    plt::title("FFT Magnitude");
    plt::show(true); // Block until closed

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return 0;
}