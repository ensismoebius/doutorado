#include <fftw3.h>
#include <math.h>
#include <stdio.h>

int main()
{
    int N = 8;
    auto in = static_cast<double*>(fftw_malloc(sizeof(double) * N));
    auto out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1)));

    // Example signal
    for (int i = 0; i < N; i++)
    {
        in[i] = sin(2 * M_PI * i / N);
    }

    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    fftw_execute(p);

    for (int k = 0; k < N / 2 + 1; k++) printf("%d: %f + %fi\n", k, out[k][0], out[k][1]);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return 0;
}
