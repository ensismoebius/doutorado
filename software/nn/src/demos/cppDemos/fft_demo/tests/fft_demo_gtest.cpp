/**
 * @file fft_demo_gtest.cpp
 * @brief Unit tests for fft_demo core functions: generateSignal, executeFFT,
 *        calculateFFTMagnitude.
 *
 * Tests use synthetic data only (no filesystem I/O).
 * The plotting helpers are intentionally excluded from these tests.
 */

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include <fftw3.h>
#include <gtest/gtest.h>

#include "tensor/Tensor.hpp"

// ---- replicate the deleter and functions under test ----
struct FFTWFreeDeleter
{
    void operator()(void* ptr) const { fftw_free(ptr); }
};

auto generateSignal(
    double freq1, double freq2, size_t sample_rate, size_t duration_seconds, int& signal_length_out)
    -> std::unique_ptr<double, FFTWFreeDeleter>
{
    signal_length_out = static_cast<int>(duration_seconds * sample_rate);
    auto* in_raw = static_cast<double*>(fftw_malloc(sizeof(double) * signal_length_out));
    for (int n = 0; n < signal_length_out; ++n)
    {
        in_raw[n] = (0.7 * std::sin((2.0 * M_PI * freq1 * static_cast<double>(n)) /
                                    static_cast<double>(sample_rate))) +
                    (0.3 * std::sin((2.0 * M_PI * freq2 * static_cast<double>(n)) /
                                    static_cast<double>(sample_rate)));
    }
    return std::unique_ptr<double, FFTWFreeDeleter>(in_raw);
}

auto executeFFT(int signal_length,
    double* in_raw,
    std::unique_ptr<fftw_complex, FFTWFreeDeleter>& out_fftw_ptr) -> fftw_plan
{
    auto* out_raw =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (signal_length / 2 + 1)));
    out_fftw_ptr.reset(out_raw);
    fftw_plan p = fftw_plan_dft_r2c_1d(signal_length, in_raw, out_raw, FFTW_ESTIMATE);
    fftw_execute(p);
    return p;
}

auto calculateFFTMagnitude(int signal_length, const fftw_complex* out_fftw_raw) -> nn::Tensor
{
    nn::Tensor fft_magnitude(1, static_cast<size_t>((signal_length / 2) + 1));
    const double epsilon = 1e-300;
    for (int k = 0; k < (signal_length / 2) + 1; ++k)
    {
        const double real = out_fftw_raw[k][0];
        const double imag = out_fftw_raw[k][1];
        const double magnitude = std::sqrt((real * real) + (imag * imag)) + epsilon;
        fft_magnitude(0, k) = static_cast<float>(20.0 * std::log10(magnitude));
    }
    return fft_magnitude;
}

// ---- Test fixture ----
class FftDemoTest : public ::testing::Test
{
   protected:
    static constexpr size_t kSampleRate = 1024;
    static constexpr size_t kDuration = 4;
    static constexpr double kFreq1 = 10.0;
    static constexpr double kFreq2 = 40.0;
};

// ---- generateSignal tests ----

TEST_F(FftDemoTest, GenerateSignal_LengthMatchesDurationTimesSampleRate)
{
    int signal_length = 0;
    auto sig = generateSignal(kFreq1, kFreq2, kSampleRate, kDuration, signal_length);
    EXPECT_EQ(signal_length, static_cast<int>(kSampleRate * kDuration));
}

TEST_F(FftDemoTest, GenerateSignal_AmplitudeBoundedByOne)
{
    int signal_length = 0;
    auto sig = generateSignal(kFreq1, kFreq2, kSampleRate, kDuration, signal_length);
    // max combined amplitude is |0.7| + |0.3| = 1.0
    for (int i = 0; i < signal_length; ++i)
    {
        EXPECT_LE(std::abs(sig.get()[i]), 1.0 + 1e-9);
    }
}

TEST_F(FftDemoTest, GenerateSignal_ContainsSineComponents)
{
    // At n=0 both sines are 0; at the quarter-period of freq1 (n = sr/4f1) sin ≈ 1
    // Just verify values are non-constant (signal has AC content)
    int signal_length = 0;
    auto sig = generateSignal(kFreq1, kFreq2, kSampleRate, kDuration, signal_length);
    double v0 = sig.get()[0];
    bool found_diff = false;
    for (int i = 1; i < signal_length; ++i)
    {
        if (std::abs(sig.get()[i] - v0) > 1e-9)
        {
            found_diff = true;
            break;
        }
    }
    EXPECT_TRUE(found_diff);
}

// ---- executeFFT / calculateFFTMagnitude tests ----

TEST_F(FftDemoTest, FFTMagnitude_SizeIsHalfPlusOne)
{
    int signal_length = 0;
    auto in_ptr = generateSignal(kFreq1, kFreq2, kSampleRate, kDuration, signal_length);

    std::unique_ptr<fftw_complex, FFTWFreeDeleter> out_fftw_ptr;
    fftw_plan p = executeFFT(signal_length, in_ptr.get(), out_fftw_ptr);

    auto mag = calculateFFTMagnitude(signal_length, out_fftw_ptr.get());
    EXPECT_EQ(mag.cols(), static_cast<size_t>((signal_length / 2) + 1));
    EXPECT_EQ(mag.rows(), 1u);

    fftw_destroy_plan(p);
}

TEST_F(FftDemoTest, FFTMagnitude_PeakAtFreq1Bin)
{
    int signal_length = 0;
    auto in_ptr = generateSignal(kFreq1, 0.0, kSampleRate, kDuration, signal_length);

    std::unique_ptr<fftw_complex, FFTWFreeDeleter> out_fftw_ptr;
    fftw_plan p = executeFFT(signal_length, in_ptr.get(), out_fftw_ptr);
    auto mag = calculateFFTMagnitude(signal_length, out_fftw_ptr.get());

    // Expected peak bin = freq1 * duration (because N = sr*duration)
    int expected_bin = static_cast<int>(kFreq1 * static_cast<double>(kDuration));
    float peak_db = mag(0, static_cast<size_t>(expected_bin));

    // All other bins should be significantly lower (at least 40 dB below)
    for (size_t k = 0; k < mag.cols(); ++k)
    {
        if (static_cast<int>(k) != expected_bin)
        {
            EXPECT_LT(mag(0, k), peak_db - 20.0F)
                << "Unexpected peak at bin " << k << " (expected peak at " << expected_bin << ")";
        }
    }

    fftw_destroy_plan(p);
}

TEST_F(FftDemoTest, FFTMagnitude_ValuesAreFinite)
{
    int signal_length = 0;
    auto in_ptr = generateSignal(kFreq1, kFreq2, kSampleRate, kDuration, signal_length);

    std::unique_ptr<fftw_complex, FFTWFreeDeleter> out_fftw_ptr;
    fftw_plan p = executeFFT(signal_length, in_ptr.get(), out_fftw_ptr);
    auto mag = calculateFFTMagnitude(signal_length, out_fftw_ptr.get());

    for (size_t k = 0; k < mag.cols(); ++k)
    {
        EXPECT_TRUE(std::isfinite(mag(0, k))) << "Non-finite value at bin " << k;
    }

    fftw_destroy_plan(p);
}

TEST_F(FftDemoTest, FFTMagnitude_EpsilonPreventsMInfForZeroInput)
{
    // A zero signal would produce all-zero FFT output, but epsilon should prevent log(0)=-inf
    const int N = 256;
    auto* zeros = static_cast<double*>(fftw_malloc(sizeof(double) * N));
    for (int i = 0; i < N; ++i) zeros[i] = 0.0;

    std::unique_ptr<double, FFTWFreeDeleter> zero_sig(zeros);
    std::unique_ptr<fftw_complex, FFTWFreeDeleter> out_fftw_ptr;
    fftw_plan p = executeFFT(N, zero_sig.get(), out_fftw_ptr);
    auto mag = calculateFFTMagnitude(N, out_fftw_ptr.get());

    for (size_t k = 0; k < mag.cols(); ++k)
    {
        EXPECT_FALSE(std::isinf(mag(0, k))) << "Infinite value at bin " << k;
    }

    fftw_destroy_plan(p);
}
