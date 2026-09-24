// bandpass_notch_filter_gtest.cpp — Unit tests for nn::utility::bandpass_notch.
//
// Deliberately NOT "does the output match this function's own formula" (that style of test
// caught nothing when wave/filter_operations.hpp's min-max normalization silently broke its
// sinc kernels -- see .wiki/Core/DataLoaders.md). These tests instead measure the actual
// frequency response: synthesize a signal from known sine tones, filter it, and check which
// tones survived and which didn't, via a single-frequency DFT projection (a Goertzel-style
// magnitude, not a full FFT).

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

#include "utility/BandpassNotchFilter.hpp"

namespace
{

auto make_tone(std::size_t n, double sampling_rate, double freq_hz, double amplitude = 1.0)
    -> std::vector<float>
{
    std::vector<float> s(n);
    const double w = 2.0 * std::numbers::pi * freq_hz / sampling_rate;
    for (std::size_t i = 0; i < n; ++i)
        s[i] = static_cast<float>(amplitude * std::sin(w * static_cast<double>(i)));
    return s;
}

auto add(std::vector<float> a, const std::vector<float>& b) -> std::vector<float>
{
    for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    return a;
}

// Single-frequency DFT magnitude (Goertzel-equivalent): how much of `signal` is explained by
// a sinusoid at exactly freq_hz.
auto tone_amplitude(const std::vector<float>& signal, double sampling_rate, double freq_hz)
    -> double
{
    double sum_cos = 0.0, sum_sin = 0.0;
    const double w = 2.0 * std::numbers::pi * freq_hz / sampling_rate;
    for (std::size_t n = 0; n < signal.size(); ++n)
    {
        sum_cos += static_cast<double>(signal[n]) * std::cos(w * static_cast<double>(n));
        sum_sin += static_cast<double>(signal[n]) * std::sin(w * static_cast<double>(n));
    }
    const double scale = 2.0 / static_cast<double>(signal.size());
    return scale * std::sqrt(sum_cos * sum_cos + sum_sin * sum_sin);
}

} // namespace

TEST(BandpassNotchFilter, ThrowsOnNonPositiveSamplingRate)
{
    EXPECT_THROW(nn::utility::bandpass_notch({1.0F, 2.0F, 3.0F}, 0.0, 0.5, 40.0, -1.0),
        std::invalid_argument);
}

TEST(BandpassNotchFilter, ThrowsWhenLowNotBelowHigh)
{
    EXPECT_THROW(nn::utility::bandpass_notch({1.0F, 2.0F, 3.0F}, 200.0, 40.0, 40.0, -1.0),
        std::invalid_argument);
}

TEST(BandpassNotchFilter, ThrowsWhenHighAtOrAboveNyquist)
{
    EXPECT_THROW(nn::utility::bandpass_notch({1.0F, 2.0F, 3.0F}, 200.0, 0.5, 100.0, -1.0),
        std::invalid_argument);
}

TEST(BandpassNotchFilter, ThrowsWhenNotchBandExceedsNyquist)
{
    EXPECT_THROW(
        nn::utility::bandpass_notch(
            {1.0F, 2.0F, 3.0F}, 100.0, 0.5, 40.0, /*notch_hz=*/49.0, /*notch_width_hz=*/4.0),
        std::invalid_argument);
}

TEST(BandpassNotchFilter, EmptySignalReturnsEmpty)
{
    const auto out = nn::utility::bandpass_notch({}, 200.0, 0.5, 40.0, -1.0);
    EXPECT_TRUE(out.empty());
}

TEST(BandpassNotchFilter, OutputLengthMatchesInput)
{
    const auto tone = make_tone(4000, 200.0, 10.0);
    const auto out = nn::utility::bandpass_notch(tone, 200.0, 0.5, 40.0, -1.0, 2.0, 5.0);
    EXPECT_EQ(out.size(), tone.size());
}

TEST(BandpassNotchFilter, InBandToneSurvivesNearUnityGain)
{
    // 10 Hz is well inside [0.5, 40] Hz -- the whole point of a *correctly normalized*
    // filter is that a passband tone comes out close to its original amplitude. Under the
    // wave/filter_operations.hpp min-max-normalized kernel this project already has, a
    // filter built the same way would NOT preserve amplitude this way.
    constexpr double kFs = 200.0, kFreq = 10.0, kAmp = 3.0;
    const auto tone = make_tone(8000, kFs, kFreq, kAmp);
    const auto out = nn::utility::bandpass_notch(tone, kFs, 0.5, 40.0, -1.0, 2.0, 5.0);

    const double before = tone_amplitude(tone, kFs, kFreq);
    const double after = tone_amplitude(out, kFs, kFreq);
    EXPECT_NEAR(before, kAmp, 0.05);
    EXPECT_NEAR(after / before, 1.0, 0.15) << "in-band tone should survive near unity gain";
}

TEST(BandpassNotchFilter, SubLowCutoffDriftIsAttenuated)
{
    // 0.05 Hz is far below the 0.5 Hz highpass edge -- simulates DC/slow electrode drift.
    constexpr double kFs = 200.0, kDrift = 0.05, kTone = 10.0;
    auto signal = add(make_tone(8000, kFs, kDrift, 5.0), make_tone(8000, kFs, kTone, 1.0));
    const auto out = nn::utility::bandpass_notch(signal, kFs, 0.5, 40.0, -1.0, 2.0, 5.0);

    const double drift_before = tone_amplitude(signal, kFs, kDrift);
    const double drift_after = tone_amplitude(out, kFs, kDrift);
    EXPECT_LT(drift_after / drift_before, 0.1) << "sub-0.5Hz drift should be strongly attenuated";
}

TEST(BandpassNotchFilter, AboveHighCutoffNoiseIsAttenuated)
{
    // 90 Hz is far above the 40 Hz lowpass edge -- simulates EMG/muscle artifact.
    constexpr double kFs = 200.0, kNoise = 90.0, kTone = 10.0;
    auto signal = add(make_tone(8000, kFs, kNoise, 4.0), make_tone(8000, kFs, kTone, 1.0));
    const auto out = nn::utility::bandpass_notch(signal, kFs, 0.5, 40.0, -1.0, 2.0, 5.0);

    const double noise_before = tone_amplitude(signal, kFs, kNoise);
    const double noise_after = tone_amplitude(out, kFs, kNoise);
    EXPECT_LT(noise_after / noise_before, 0.1) << "above-40Hz noise should be strongly attenuated";
}

TEST(BandpassNotchFilter, MainsNotchAttenuatesOnlyItsOwnBand)
{
    // Wide bandpass (0.5-80 Hz) so both the 20 Hz signal tone and the 50 Hz mains tone are
    // inside it -- isolates the notch's effect from the bandpass edges.
    constexpr double kFs = 200.0, kMains = 50.0, kTone = 20.0;
    auto signal = add(make_tone(8000, kFs, kMains, 4.0), make_tone(8000, kFs, kTone, 1.0));
    const auto out =
        nn::utility::bandpass_notch(signal, kFs, 0.5, 80.0, /*notch_hz=*/kMains, 2.0, 5.0);

    const double mains_before = tone_amplitude(signal, kFs, kMains);
    const double mains_after = tone_amplitude(out, kFs, kMains);
    const double tone_before = tone_amplitude(signal, kFs, kTone);
    const double tone_after = tone_amplitude(out, kFs, kTone);

    EXPECT_LT(mains_after / mains_before, 0.1) << "50Hz mains tone should be notched out";
    EXPECT_NEAR(tone_after / tone_before, 1.0, 0.15)
        << "20Hz signal tone should survive the notch untouched";
}

TEST(BandpassNotchFilter, NotchDisabledLeavesMainsToneUnattenuated)
{
    // Same setup as above but notch_hz <= 0 (disabled): the 50Hz tone must survive, proving
    // the notch is opt-in, not an always-on side effect of the bandpass alone.
    constexpr double kFs = 200.0, kMains = 50.0;
    const auto signal = make_tone(8000, kFs, kMains, 4.0);
    const auto out =
        nn::utility::bandpass_notch(signal, kFs, 0.5, 80.0, /*notch_hz=*/-1.0, 2.0, 5.0);

    const double before = tone_amplitude(signal, kFs, kMains);
    const double after = tone_amplitude(out, kFs, kMains);
    EXPECT_NEAR(after / before, 1.0, 0.15);
}
