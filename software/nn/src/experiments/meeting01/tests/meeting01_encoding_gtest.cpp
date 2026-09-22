// meeting01_encoding_gtest.cpp — coverage for `encode_sample`'s time-major contract.
//
// These tests were rewritten on 2026-09-22. The previous LatencyEncodesExactlyOneSpike
// test held each channel's value CONSTANT across time (`sample.at(t, d) = values[d]`),
// which is the one input distribution under which the old implementation looked
// correct. With a real, time-varying window the old code emitted ~1 spike per 256
// samples and often none at all, and no test noticed. The tests below therefore use
// varying signals on purpose, and assert on the shape contract as well as the counts.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "Meeting01Encoding.hpp"

namespace
{

constexpr int kSteps = 16;

// A realistic window: 256 consecutive samples of a varying, z-scored signal.
meeting01::Tensor make_window(int n = 256)
{
    meeting01::Tensor w(n, 1);
    double sum = 0.0;
    double sq = 0.0;
    for (nn::Index i = 0; i < n; ++i)
    {
        const float v = std::sin(0.11f * static_cast<float>(i)) +
                        0.4f * std::sin(0.73f * static_cast<float>(i) + 1.1f);
        w.at(i, 0) = v;
        sum += v;
        sq += static_cast<double>(v) * v;
    }
    const double mean = sum / n;
    const double sd = std::sqrt(std::max(1e-12, sq / n - mean * mean));
    for (nn::Index i = 0; i < n; ++i) w.at(i, 0) = static_cast<float>((w.at(i, 0) - mean) / sd);
    return w;
}

int count_spikes(const meeting01::Tensor& t)
{
    int n = 0;
    for (nn::Index i = 0; i < t.size(); ++i) n += (t.at(i) > 0.5f) ? 1 : 0;
    return n;
}

TEST(Meeting01Encoding, EncodesToTimeMajorStepsByFeatures)
{
    const auto window = make_window(256);
    for (const char* enc : {"direct", "poisson", "latency"})
    {
        const auto e = meeting01::encode_sample(window, enc, /*seed=*/7, kSteps);
        EXPECT_EQ(e.rows(), kSteps) << enc << ": rows must be the simulation steps";
        EXPECT_EQ(e.cols(), 256) << enc << ": cols must be the window's samples";
    }
}

TEST(Meeting01Encoding, RejectsSingleTimeStep)
{
    const auto window = make_window(32);
    // Must fail loudly: at T=1 rate and latency coding have no axis to encode on, and
    // the old behaviour was to silently produce a plausible-looking empty tensor.
    EXPECT_THROW((void) meeting01::encode_sample(window, "latency", 0, 1), std::invalid_argument);
    EXPECT_THROW((void) meeting01::encode_sample(window, "poisson", 0, 0), std::invalid_argument);
}

TEST(Meeting01Encoding, LatencyFiresEachFeatureExactlyOnceOnAVaryingSignal)
{
    const auto window = make_window(256);
    const auto e = meeting01::encode_sample(window, "latency", /*seed=*/0, kSteps);

    // Canonical TTFS: every input neuron (window sample) fires exactly once.
    for (nn::Index f = 0; f < e.cols(); ++f)
    {
        int spikes = 0;
        for (nn::Index t = 0; t < e.rows(); ++t) spikes += (e.at(t, f) > 0.5f) ? 1 : 0;
        EXPECT_EQ(spikes, 1) << "feature " << f << " did not fire exactly once";
    }
    // Regression guard on the measured old behaviour: 256 features must produce 256
    // spikes, not the ~1 the coincidence-detector version emitted for a whole window.
    EXPECT_EQ(count_spikes(e), 256);
}

TEST(Meeting01Encoding, LatencyLargerValuesFireEarlier)
{
    meeting01::Tensor w(4, 1);
    w.at(0, 0) = -8.0f; // global min -> latest
    w.at(1, 0) = 0.0f;
    w.at(2, 0) = 3.0f;
    w.at(3, 0) = 8.0f; // global max -> earliest
    const auto e = meeting01::encode_sample(w, "latency", 0, kSteps);

    auto spike_time = [&](nn::Index f) -> nn::Index
    {
        for (nn::Index t = 0; t < e.rows(); ++t)
            if (e.at(t, f) > 0.5f) return t;
        ADD_FAILURE() << "feature " << f << " never fired";
        return -1;
    };
    EXPECT_LT(spike_time(3), spike_time(2));
    EXPECT_LT(spike_time(2), spike_time(1));
    EXPECT_LT(spike_time(1), spike_time(0));
    EXPECT_EQ(spike_time(3), 0);
    EXPECT_EQ(spike_time(0), kSteps - 1);
}

TEST(Meeting01Encoding, PoissonRateMatchesNormalisedAmplitudeOverSteps)
{
    // One feature at the global min, one at the global max, one halfway. Firing RATE
    // over the T steps must track the normalised amplitude — that is what makes this a
    // rate code rather than the single Bernoulli draw the old T=1 layout allowed.
    constexpr int kLongRun = 4000;
    meeting01::Tensor w(3, 1);
    w.at(0, 0) = -10.0f;
    w.at(1, 0) = 0.0f;
    w.at(2, 0) = 10.0f;

    const auto e = meeting01::encode_sample(w, "poisson", /*seed=*/1234, kLongRun);
    auto rate = [&](nn::Index f)
    {
        int n = 0;
        for (nn::Index t = 0; t < e.rows(); ++t) n += (e.at(t, f) > 0.5f) ? 1 : 0;
        return static_cast<float>(n) / static_cast<float>(e.rows());
    };

    EXPECT_NEAR(rate(0), 0.0f, 0.02f);
    EXPECT_NEAR(rate(1), 0.5f, 0.03f);
    EXPECT_NEAR(rate(2), 1.0f, 0.02f);
}

TEST(Meeting01Encoding, DirectHoldsTheAnalogValueAtEveryStep)
{
    const auto window = make_window(64);
    const auto e = meeting01::encode_sample(window, "direct", 0, kSteps);
    for (nn::Index t = 0; t < e.rows(); ++t)
        for (nn::Index f = 0; f < e.cols(); ++f) EXPECT_FLOAT_EQ(e.at(t, f), window.at(f));
}

TEST(Meeting01Encoding, ReconstructionTargetIsTheOriginalWindowNotTheCode)
{
    const auto window = make_window(64);
    const auto target = meeting01::make_reconstruction_target(window, kSteps);

    ASSERT_EQ(target.rows(), kSteps);
    ASSERT_EQ(target.cols(), 64);
    for (nn::Index t = 0; t < target.rows(); ++t)
        for (nn::Index f = 0; f < target.cols(); ++f)
            EXPECT_FLOAT_EQ(target.at(t, f), window.at(f));
}

TEST(Meeting01Encoding, TargetVarianceIsIdenticalAcrossEncodings)
{
    // The property that makes val_mse comparable between encodings, and that stops the
    // GA from winning 261x by simply selecting the lowest-variance code: the target
    // does not depend on the encoding at all.
    const auto window = make_window(128);
    const auto target = meeting01::make_reconstruction_target(window, kSteps);

    double sum = 0.0;
    for (nn::Index i = 0; i < target.size(); ++i) sum += target.at(i);
    const double mean = sum / target.size();
    double var = 0.0;
    for (nn::Index i = 0; i < target.size(); ++i)
        var += (target.at(i) - mean) * (target.at(i) - mean);
    var /= target.size();

    // A z-scored window repeated across steps keeps unit variance regardless of which
    // encoding feeds the network.
    EXPECT_NEAR(var, 1.0, 0.05);
}

TEST(Meeting01Encoding, ReduceTimeMajorOutputAveragesOverSteps)
{
    meeting01::Tensor out(kSteps, 3); // B = 1
    for (nn::Index t = 0; t < kSteps; ++t)
        for (nn::Index f = 0; f < 3; ++f) out.at(t, f) = static_cast<float>(t) + f;

    const auto reduced = meeting01::reduce_time_major_output(out, kSteps);
    ASSERT_EQ(reduced.rows(), 1);
    ASSERT_EQ(reduced.cols(), 3);
    const float mean_t = (kSteps - 1) / 2.0f;
    for (nn::Index f = 0; f < 3; ++f) EXPECT_NEAR(reduced.at(0, f), mean_t + f, 1e-4f);
}

TEST(Meeting01Encoding, ReduceTimeMajorOutputRejectsRaggedInput)
{
    meeting01::Tensor out(7, 2);
    EXPECT_THROW((void) meeting01::reduce_time_major_output(out, kSteps), std::invalid_argument);
}

} // namespace
