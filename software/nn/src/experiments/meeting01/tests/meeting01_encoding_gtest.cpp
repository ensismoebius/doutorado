// meeting01_encoding_gtest.cpp — direct coverage for the two `encode_sample` fixes:
// (1) "latency" must be canonical time-to-first-spike (exactly one spike per
//     channel, larger values fire earlier), not a threshold-crossing code that
//     stays on; (2) "poisson" must normalize over the full [min, max] range, not
//     max-only, so sub-mean (negative, post-z-score) samples keep a real firing
//     probability instead of being silently clamped to 0.

#include <gtest/gtest.h>

#include "Meeting01Encoding.hpp"

namespace
{

TEST(Meeting01Encoding, LatencyEncodesExactlyOneSpikePerChannel)
{
    // 16 time steps, 4 channels spanning distinct values so t_spike differs per
    // channel; values held constant per channel in "time" since encode_sample's
    // t_spike is purely a function of the (t, d) cell's own value, not of t.
    meeting01::Tensor sample(16, 4);
    const float values[4] = {-3.0f, -1.0f, 1.0f, 3.0f};
    for (nn::Index t = 0; t < sample.rows(); ++t)
        for (nn::Index d = 0; d < sample.cols(); ++d) sample.at(t, d) = values[d];

    const auto encoded = meeting01::encode_sample(sample, "latency", /*seed=*/0);

    for (nn::Index d = 0; d < sample.cols(); ++d)
    {
        int spikes = 0;
        for (nn::Index t = 0; t < sample.rows(); ++t) spikes += (encoded.at(t, d) > 0.5f) ? 1 : 0;
        EXPECT_EQ(spikes, 1) << "channel " << d
                             << " did not fire exactly once -- latency must be TTFS, "
                                "not a threshold-crossing code that stays on";
    }
}

TEST(Meeting01Encoding, LatencyLargerValuesFireEarlier)
{
    meeting01::Tensor sample(10, 2);
    for (nn::Index t = 0; t < sample.rows(); ++t)
    {
        sample.at(t, 0) = 8.0f;  // largest value in the sample -> should fire earliest
        sample.at(t, 1) = -8.0f; // smallest value in the sample -> should fire latest
    }

    const auto encoded = meeting01::encode_sample(sample, "latency", /*seed=*/0);

    auto spike_time = [&](nn::Index d) -> nn::Index
    {
        for (nn::Index t = 0; t < sample.rows(); ++t)
            if (encoded.at(t, d) > 0.5f) return t;
        ADD_FAILURE() << "channel " << d << " never fired";
        return -1;
    };

    EXPECT_LT(spike_time(0), spike_time(1));
}

TEST(Meeting01Encoding, PoissonUsesFullRangeNormalizationNotMaxOnly)
{
    constexpr int kTrials = 5000;
    // Row 0: the same sub-mean NEGATIVE value repeated kTrials times -- kTrials
    // independent Bernoulli trials of the same firing probability. Row 1: the
    // true min/max sentinels (columns 0/1) that set the sample's global range;
    // the rest is filler strictly inside [min, max] so it cannot move the range.
    meeting01::Tensor sample(2, kTrials);
    for (int c = 0; c < kTrials; ++c) sample.at(0, c) = -5.0f;
    sample.at(1, 0) = -10.0f;
    sample.at(1, 1) = 10.0f;
    for (int c = 2; c < kTrials; ++c) sample.at(1, c) = 0.0f;

    const auto encoded = meeting01::encode_sample(sample, "poisson", /*seed=*/1234);

    int fired = 0;
    for (int c = 0; c < kTrials; ++c) fired += (encoded.at(0, c) > 0.5f) ? 1 : 0;
    const float empirical_rate = static_cast<float>(fired) / static_cast<float>(kTrials);

    // Old max-only normalization clamped this negative value's probability to
    // exactly 0 (sample/max = -5/10, clamped to [0,1] -> 0): every trial would be
    // silent. Full-range min-max gives p = (-5 - (-10)) / 20 = 0.25.
    EXPECT_GT(fired, 0) << "poisson must not silently zero out sub-mean "
                           "(negative, post-z-score) samples";
    EXPECT_NEAR(empirical_rate, 0.25f, 0.03f);
}

TEST(Meeting01Encoding, PoissonFiringRateMatchesFullRangeAtExtremes)
{
    constexpr int kTrials = 5000;
    meeting01::Tensor sample(2, kTrials);
    for (int c = 0; c < kTrials; ++c) sample.at(0, c) = -10.0f; // == global min -> p ~ 0
    sample.at(1, 0) = -10.0f;
    sample.at(1, 1) = 10.0f;
    for (int c = 2; c < kTrials; ++c) sample.at(1, c) = 0.0f;

    const auto encoded_min = meeting01::encode_sample(sample, "poisson", /*seed=*/7);
    int fired_min = 0;
    for (int c = 0; c < kTrials; ++c) fired_min += (encoded_min.at(0, c) > 0.5f) ? 1 : 0;
    EXPECT_NEAR(static_cast<float>(fired_min) / kTrials, 0.0f, 0.02f);

    for (int c = 0; c < kTrials; ++c) sample.at(0, c) = 10.0f; // == global max -> p ~ 1
    const auto encoded_max = meeting01::encode_sample(sample, "poisson", /*seed=*/7);
    int fired_max = 0;
    for (int c = 0; c < kTrials; ++c) fired_max += (encoded_max.at(0, c) > 0.5f) ? 1 : 0;
    EXPECT_NEAR(static_cast<float>(fired_max) / kTrials, 1.0f, 0.02f);
}

} // namespace
