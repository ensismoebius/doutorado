/**
 * @file src/experiments/02/tests/Experiment02Data_gtest.cpp
 * @brief Implementation for Experiment02data gtest.
 *

 */

#include <vector>

#include "../Experiment02Data.hpp"
#include "gtest/gtest.h"

namespace
{
auto make_eeg_sample(int stimulus = 1, int artifacts = 1) -> EEGSample
{
    EEGSample sample;
    sample.channels.resize(6, std::vector<double>(4096, 0.5));
    sample.modality = 1;
    sample.stimulus = stimulus;
    sample.artifacts = artifacts;
    return sample;
}

auto make_audio_sample(int eeg_index = 0, int stimulus = 1, std::size_t size = 4096) -> AudioSample
{
    AudioSample sample;
    sample.signal.assign(size, 0.25);
    sample.stimulus = stimulus;
    sample.eeg_index = eeg_index;
    return sample;
}
} // namespace

TEST(Experiment02DataTest, ExtractWindowsRejectsInvalidOverlap)
{
    std::vector<EEGSample> eeg_samples{make_eeg_sample()};
    std::vector<AudioSample> audio_samples{make_audio_sample()};

    EXPECT_THROW((void) extract_windows(eeg_samples, audio_samples, 1.0, 1.0, 1000, 1000),
        std::invalid_argument);
    EXPECT_THROW((void) extract_windows(eeg_samples, audio_samples, 1.0, 1.1, 1000, 1000),
        std::invalid_argument);
}

TEST(Experiment02DataTest, ExtractWindowsRejectsInvalidRatesOrWindow)
{
    std::vector<EEGSample> eeg_samples{make_eeg_sample()};
    std::vector<AudioSample> audio_samples{make_audio_sample()};

    EXPECT_THROW((void) extract_windows(eeg_samples, audio_samples, 1.0, 0.0, 0, 1000),
        std::invalid_argument);
    EXPECT_THROW((void) extract_windows(eeg_samples, audio_samples, 0.001, 0.0, 10, 10),
        std::invalid_argument);
}

TEST(Experiment02DataTest, ExtractWindowsProducesOutputForValidInput)
{
    std::vector<EEGSample> eeg_samples{make_eeg_sample(3, 1)};
    std::vector<AudioSample> audio_samples{make_audio_sample(0, 3)};

    auto windows = extract_windows(eeg_samples, audio_samples, 0.1, 0.0, 10, 10);

    ASSERT_FALSE(windows.empty());
    EXPECT_EQ(windows.front().label, 3);
    EXPECT_FALSE(windows.front().eeg_window.empty());
    EXPECT_FALSE(windows.front().audio_window.empty());
}
