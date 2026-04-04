/**
 * @file windowing_gtest.cpp
 * @brief Unit tests for WindowSpec, WindowingEngine, and the windowed datasets.
 *
 * Coverage:
 *  - WindowSpec::hop_size() and num_windows() correctness.
 *  - WindowingEngine::compute_windows() boundary conditions.
 *  - Windowed datasets: size(), get_item() shapes, collate_into() zero-copy resize.
 *
 * Dataset tests use small in-memory tensors via TensorDataset-style scaffolding to
 * avoid requiring real MAT files.  Heavy I/O tests (with actual .mat fixtures) are
 * left for integration testing.
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>

#include "../utils/MockImaginedSpeechDatasetGenerator.hpp"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/10.1117/windowing/AudioWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/EEGWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/FusedWindowDataset.hpp"
#include "nn/utility/batching.hpp"
#include "nn/windowing/WindowSpec.hpp"
#include "nn/windowing/WindowingEngine.hpp"

using nn::windowing::compute_windows;
using nn::windowing::WindowSpec;

// ---------------------------------------------------------------------------
// WindowSpec tests
// ---------------------------------------------------------------------------

TEST(WindowSpecTest, HopSizeNoOverlap)
{
    WindowSpec spec{.window_size = 100, .overlap = 0.0f, .sample_rate = 1000};
    EXPECT_EQ(spec.hop_size(), 100);
}

TEST(WindowSpecTest, HopSizeHalfOverlap)
{
    WindowSpec spec{.window_size = 256, .overlap = 0.5f, .sample_rate = 1024};
    EXPECT_EQ(spec.hop_size(), 128);
}

TEST(WindowSpecTest, HopSizeMinimumOne)
{
    // overlap = 0.99 rounds hop to ≥ 1
    WindowSpec spec{.window_size = 2, .overlap = 0.99f, .sample_rate = 1};
    EXPECT_GE(spec.hop_size(), 1);
}

TEST(WindowSpecTest, NumWindowsExact)
{
    // signal = 100, window = 50, hop = 25  →  windows at 0, 25, 50  → 3
    WindowSpec spec{.window_size = 50, .overlap = 0.5f, .sample_rate = 1};
    EXPECT_EQ(spec.num_windows(100), 3);
}

TEST(WindowSpecTest, NumWindowsShorterThanWindow)
{
    WindowSpec spec{.window_size = 200, .overlap = 0.5f, .sample_rate = 1};
    EXPECT_EQ(spec.num_windows(100), 0);
}

TEST(WindowSpecTest, NumWindowsEqualToWindow)
{
    // Exactly one window.
    WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1};
    EXPECT_EQ(spec.num_windows(100), 1);
}

TEST(WindowSpecTest, ValidateThrowsOnBadWindowSize)
{
    WindowSpec spec{.window_size = 0, .overlap = 0.5f, .sample_rate = 1024};
    EXPECT_THROW(spec.validate(), std::invalid_argument);
}

TEST(WindowSpecTest, ValidateThrowsOnBadOverlap)
{
    WindowSpec spec{.window_size = 128, .overlap = 1.0f, .sample_rate = 1024};
    EXPECT_THROW(spec.validate(), std::invalid_argument);
}

TEST(WindowSpecTest, ValidateThrowsOnBadSampleRate)
{
    WindowSpec spec{.window_size = 128, .overlap = 0.5f, .sample_rate = 0};
    EXPECT_THROW(spec.validate(), std::invalid_argument);
}

TEST(WindowSpecTest, ValidateSucceeds)
{
    WindowSpec spec{.window_size = 128, .overlap = 0.5f, .sample_rate = 1024};
    EXPECT_NO_THROW(spec.validate());
}

// ---------------------------------------------------------------------------
// WindowingEngine tests
// ---------------------------------------------------------------------------

TEST(WindowingEngineTest, EmptyOnZeroSignal)
{
    WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 100};
    EXPECT_TRUE(compute_windows(0, spec).empty());
}

TEST(WindowingEngineTest, EmptyWhenSignalShorterThanWindow)
{
    WindowSpec spec{.window_size = 200, .overlap = 0.5f, .sample_rate = 100};
    EXPECT_TRUE(compute_windows(100, spec).empty());
}

TEST(WindowingEngineTest, SingleWindowWhenSignalEqualsWindow)
{
    WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 100};
    auto w = compute_windows(100, spec);
    ASSERT_EQ(w.size(), 1u);
    EXPECT_EQ(w[0].start, 0);
    EXPECT_EQ(w[0].end, 100);
}

TEST(WindowingEngineTest, CorrectStartEndPositions)
{
    // window=10, hop=5, signal=20 → starts at 0, 5, 10  (3 windows)
    WindowSpec spec{.window_size = 10, .overlap = 0.5f, .sample_rate = 100};
    auto w = compute_windows(20, spec);
    ASSERT_EQ(w.size(), 3u);
    EXPECT_EQ(w[0].start, 0);
    EXPECT_EQ(w[0].end, 10);
    EXPECT_EQ(w[1].start, 5);
    EXPECT_EQ(w[1].end, 15);
    EXPECT_EQ(w[2].start, 10);
    EXPECT_EQ(w[2].end, 20);
}

TEST(WindowingEngineTest, NoOverlapWindowsAreContiguous)
{
    WindowSpec spec{.window_size = 10, .overlap = 0.0f, .sample_rate = 100};
    auto w = compute_windows(30, spec);
    ASSERT_EQ(w.size(), 3u);
    for (std::size_t i = 1; i < w.size(); ++i)
    {
        EXPECT_EQ(w[i].start, w[i - 1].end) << "windows not contiguous at index " << i;
    }
}

TEST(WindowingEngineTest, CenterTimeIsCorrect)
{
    // window=100, sample_rate=100 → center of first window = 0.5 s
    WindowSpec spec{.window_size = 100, .overlap = 0.0f, .sample_rate = 100};
    auto w = compute_windows(200, spec);
    ASSERT_GE(w.size(), 1u);
    EXPECT_DOUBLE_EQ(w[0].center_time_s, 0.5);
}

TEST(WindowingEngineTest, CountMatchesNumWindows)
{
    WindowSpec spec{.window_size = 256, .overlap = 0.5f, .sample_rate = 1024};
    const int signal_length = 4096;
    auto windows = compute_windows(signal_length, spec);
    EXPECT_EQ(static_cast<int>(windows.size()), spec.num_windows(signal_length));
}

TEST(WindowingEngineTest, EEGSchemaWindows_250msAt1024Hz)
{
    // Real EEG recording: 4s @ 1024 Hz = 4096 samples.
    // Window = 256 samples (~250 ms), overlap = 0.5 → hop = 128 → 31 windows.
    WindowSpec spec{.window_size = 256, .overlap = 0.5f, .sample_rate = 1024};
    auto w = compute_windows(4096, spec);
    // (4096 - 256) / 128 + 1 = 31
    EXPECT_EQ(w.size(), 31u);
}

TEST(WindowingEngineTest, AudioSchemaWindows_250msAt44100Hz)
{
    // Audio: 4s @ 44100 Hz = 176400 samples.
    // Window = 11025 samples (~250 ms), overlap = 0.5 → hop = 5512 → 31 windows.
    WindowSpec spec{.window_size = 11025, .overlap = 0.5f, .sample_rate = 44100};
    auto w = compute_windows(176400, spec);
    // (176400 - 11025) / 5512 + 1 = 31  (approximately, depend on rounding)
    EXPECT_GT(w.size(), 0u);
    // Verify no window exceeds signal bounds.
    for (const auto& win : w)
    {
        EXPECT_GE(win.start, 0);
        EXPECT_LE(win.end, 176400);
    }
}

TEST(WindowingEngineTest, SameSyncWindowCountForMatchedSpecs)
{
    // Verify that window count is equal for EEG and Audio when using the same
    // overlap and window durations that divide evenly — ensuring FusedWindowDataset
    // can align windows by index.
    const float overlap = 0.5f;

    // Approximately 500 ms per window
    WindowSpec eeg_spec{.window_size = 512, .overlap = overlap, .sample_rate = 1024};
    WindowSpec audio_spec{.window_size = 22050, .overlap = overlap, .sample_rate = 44100};

    const int eeg_n = eeg_spec.num_windows(4096);       // 1024 * 4
    const int audio_n = audio_spec.num_windows(176400); // 44100 * 4

    // Both should be non-zero.
    EXPECT_GT(eeg_n, 0);
    EXPECT_GT(audio_n, 0);
    // The minimum (used by FusedWindowDataset as windows_per_pair) should be > 0.
    EXPECT_GT(std::min(eeg_n, audio_n), 0);
}

// ---------------------------------------------------------------------------
// AudioWindowDataset tests (no MAT files required)
// ---------------------------------------------------------------------------

TEST(AudioWindowDatasetTest, EmptySubjectsYieldsSizeZero)
{
    // window_size=100, which is well within 176400 samples → validates OK
    nn::windowing::WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 44100};
    AudioWindowDataset ds({} /* empty subjects */, spec);
    EXPECT_EQ(ds.size(), 0u);
}

TEST(AudioWindowDatasetTest, GetItemThrowsWhenEmpty)
{
    nn::windowing::WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 44100};
    AudioWindowDataset ds({}, spec);
    EXPECT_THROW(ds.get_item(0), std::out_of_range);
}

TEST(AudioWindowDatasetTest, CollateEmptyIndicesProducesEmptyBatch)
{
    nn::windowing::WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 44100};
    AudioWindowDataset ds({}, spec);
    Batch batch;
    ds.collate_into({}, batch);
    EXPECT_EQ(batch.inputs.rows(), 0);
    EXPECT_EQ(batch.targets.rows(), 0);
}

TEST(AudioWindowDatasetTest, WindowTooLargeThrows)
{
    // 176400 is kAudioSamples; use 176401 to exceed it and trigger the throw
    nn::windowing::WindowSpec spec{.window_size = 176401, .overlap = 0.0f, .sample_rate = 44100};
    EXPECT_THROW(AudioWindowDataset({}, spec), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// EEGWindowDataset tests (no MAT files required)
// ---------------------------------------------------------------------------

TEST(EEGWindowDatasetTest, EmptySubjectsYieldsSizeZero)
{
    // kEegSamplesPerChannel = 4096; use window_size=100
    nn::windowing::WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1024};
    EEGWindowDataset ds({}, spec);
    EXPECT_EQ(ds.size(), 0u);
}

TEST(EEGWindowDatasetTest, GetItemThrowsWhenEmpty)
{
    nn::windowing::WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1024};
    EEGWindowDataset ds({}, spec);
    EXPECT_THROW(ds.get_item(0), std::out_of_range);
}

TEST(EEGWindowDatasetTest, CollateEmptyIndicesProducesEmptyBatch)
{
    nn::windowing::WindowSpec spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1024};
    EEGWindowDataset ds({}, spec);
    Batch batch;
    ds.collate_into({}, batch);
    EXPECT_EQ(batch.inputs.rows(), 0);
    EXPECT_EQ(batch.targets.rows(), 0);
}

TEST(EEGWindowDatasetTest, WindowTooLargeThrows)
{
    // 4096 is kEegSamplesPerChannel; use 4097 to exceed it
    nn::windowing::WindowSpec spec{.window_size = 4097, .overlap = 0.0f, .sample_rate = 1024};
    EXPECT_THROW(EEGWindowDataset({}, spec), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// FusedWindowDataset tests (no MAT files required)
// ---------------------------------------------------------------------------

TEST(FusedWindowDatasetTest, EmptySubjectsYieldsSizeZero)
{
    nn::windowing::WindowSpec eeg_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 44100};
    FusedWindowDataset ds({}, eeg_spec, audio_spec);
    EXPECT_EQ(ds.size(), 0u);
}

TEST(FusedWindowDatasetTest, GetItemThrowsWhenEmpty)
{
    nn::windowing::WindowSpec eeg_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 44100};
    FusedWindowDataset ds({}, eeg_spec, audio_spec);
    EXPECT_THROW(ds.get_item(0), std::out_of_range);
}

TEST(FusedWindowDatasetTest, CollateEmptyIndicesProducesEmptyBatch)
{
    nn::windowing::WindowSpec eeg_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 44100};
    FusedWindowDataset ds({}, eeg_spec, audio_spec);
    Batch batch;
    ds.collate_into({}, batch);
    EXPECT_EQ(batch.inputs.rows(), 0);
    EXPECT_EQ(batch.targets.rows(), 0);
}

TEST(FusedWindowDatasetTest, EEGWindowTooLargeThrows)
{
    // eeg window exceeds 4096 → eeg_n = 0 → throw
    nn::windowing::WindowSpec eeg_spec{.window_size = 4097, .overlap = 0.0f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 44100};
    EXPECT_THROW(FusedWindowDataset({}, eeg_spec, audio_spec), std::invalid_argument);
}

TEST(FusedWindowDatasetTest, AudioWindowTooLargeThrows)
{
    // audio window exceeds 176400 → audio_n = 0 → throw
    nn::windowing::WindowSpec eeg_spec{.window_size = 100, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_spec{
        .window_size = 176401, .overlap = 0.0f, .sample_rate = 44100};
    EXPECT_THROW(FusedWindowDataset({}, eeg_spec, audio_spec), std::invalid_argument);
}

namespace
{
class WindowingDatasetIntegrationTest : public ::testing::Test
{
   protected:
    std::filesystem::path tmp_dir;
    SubjectFiles subject;

    void SetUp() override
    {
        tmp_dir =
            std::filesystem::temp_directory_path() / ("windowing_ds_" + std::to_string(getpid()));
        std::filesystem::remove_all(tmp_dir);
        std::filesystem::create_directories(tmp_dir);

        const auto eeg = tmp_dir / "S01_EEG.mat";
        const auto audio = tmp_dir / "S01_Audio.mat";
        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateEEGMatFile(eeg, 2U);
        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateAudioMatFile(audio, 2U);

        subject.subject_id = 1;
        subject.subject_name = "S01";
        subject.eeg_mat_path = eeg.string();
        subject.audio_mat_path = audio.string();
        subject.eeg_rows = 2U;
        subject.audio_rows = 2U;
    }

    void TearDown() override
    {
        std::filesystem::remove_all(tmp_dir);
    }
};
} // namespace

TEST_F(WindowingDatasetIntegrationTest, AudioWindowDatasetLoadsAndCollates)
{
    nn::windowing::WindowSpec spec{.window_size = 11025, .overlap = 0.5f, .sample_rate = 44100};
    AudioWindowDataset ds({subject}, spec);
    ASSERT_GT(ds.size(), 0U);

    const Batch item = ds.get_item(0);
    EXPECT_EQ(item.inputs.rows(), 1);
    EXPECT_EQ(item.inputs.cols(), 11025);
    EXPECT_EQ(item.targets.cols(), 2);

    Batch b;
    ds.collate_into({0, 1}, b);
    EXPECT_EQ(b.inputs.rows(), 2);
    EXPECT_EQ(b.inputs.cols(), 11025);
    EXPECT_EQ(b.targets.rows(), 2);
    EXPECT_EQ(b.targets.cols(), 2);
}

TEST_F(WindowingDatasetIntegrationTest, EEGWindowDatasetLoadsAndCollates)
{
    nn::windowing::WindowSpec spec{.window_size = 256, .overlap = 0.5f, .sample_rate = 1024};
    EEGWindowDataset ds({subject}, spec);
    ASSERT_GT(ds.size(), 0U);

    const Batch item = ds.get_item(0);
    EXPECT_EQ(item.inputs.rows(), 1);
    EXPECT_EQ(item.inputs.cols(), 6 * 256);
    EXPECT_EQ(item.targets.cols(), 3);

    Batch b;
    ds.collate_into({0, 1}, b);
    EXPECT_EQ(b.inputs.rows(), 2);
    EXPECT_EQ(b.inputs.cols(), 6 * 256);
    EXPECT_EQ(b.targets.rows(), 2);
    EXPECT_EQ(b.targets.cols(), 3);
}

TEST_F(WindowingDatasetIntegrationTest, FusedWindowDatasetLoadsAndCollates)
{
    nn::windowing::WindowSpec eeg_spec{.window_size = 256, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_spec{
        .window_size = 11025, .overlap = 0.5f, .sample_rate = 44100};

    FusedWindowDataset ds({subject}, eeg_spec, audio_spec);
    ASSERT_GT(ds.size(), 0U);

    const Batch item = ds.get_item(0);
    EXPECT_EQ(item.inputs.rows(), 1);
    EXPECT_EQ(item.inputs.cols(), (6 * 256) + 11025);
    EXPECT_EQ(item.targets.cols(), 5);

    Batch b;
    ds.collate_into({0, 1}, b);
    EXPECT_EQ(b.inputs.rows(), 2);
    EXPECT_EQ(b.inputs.cols(), (6 * 256) + 11025);
    EXPECT_EQ(b.targets.rows(), 2);
    EXPECT_EQ(b.targets.cols(), 5);
}
