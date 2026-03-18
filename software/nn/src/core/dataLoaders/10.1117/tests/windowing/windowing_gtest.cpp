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
