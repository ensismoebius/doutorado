// fsdd_loader_gtest.cpp — Unit tests for FsddLoader (filename parsing).
// FsddWindowDataset integration tests require the actual dataset on disk
// and are therefore skipped when the path is absent, EXCEPT for the
// valid_length/z-score synthetic-fixture tests below, which write a minimal WAV
// via Wav::write() so that windowing/padding mechanics are exercised even without
// the real corpus.

#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "data_loaders/10.5281/zenodo.1342401/loaders/FsddLoader.hpp"
#include "data_loaders/10.5281/zenodo.1342401/schema/Metadata.hpp"
#include "data_loaders/10.5281/zenodo.1342401/schema/Names.hpp"
#include "wave/Wav.hpp"

using nn::dataLoaders::fsdd::FsddLoader;
using nn::dataLoaders::fsdd::FsddWindowDataset;

// ---------------------------------------------------------------------------
// FsddLoader::parse_filename
// ---------------------------------------------------------------------------

TEST(FsddLoader, ParseFilenameTypical)
{
    const auto info = FsddLoader::parse_filename("0_jackson_0");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->digit, 0);
    EXPECT_EQ(info->speaker, "jackson");
    EXPECT_EQ(info->trial, 0);
}

TEST(FsddLoader, ParseFilenameMaxValues)
{
    const auto info = FsddLoader::parse_filename("9_yweweler_49");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->digit, 9);
    EXPECT_EQ(info->speaker, "yweweler");
    EXPECT_EQ(info->trial, 49);
}

TEST(FsddLoader, ParseFilenameHyphenatedSpeaker)
{
    // Hypothetical future speaker with underscore in name — last underscore = trial delimiter.
    const auto info = FsddLoader::parse_filename("3_some_speaker_5");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->digit, 3);
    EXPECT_EQ(info->speaker, "some_speaker");
    EXPECT_EQ(info->trial, 5);
}

TEST(FsddLoader, ParseFilenameInvalidNoUnderscore)
{
    EXPECT_FALSE(FsddLoader::parse_filename("0jackson0").has_value());
}

TEST(FsddLoader, ParseFilenameInvalidNonNumericDigit)
{
    EXPECT_FALSE(FsddLoader::parse_filename("x_jackson_0").has_value());
}

TEST(FsddLoader, ParseFilenameInvalidNonNumericTrial)
{
    EXPECT_FALSE(FsddLoader::parse_filename("0_jackson_x").has_value());
}

TEST(FsddLoader, ParseFilenameEmpty)
{
    EXPECT_FALSE(FsddLoader::parse_filename("").has_value());
}

// ---------------------------------------------------------------------------
// Schema constants sanity
// ---------------------------------------------------------------------------

TEST(FsddMetadata, TotalFileCount)
{
    EXPECT_EQ(nn::dataLoaders::fsdd::kTotalFiles,
        nn::dataLoaders::fsdd::kDigitCount * nn::dataLoaders::fsdd::kSpeakerCount *
            nn::dataLoaders::fsdd::kTrialsPerSpeakerDigit);
    EXPECT_EQ(nn::dataLoaders::fsdd::kTotalFiles, 3000);
}

TEST(FsddMetadata, SpeakerListSize)
{
    EXPECT_EQ(nn::dataLoaders::fsdd::kSpeakers.size(),
        static_cast<std::size_t>(nn::dataLoaders::fsdd::kSpeakerCount));
}

TEST(FsddMetadata, DigitNamesSize)
{
    EXPECT_EQ(nn::dataLoaders::fsdd::kDigitNames.size(),
        static_cast<std::size_t>(nn::dataLoaders::fsdd::kDigitCount));
}

// ---------------------------------------------------------------------------
// FsddWindowDataset — integration (skipped when dataset absent)
// ---------------------------------------------------------------------------

TEST(FsddWindowDataset, ThrowsOnMissingRoot)
{
    EXPECT_THROW(FsddWindowDataset("/nonexistent/path/to/fsdd", 512), std::runtime_error);
}

TEST(FsddWindowDataset, ThrowsOnZeroWindowSize)
{
    EXPECT_THROW(FsddWindowDataset("/nonexistent/path/to/fsdd", 0), std::invalid_argument);
}

namespace
{
// One 12-sample recording, window_size=8: window 0 is full (take=8), window 1 is a
// trailing partial (take=4, indices [8,9,10,11) of an increasing-value signal chosen so
// the real portion has non-zero variance -- z-scoring it against its OWN mean is directly
// observable (mean -> ~0) and distinguishable from z-scoring the full padded window
// (whose mean/std would be diluted by the trailing zeros).
auto write_synthetic_fsdd_wav(const std::filesystem::path& dir) -> void
{
    std::filesystem::create_directories(dir);
    Wav wav;
    const std::vector<float> signal = {
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F, // window 0: full, constant (std=0)
        10.0F,
        20.0F,
        30.0F,
        40.0F // window 1: partial (take=4)
    };
    wav.write((dir / "3_testspk_0.wav").string(), signal, 8000);
}
} // namespace

TEST(FsddWindowDataset, TrailingPartialWindowReportsValidLength)
{
    const auto root = std::filesystem::temp_directory_path() / "fsdd_window_valid_length_test";
    std::filesystem::remove_all(root);
    write_synthetic_fsdd_wav(root);

    FsddWindowDataset ds(root, /*window_size=*/8);
    ASSERT_EQ(ds.size(), 2u);
    EXPECT_EQ(ds.metadata()[0].valid_length, 8); // full window
    EXPECT_EQ(ds.metadata()[1].valid_length, 4); // trailing partial window

    std::filesystem::remove_all(root);
}

// Regression test for the z-score-over-padding bug: normalizing the whole (real + zero-
// padded) window let the padding drag the statistics and stopped the padded tail from
// even being literally zero. After the fix, the real prefix is normalized against its OWN
// statistics (mean -> ~0) and the padded tail stays exactly 0.0.
TEST(FsddWindowDataset, PartialWindowZScoreExcludesPadding)
{
    const auto root = std::filesystem::temp_directory_path() / "fsdd_window_zscore_test";
    std::filesystem::remove_all(root);
    write_synthetic_fsdd_wav(root);

    FsddWindowDataset ds(root, /*window_size=*/8);
    ASSERT_EQ(ds.size(), 2u);
    const auto& partial = ds.windows()[1];

    double sum = 0.0;
    for (nn::Index t = 0; t < 4; ++t) sum += partial.at(t, 0);
    EXPECT_NEAR(sum / 4.0, 0.0, 1e-4) << "real prefix should be z-scored against its own mean";

    for (nn::Index t = 4; t < 8; ++t)
        EXPECT_FLOAT_EQ(partial.at(t, 0), 0.0F)
            << "padded tail must stay literally zero, index " << t;

    std::filesystem::remove_all(root);
}
