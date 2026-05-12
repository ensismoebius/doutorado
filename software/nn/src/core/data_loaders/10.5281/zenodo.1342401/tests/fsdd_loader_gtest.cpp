// fsdd_loader_gtest.cpp — Unit tests for FsddLoader (filename parsing).
// FsddWindowDataset integration tests require the actual dataset on disk
// and are therefore skipped when the path is absent.

#include <gtest/gtest.h>

#include "data_loaders/10.5281/zenodo.1342401/loaders/FsddLoader.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "data_loaders/10.5281/zenodo.1342401/schema/Metadata.hpp"
#include "data_loaders/10.5281/zenodo.1342401/schema/Names.hpp"

using nn::dataLoaders::fsdd::FsddLoader;
using nn::dataLoaders::fsdd::FsddWindowDataset;

// ---------------------------------------------------------------------------
// FsddLoader::parse_filename
// ---------------------------------------------------------------------------

TEST(FsddLoader, ParseFilenameTypical)
{
    const auto info = FsddLoader::parse_filename("0_jackson_0");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->digit,   0);
    EXPECT_EQ(info->speaker, "jackson");
    EXPECT_EQ(info->trial,   0);
}

TEST(FsddLoader, ParseFilenameMaxValues)
{
    const auto info = FsddLoader::parse_filename("9_yweweler_49");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->digit,   9);
    EXPECT_EQ(info->speaker, "yweweler");
    EXPECT_EQ(info->trial,   49);
}

TEST(FsddLoader, ParseFilenameHyphenatedSpeaker)
{
    // Hypothetical future speaker with underscore in name — last underscore = trial delimiter.
    const auto info = FsddLoader::parse_filename("3_some_speaker_5");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->digit,   3);
    EXPECT_EQ(info->speaker, "some_speaker");
    EXPECT_EQ(info->trial,   5);
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
              nn::dataLoaders::fsdd::kDigitCount *
              nn::dataLoaders::fsdd::kSpeakerCount *
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
    EXPECT_THROW(
        FsddWindowDataset("/nonexistent/path/to/fsdd", 512),
        std::runtime_error);
}

TEST(FsddWindowDataset, ThrowsOnZeroWindowSize)
{
    EXPECT_THROW(
        FsddWindowDataset("/nonexistent/path/to/fsdd", 0),
        std::invalid_argument);
}
