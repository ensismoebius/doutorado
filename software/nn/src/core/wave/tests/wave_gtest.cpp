/**
 * @file wave_gtest.cpp
 * @brief Unit tests for wave utilities (WAV I/O, filtering, simple feature extraction).
 */

#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nn/wave/Wav.h"
#include "nn/wave/audioFeatureExtraction.h"
#include "nn/wave/filter_operations.hpp"
#include "nn/wave/signal_operations.hpp"

TEST(SimpleSignalOperationsTest, TestAMDF)
{
    std::vector<long double> signal = {1.0, 2.0, 3.0, 2.0, 1.0};
    auto result = amdf(signal);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), signal.size());
}

TEST(FilterOperationsTest, TestCreateAlpha)
{
    double sampling_rate = 44100.0;
    double cutoff_frequency = 2000.0;
    auto alpha = createAlpha(sampling_rate, cutoff_frequency);
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 1.0);
}

TEST(WavFileTest, WriteThenRead)
{
    const std::string filepath = std::filesystem::temp_directory_path().string() + "/output.wav";
    std::vector<float> data = {0.0F, 0.1F, 0.2F, 0.3F};

    // write
    Wav writer;
    ASSERT_NO_THROW(writer.write(filepath, data, 44100));
    ASSERT_EQ(writer.get_path(), filepath);

    // read
    Wav reader;
    ASSERT_NO_THROW(reader.read(filepath)); // flawfinder: ignore
    auto read_data = reader.get_data();
    ASSERT_FALSE(read_data.empty());
    // optional: compare contents (convert types if needed)
}

TEST(AudioFeatureExtractionTest, TestHanningWindow)
{
    int length = 4;
    auto window = nn::core::wave::hanning_window(length);
    EXPECT_EQ(window.size(), static_cast<size_t>(length));
    // Hanning window values for length 4
    EXPECT_NEAR(window[0], 0.0, 1e-6);
    EXPECT_NEAR(window[1], 0.75, 1e-6);
    EXPECT_NEAR(window[2], 0.75, 1e-6);
    EXPECT_NEAR(window[3], 0.0, 1e-6);
}

TEST(AudioFeatureExtractionTest, TestHanningWindowEdgeCases)
{
    // Length 1
    auto window_1 = nn::core::wave::hanning_window(1);
    EXPECT_EQ(window_1.size(), 1U);
    EXPECT_NEAR(window_1[0], 1.0, 1e-6);

    // Length 2
    auto window_2 = nn::core::wave::hanning_window(2);
    EXPECT_EQ(window_2.size(), 2U);
    EXPECT_NEAR(window_2[0], 0.0, 1e-6);
    EXPECT_NEAR(window_2[1], 0.0, 1e-6);

    // Length 0
    auto window_0 = nn::core::wave::hanning_window(0);
    EXPECT_TRUE(window_0.empty());
}

TEST(AudioFeatureExtractionTest, TestApplyWindow)
{
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> window = {0.0, 0.5, 1.0, 0.5};
    auto result = nn::core::wave::apply_window(signal, window);
    EXPECT_EQ(result.size(), signal.size());
    EXPECT_NEAR(result[0], 0.0, 1e-6);
    EXPECT_NEAR(result[1], 1.0, 1e-6);
    EXPECT_NEAR(result[2], 3.0, 1e-6);
    EXPECT_NEAR(result[3], 2.0, 1e-6);
}

TEST(AudioFeatureExtractionTest, TestApplyWindowEdgeCases)
{
    // Empty vectors
    std::vector<double> empty_signal;
    std::vector<double> empty_window;
    auto result_empty = nn::core::wave::apply_window(empty_signal, empty_window);
    EXPECT_TRUE(result_empty.empty());

    // Mismatched sizes (should throw or handle)
    std::vector<double> signal = {1.0, 2.0};
    std::vector<double> window = {0.5};
    EXPECT_THROW(nn::core::wave::apply_window(signal, window), std::invalid_argument);
}
