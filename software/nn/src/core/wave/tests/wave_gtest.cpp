/**
 * @file wave_gtest.cpp
 * @brief Unit tests for wave utilities (WAV I/O, filtering, simple feature extraction).
 */

#include <filesystem>
#include <fstream>
#include <numbers>
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

TEST(FilterOperationsTest, TestCreateAlphaHighPass)
{
    constexpr double sampling_rate = 48000.0;
    constexpr double cutoff_frequency = 4000.0;
    const auto low_alpha = createAlpha(sampling_rate, cutoff_frequency, false);
    const auto high_alpha = createAlpha(sampling_rate, cutoff_frequency, true);
    EXPECT_NEAR(low_alpha + high_alpha, std::numbers::pi_v<double>, 1e-12);
}

TEST(FilterOperationsTest, TestCreateLowPassFilterAndValidation)
{
    EXPECT_THROW((void) createLowPassFilter(10, 44100.0, 2000.0), std::runtime_error);

    const auto filter = createLowPassFilter(11, 44100.0, 2000.0);
    ASSERT_EQ(filter.size(), 12U);
    for (double v : filter)
    {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST(FilterOperationsTest, TestCreateHighPassFilter)
{
    EXPECT_THROW((void) createHighPassFilter(8, 44100.0, 4000.0), std::runtime_error);

    const auto filter = createHighPassFilter(9, 44100.0, 4000.0);
    ASSERT_EQ(filter.size(), 10U);
}

TEST(FilterOperationsTest, TestStopAndBandStopFilter)
{
    EXPECT_THROW((void) createStopBandFilter(4, 44100.0, 1000.0, 2000.0), std::runtime_error);
    EXPECT_THROW((void) bandStopFilter(6, 44100.0, 1000.0, 2000.0), std::runtime_error);

    const auto stop = createStopBandFilter(9, 44100.0, 1000.0, 2000.0);
    const auto band_stop = bandStopFilter(9, 44100.0, 1000.0, 2000.0);
    EXPECT_EQ(stop.size(), 10U);
    EXPECT_EQ(band_stop.size(), 10U);
}

TEST(FilterOperationsTest, TestTriangularWindowAndApplyWindow)
{
    const auto window = createTriangularWindow(5);
    ASSERT_EQ(window.size(), 6U);
    EXPECT_NEAR(window.front(), 0.0, 1e-12);
    EXPECT_NEAR(window.back(), 0.0, 1e-12);

    std::vector<double> filter = {1, 2, 3, 4, 5, 6};
    applyWindow(filter, window);
    EXPECT_NEAR(filter[0], 0.0, 1e-12);
    EXPECT_NEAR(filter[5], 0.0, 1e-12);

    std::vector<double> bad_window = {1, 2};
    EXPECT_THROW(applyWindow(filter, bad_window), std::runtime_error);
}

TEST(SimpleSignalOperationsTest, TestFindFZeroPeriodSamples)
{
    std::vector<long double> v = {0.5L, 0.3L, 0.2L, 0.3L, 0.2L};
    EXPECT_EQ(findFZeroPeriodSamples(v), 2U);

    std::vector<long double> empty;
    EXPECT_THROW((void) findFZeroPeriodSamples(empty), std::invalid_argument);
}

TEST(SimpleSignalOperationsTest, TestSimpleSignalMutators)
{
    std::vector<double> signal = {100.0, -50.0, 25.0, -12.5};

    doAFineAmplification(signal.data(), static_cast<int>(signal.size()));
    EXPECT_NEAR(std::abs(signal[0]), 32767.0, 1.0);

    halfVolume(signal.data(), static_cast<int>(signal.size()));
    EXPECT_NEAR(std::abs(signal[0]), 16383.5, 2.0);

    silentHalfOfTheSoundTrack(signal.data(), static_cast<int>(signal.size()));
    EXPECT_DOUBLE_EQ(signal[2], 0.0);
    EXPECT_DOUBLE_EQ(signal[3], 0.0);

    std::vector<double> echo_signal = {1.0, 2.0, 3.0, 4.0};
    addEchoes(echo_signal.data(), static_cast<int>(echo_signal.size()));
    // With a very short signal and fixed bouncing time, no echo branch executes.
    EXPECT_DOUBLE_EQ(echo_signal[0], 1.0);
    EXPECT_DOUBLE_EQ(echo_signal[1], 2.0);
    EXPECT_DOUBLE_EQ(echo_signal[2], 3.0);
    EXPECT_DOUBLE_EQ(echo_signal[3], 4.0);
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

namespace
{
int g_callback_calls = 0;
void CountCallback(
    std::vector<double>& signal, size_t& signalLength, uint32_t samplingRate, std::string path)
{
    (void) signal;
    (void) signalLength;
    (void) samplingRate;
    (void) path;
    ++g_callback_calls;
}

void PatchU16At(const std::string& filepath, std::streamoff offset, uint16_t value)
{
    std::fstream fs(filepath, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(fs.is_open());
    fs.seekp(offset);
    fs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void PatchU32At(const std::string& filepath, std::streamoff offset, uint32_t value)
{
    std::fstream fs(filepath, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(fs.is_open());
    fs.seekp(offset);
    fs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
} // namespace

TEST(WavFileTest, ReadRejectsInvalidPaths)
{
    Wav wav;
    EXPECT_THROW(wav.read(""), std::invalid_argument);
    EXPECT_THROW(wav.read("/tmp/this-file-does-not-exist-1234.wav"), std::runtime_error);

    const auto dirpath = std::filesystem::temp_directory_path().string();
    EXPECT_THROW(wav.read(dirpath), std::runtime_error);
}

TEST(WavFileTest, Vector2DWriteSupportsMonoAndStereoAndValidatesInputs)
{
    const auto mono_path =
        std::filesystem::temp_directory_path().string() + "/wave_mono_2d_test.wav";
    const auto stereo_path =
        std::filesystem::temp_directory_path().string() + "/wave_stereo_2d_test.wav";

    Wav wav;
    std::vector<std::vector<float>> mono = {{0.0F, 0.2F, -0.2F, 0.1F}};
    ASSERT_NO_THROW(wav.write(mono_path, mono, 22050));

    Wav mono_reader;
    ASSERT_NO_THROW(mono_reader.read(mono_path));
    EXPECT_FALSE(mono_reader.get_data().empty());

    std::vector<std::vector<float>> stereo = {{0.0F, 0.1F, 0.2F}, {0.0F, -0.1F, -0.2F}};
    ASSERT_NO_THROW(wav.write(stereo_path, stereo, 44100));

    Wav stereo_reader;
    ASSERT_NO_THROW(stereo_reader.read(stereo_path));
    EXPECT_FALSE(stereo_reader.get_data_left().empty());
    EXPECT_FALSE(stereo_reader.get_data_right().empty());

    EXPECT_THROW(
        wav.write(stereo_path, std::vector<std::vector<float>>{}, 44100), std::runtime_error);
    EXPECT_THROW(wav.write(stereo_path,
                     std::vector<std::vector<float>>{{0.0F, 1.0F}, {0.0F, 1.0F}, {0.0F, 1.0F}},
                     44100),
        std::runtime_error);
}

TEST(WavFileTest, ProcessInvokesCallbackForMonoAndStereo)
{
    const auto mono_path =
        std::filesystem::temp_directory_path().string() + "/wave_process_mono_test.wav";
    const auto stereo_path =
        std::filesystem::temp_directory_path().string() + "/wave_process_stereo_test.wav";

    Wav wav;
    wav.write(mono_path, std::vector<float>{0.1F, -0.1F, 0.2F}, 16000);
    wav.write(stereo_path, std::vector<std::vector<float>>{{0.1F, 0.2F}, {-0.1F, -0.2F}}, 16000);

    g_callback_calls = 0;
    Wav mono_reader;
    mono_reader.read(mono_path);
    mono_reader.set_callback_function(&CountCallback);
    mono_reader.process();
    EXPECT_EQ(g_callback_calls, 1);

    Wav stereo_reader;
    stereo_reader.read(stereo_path);
    stereo_reader.set_callback_function(&CountCallback);
    stereo_reader.process();
    EXPECT_EQ(g_callback_calls, 3);
}

TEST(WavFileTest, ReadRejectsNonPCMAndTruncatedFiles)
{
    const auto wav_path = std::filesystem::temp_directory_path().string() + "/wave_patch_test.wav";
    Wav writer;
    writer.write(wav_path, std::vector<float>{0.1F, -0.2F, 0.3F, -0.4F}, 8000);

    PatchU16At(wav_path, 20, static_cast<uint16_t>(3));
    Wav non_pcm_reader;
    EXPECT_THROW(non_pcm_reader.read(wav_path), std::runtime_error);

    writer.write(wav_path, std::vector<float>{0.1F, -0.2F, 0.3F, -0.4F}, 8000);
    PatchU32At(wav_path, 40, static_cast<uint32_t>(999999));
    Wav truncated_reader;
    EXPECT_THROW(truncated_reader.read(wav_path), std::runtime_error);
}

TEST(WavFileTest, ReadRejectsUnsupportedBitDepthFormat)
{
    const auto wav_path =
        std::filesystem::temp_directory_path().string() + "/wave_bad_formatkey_test.wav";
    const uint32_t sample_rate = 8000;

    Wav writer;
    writer.write(wav_path, std::vector<float>{0.1F, -0.2F, 0.3F, -0.4F}, sample_rate);

    // Force waveResolution to 24 by patching byteRate and blockAlign for 1 channel.
    PatchU32At(wav_path, 28, static_cast<uint32_t>(sample_rate * 3U));
    PatchU16At(wav_path, 32, static_cast<uint16_t>(3));

    Wav reader;
    EXPECT_THROW(reader.read(wav_path), std::runtime_error);
}

TEST(WavFileTest, ClassicWriteReadEightBitMono)
{
    const auto wav_path =
        std::filesystem::temp_directory_path().string() + "/wave_8bit_mono_test.wav";
    const std::vector<double> raw = {0.0, 32.0, 127.0, 255.0};

    Wav writer(8000, 8, 1, raw.data(), raw.size());
    ASSERT_NO_THROW(writer.write(wav_path));

    Wav reader;
    ASSERT_NO_THROW(reader.read(wav_path));
    EXPECT_EQ(reader.get_data().size(), raw.size());
}

TEST(WavFileTest, ReadEightBitBranchesViaHeaderPatching)
{
    const auto mono_path =
        std::filesystem::temp_directory_path().string() + "/wave_patch_8bit_mono_test.wav";
    const auto stereo_path =
        std::filesystem::temp_directory_path().string() + "/wave_patch_8bit_stereo_test.wav";
    constexpr uint32_t sample_rate = 8000;

    Wav writer;
    writer.write(mono_path, std::vector<float>{0.1F, -0.2F, 0.3F, -0.4F}, sample_rate);
    PatchU16At(mono_path, 34, static_cast<uint16_t>(8));
    PatchU32At(mono_path, 28, sample_rate);
    PatchU16At(mono_path, 32, static_cast<uint16_t>(1));

    Wav mono_reader;
    ASSERT_NO_THROW(mono_reader.read(mono_path));
    EXPECT_FALSE(mono_reader.get_data().empty());

    writer.write(stereo_path,
        std::vector<std::vector<float>>{{0.1F, 0.2F, -0.3F}, {-0.1F, -0.2F, 0.3F}},
        sample_rate);
    PatchU16At(stereo_path, 34, static_cast<uint16_t>(8));
    PatchU32At(stereo_path, 28, static_cast<uint32_t>(sample_rate * 2U));
    PatchU16At(stereo_path, 32, static_cast<uint16_t>(2));

    Wav stereo_reader;
    ASSERT_NO_THROW(stereo_reader.read(stereo_path));
    EXPECT_FALSE(stereo_reader.get_data_left().empty());
    EXPECT_FALSE(stereo_reader.get_data_right().empty());
}

TEST(WavFileTest, ClassicApiRejectsUnsupportedFormatOnWriteAndProcess)
{
    const auto wav_path =
        std::filesystem::temp_directory_path().string() + "/wave_classic_invalid_format.wav";
    const std::vector<double> raw = {0.0, 1.0, -1.0};

    Wav bad(16000, 24, 1, raw.data(), raw.size());
    EXPECT_THROW(bad.write(wav_path), std::runtime_error);

    bad.set_callback_function(&CountCallback);
    EXPECT_THROW(bad.process(), std::runtime_error);
}

TEST(WavFileTest, ClassicWriteRoundtripFromReadSixteenBitMonoAndStereo)
{
    const auto mono_in =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_16_mono_in.wav";
    const auto mono_out =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_16_mono_out.wav";
    const auto stereo_in =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_16_stereo_in.wav";
    const auto stereo_out =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_16_stereo_out.wav";

    Wav writer;
    writer.write(mono_in, std::vector<float>{0.05F, -0.1F, 0.2F}, 16000);
    writer.write(stereo_in,
        std::vector<std::vector<float>>{{0.1F, 0.2F, 0.3F}, {-0.1F, -0.2F, -0.3F}},
        16000);

    Wav mono_reader;
    ASSERT_NO_THROW(mono_reader.read(mono_in));
    ASSERT_NO_THROW(mono_reader.write(mono_out));

    Wav stereo_reader;
    ASSERT_NO_THROW(stereo_reader.read(stereo_in));
    ASSERT_NO_THROW(stereo_reader.write(stereo_out));
}

TEST(WavFileTest, ClassicWriteRoundtripFromReadEightBitMonoAndStereo)
{
    const auto mono_in =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_8_mono_in.wav";
    const auto mono_out =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_8_mono_out.wav";
    const auto stereo_in =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_8_stereo_in.wav";
    const auto stereo_out =
        std::filesystem::temp_directory_path().string() + "/wave_roundtrip_8_stereo_out.wav";
    constexpr uint32_t sample_rate = 12000;

    Wav writer;
    writer.write(mono_in, std::vector<float>{0.1F, -0.2F, 0.3F, -0.4F}, sample_rate);
    PatchU16At(mono_in, 34, static_cast<uint16_t>(8));
    PatchU32At(mono_in, 28, sample_rate);
    PatchU16At(mono_in, 32, static_cast<uint16_t>(1));

    writer.write(stereo_in,
        std::vector<std::vector<float>>{{0.1F, 0.2F, -0.3F}, {-0.1F, -0.2F, 0.3F}},
        sample_rate);
    PatchU16At(stereo_in, 34, static_cast<uint16_t>(8));
    PatchU32At(stereo_in, 28, static_cast<uint32_t>(sample_rate * 2U));
    PatchU16At(stereo_in, 32, static_cast<uint16_t>(2));

    Wav mono_reader;
    ASSERT_NO_THROW(mono_reader.read(mono_in));
    ASSERT_NO_THROW(mono_reader.write(mono_out));

    Wav stereo_reader;
    ASSERT_NO_THROW(stereo_reader.read(stereo_in));
    ASSERT_NO_THROW(stereo_reader.write(stereo_out));
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
