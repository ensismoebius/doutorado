/**
 * @file lfcc_pipeline_utils_gtest.cpp
 * @brief Unit tests for the LFCC feature extraction helpers.
 *
 * These tests validate core signal-processing steps (pre-emphasis, framing,
 * windowing, filterbanks, DCT, deltas) with small synthetic inputs.
 */

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "tensor/Tensor.hpp"             // For Tensor
#include "wave/audioFeatureExtraction.hpp" // Include the new audio feature extraction header
#include "wave/audioTypes.hpp"             // Include the new audio types header

using namespace nn::core::wave; // Use the namespace for moved functions

// Helper function to create dummy LoadingAndProcessingParameters
auto create_dummy_loading_params() -> LoadingAndProcessingParameters
{
    AudioProcessingParams audio_params = {.target_sampling_rate = 16000,
        .preemphasis_coefficient = 0.97,
        .frame_duration_ms = 25.0,
        .frame_shift_ms = 10.0,
        .number_of_filters = 24,
        .number_of_cepstrals = 19,
        .delta_window_span = 2};

    HammingWindowConfig hamming_window_config = {.alpha = 0.54F, .beta = 0.46F};

    DctConfig dct_config = {.normalization_factor_sqrt = 2.0F, .filter_index_offset = 0.5F};

    DeltaConfig delta_config = {.denominator_factor = 2.0F};

    GeneralConstants constants = {.ms_to_seconds_factor = 1000.0F,
        .min_log_energy = 1e-12F,
        .default_sampling_rate = 16000, // Should match target_sampling_rate for consistency
        .debug_frame_limit = 5};

    return {.audio_params = audio_params,
        .hamming_window_config = hamming_window_config,
        .dct_config = dct_config,
        .delta_config = delta_config,
        .constants = constants};
}

// Test fixture for common setup if needed
class LfccPipelineUtilsTest : public ::testing::Test
{
   protected:
    LoadingAndProcessingParameters dummy_loading_params;

    void SetUp() override
    {
        dummy_loading_params = create_dummy_loading_params();
    }
};

// Test for pre_emphasis_inplace
TEST_F(LfccPipelineUtilsTest, PreEmphasisInplace)
{
    std::vector<float> signal = {1.0F, 2.0F, 3.0F, 4.0F};
    float coefficient = 0.97F;
    pre_emphasis_inplace(signal, coefficient);

    // Expected: signal[0] = 1.0
    // signal[1] = 2.0 - 0.97 * 1.0 = 1.03
    // signal[2] = 3.0 - 0.97 * 2.0 = 1.06
    // signal[3] = 4.0 - 0.97 * 3.0 = 1.09
    ASSERT_NEAR(signal[0], 1.0F, 1e-6);
    ASSERT_NEAR(signal[1], 1.03F, 1e-6);
    ASSERT_NEAR(signal[2], 1.06F, 1e-6);
    ASSERT_NEAR(signal[3], 1.09F, 1e-6);
}

TEST_F(LfccPipelineUtilsTest, PreEmphasisInplace_ZeroCoefficient)
{
    std::vector<float> signal = {1.0F, 2.0F, 3.0F};
    float coefficient = 0.0F;
    pre_emphasis_inplace(signal, coefficient);
    ASSERT_NEAR(signal[0], 1.0F, 1e-6);
    ASSERT_NEAR(signal[1], 2.0F, 1e-6);
    ASSERT_NEAR(signal[2], 3.0F, 1e-6);
}

TEST_F(LfccPipelineUtilsTest, PreEmphasisInplace_EmptySignal)
{
    std::vector<float> signal = {};
    float coefficient = 0.97F;
    pre_emphasis_inplace(signal, coefficient);
    ASSERT_TRUE(signal.empty());
}

TEST_F(LfccPipelineUtilsTest, PreEmphasisInplace_SingleElementSignal)
{
    std::vector<float> signal = {5.0F};
    float coefficient = 0.97F;
    pre_emphasis_inplace(signal, coefficient);
    ASSERT_NEAR(signal[0], 5.0F, 1e-6); // First element should remain unchanged
}

// Test for framing_and_window
TEST_F(LfccPipelineUtilsTest, FramingAndWindow_Basic)
{
    std::vector<float> signal(100, 1.0F); // 100 samples, all 1.0

    // Let's adjust dummy_loading_params for a smaller signal
    dummy_loading_params.audio_params.frame_duration_ms = 10.0; // 10ms -> 160 samples
    dummy_loading_params.audio_params.frame_shift_ms = 5.0;     // 5ms -> 80 samples

    FramingConfig framing_context = {.frame_length = 0, // Will be calculated
        .frame_step = 0,                                // Will be calculated
        .loading_params = dummy_loading_params};

    auto frames = framing_and_window(signal, framing_context);

    int expected_frame_length = static_cast<int>(roundf(10.0 * 16000.0 / 1000.0)); // 160
    int expected_frame_step = static_cast<int>(roundf(5.0 * 16000.0 / 1000.0));    // 80

    ASSERT_EQ(framing_context.frame_length, expected_frame_length);
    ASSERT_EQ(framing_context.frame_step, expected_frame_step);

    // signal_length = 100
    // number_of_frames = 1 + max(0, (100 - 160) / 80) = 1 + 0 = 1
    ASSERT_EQ(frames.size(), 1);
    ASSERT_EQ(frames[0].size(), expected_frame_length);

    // Check windowing effect (all 1.0 signal should be multiplied by window function)
    // Hamming window: 0.54 - 0.46 * cos(2πn / (N-1))
    // For N=160, n=0: 0.54 - 0.46 * cos(0) = 0.54 - 0.46 = 0.08
    // For N=160, n=79 (middle): 0.54 - 0.46 * cos(pi) = 0.54 + 0.46 = 1.0
    // For N=160, n=159 (end): 0.54 - 0.46 * cos(2pi) = 0.54 - 0.46 = 0.08
    ASSERT_NEAR(frames[0][0], 0.08F, 1e-6);
    ASSERT_NEAR(frames[0][expected_frame_length / 2], 1.0F, 1e-4);
    ASSERT_NEAR(frames[0][expected_frame_length - 1], 0.0F, 1e-6);
}

TEST_F(LfccPipelineUtilsTest, FramingAndWindow_EmptySignal)
{
    std::vector<float> signal = {};
    FramingConfig framing_context = {
        .frame_length = 0, .frame_step = 0, .loading_params = dummy_loading_params};
    auto frames = framing_and_window(signal, framing_context);
    ASSERT_TRUE(frames.empty());
}

TEST_F(LfccPipelineUtilsTest, FramingAndWindow_MultipleFrames)
{
    std::vector<float> signal(500, 1.0F);                       // 500 samples, all 1.0
    dummy_loading_params.audio_params.frame_duration_ms = 25.0; // 400 samples
    dummy_loading_params.audio_params.frame_shift_ms = 10.0;    // 160 samples

    FramingConfig framing_context = {
        .frame_length = 0, .frame_step = 0, .loading_params = dummy_loading_params};

    auto frames = framing_and_window(signal, framing_context);

    int expected_frame_length = 400;
    int expected_frame_step = 160;

    ASSERT_EQ(framing_context.frame_length, expected_frame_length);
    ASSERT_EQ(framing_context.frame_step, expected_frame_step);

    // signal_length = 500
    // number_of_frames = 1 + max(0, (500 - 400) / 160) = 1 + 0 = 1
    // This calculation is based on the original code's logic.
    ASSERT_EQ(frames.size(), 1);
    ASSERT_EQ(frames[0].size(), expected_frame_length);
}

// Test for rfft_power
TEST_F(LfccPipelineUtilsTest, RFFTPower_BasicSineWave)
{
    // Test with a simple sine wave to verify FFT and power spectrum
    int sample_rate = 16000;
    float frequency = 1000.0F; // 1 kHz sine wave
    int fft_points = 512;
    std::vector<float> sine_wave(fft_points);
    for (int i = 0; i < fft_points; ++i)
    {
        sine_wave[i] = sinf(2 * std::numbers::pi_v<float> * frequency * static_cast<float>(i) /
                            static_cast<float>(sample_rate));
    }

    std::vector<std::vector<float>> frames = {sine_wave};
    nn::Tensor power_spectrum = rfft_power(frames, fft_points);

    ASSERT_EQ(power_spectrum.rows(), 1);
    ASSERT_EQ(power_spectrum.cols(), (fft_points / 2) + 1);

    // Expect a peak at the corresponding frequency bin
    // Frequency bin = frequency * fft_points / sample_rate
    int expected_bin = static_cast<int>(
        roundf(frequency * static_cast<float>(fft_points) / static_cast<float>(sample_rate)));
    // Check that the expected bin has a high value, and others are low
    for (int i = 0; i < power_spectrum.cols(); ++i)
    {
        if (i == expected_bin)
        {
            ASSERT_GT(power_spectrum(0, i), 0.1F); // Should be a significant peak
        }
        else
        {
            // Allow some leakage, but generally much lower
            ASSERT_LT(power_spectrum(0, i), 0.01F);
        }
    }
}

TEST_F(LfccPipelineUtilsTest, RFFTPower_EmptyFrames)
{
    std::vector<std::vector<float>> frames = {};
    int fft_points = 512;
    ASSERT_THROW(rfft_power(frames, fft_points), std::invalid_argument);
}

// Test for build_linear_filterbank
TEST_F(LfccPipelineUtilsTest, BuildLinearFilterbank_Basic)
{
    int fft_points = 512;
    nn::Tensor filterbank_test;                 // Declare local Tensor
    std::vector<float> center_frequencies_test; // Declare local vector

    LoadingAndProcessingParameters custom_loading_params = dummy_loading_params;
    custom_loading_params.audio_params.number_of_filters = 10;
    custom_loading_params.audio_params.target_sampling_rate = 16000;
    custom_loading_params.constants.default_sampling_rate = 16000;

    FilterbankConfig filterbank_context = {.filterbank = filterbank_test,
        .center_frequencies = center_frequencies_test,
        .loading_params = custom_loading_params};

    build_linear_filterbank(fft_points, filterbank_context);

    ASSERT_EQ(filterbank_test.rows(), 10);
    ASSERT_EQ(filterbank_test.cols(), (fft_points / 2) + 1);

    // Check some properties of the filterbank
    // The sum of values in each filter should be positive
    for (int i = 0; i < 10; ++i)
    {
        float sum = 0.0F;
        for (int j = 0; j < (fft_points / 2) + 1; ++j)
        {
            sum += filterbank_test(i, j);
        }
        ASSERT_GT(sum, 0.0F);
    }

    // Check that center frequencies are increasing
    for (size_t i = 0; i < center_frequencies_test.size() - 1; ++i)
    {
        ASSERT_LT(center_frequencies_test[i], center_frequencies_test[i + 1]);
    }
}

// Test for dot_power_filterbank
TEST_F(LfccPipelineUtilsTest, DotPowerFilterbank_Basic)
{
    // Create a dummy power spectrum (e.g., all 1.0)
    nn::Tensor power_spectrum(2, 257); // 2 frames, 257 bins (for fft_points = 512)
    power_spectrum.setConstant(1.0F);

    // Create a dummy filterbank (e.g., a simple triangular filter)
    nn::Tensor filterbank_test;                 // Declare local Tensor
    std::vector<float> center_frequencies_test; // Declare local vector

    LoadingAndProcessingParameters custom_loading_params = dummy_loading_params;
    custom_loading_params.audio_params.number_of_filters = 2;
    custom_loading_params.audio_params.target_sampling_rate = 16000;
    custom_loading_params.constants.default_sampling_rate = 16000;

    FilterbankConfig filterbank_context = {.filterbank = filterbank_test,
        .center_frequencies = center_frequencies_test,
        .loading_params = custom_loading_params};
    build_linear_filterbank(512, filterbank_context); // Build a real filterbank

    PowerFilterbankConfig power_filterbank_context = {
        .filterbank = filterbank_test, .loading_params = custom_loading_params};

    nn::Tensor log_energies = dot_power_filterbank(power_spectrum, power_filterbank_context);

    ASSERT_EQ(log_energies.rows(), 2);
    ASSERT_EQ(log_energies.cols(), 2);

    // Since power_spectrum is all 1.0, log_energies should be log(sum of filterbank values)
    // The sum of filterbank values will be positive, so log_energies should be positive
    for (int i = 0; i < log_energies.rows(); ++i)
    {
        for (int j = 0; j < log_energies.cols(); ++j)
        {
            ASSERT_GT(log_energies(i, j),
                logf(dummy_loading_params.constants.min_log_energy) -
                    1.0); // Should be greater than min_log_energy after log
        }
    }
}

// Test for dct2
TEST_F(LfccPipelineUtilsTest, DCT2_Basic)
{
    // Create dummy log energies
    nn::Tensor log_energies(2, 10); // 2 frames, 10 filter energies
    log_energies.setConstant(1.0F);

    LoadingAndProcessingParameters loading_params = dummy_loading_params;
    loading_params.audio_params.number_of_cepstrals = 5; // Request 5 cepstral coefficients

    nn::Tensor cepstral_coeff = dct2(log_energies, loading_params);

    ASSERT_EQ(cepstral_coeff.rows(), 2);
    ASSERT_EQ(cepstral_coeff.cols(), 5);

    // Check that values are not NaN or Inf
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    for (int i = 0; i < cepstral_coeff.rows(); ++i)
    {
        for (int j = 0; j < cepstral_coeff.cols(); ++j)
        {
            // NOLINTNEXTLINE(bugprone-use-of-uninitialized-value)
            ASSERT_FALSE(std::isnan(cepstral_coeff(i, j)));
            // NOLINTNEXTLINE(bugprone-use-of-uninitialized-value)
            ASSERT_FALSE(std::isinf(cepstral_coeff(i, j)));
        }
    }
#pragma GCC diagnostic pop
}

// Test for compute_deltas
TEST_F(LfccPipelineUtilsTest, ComputeDeltas_Basic)
{
    // Create a simple feature matrix
    nn::Tensor features(5, 3); // 5 frames, 3 features
    // Use << operator (fills Column-Major)
    features << 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F,
        14.0F, 15.0F;

    LoadingAndProcessingParameters loading_params = dummy_loading_params;
    loading_params.audio_params.delta_window_span = 1; // Simple delta calculation

    nn::Tensor delta_features = compute_deltas(features, loading_params);

    ASSERT_EQ(delta_features.rows(), 5);
    ASSERT_EQ(delta_features.cols(), 3);

    // Expected delta for first frame (using padding):
    // Matrix is filled Column-Major:
    // Col 0: 1, 2, 3, 4, 5
    // Col 1: 6, 7, 8, 9, 10
    // Col 2: 11, 12, 13, 14, 15
    //
    // Row 0: 1, 6, 11
    // Row 1: 2, 7, 12
    // Row 2: 3, 8, 13
    //
    // Frame 0 delta: (Row 1 - Row 0) / 2
    // Col 0: (2 - 1) / 2 = 0.5
    // Col 1: (7 - 6) / 2 = 0.5
    // Col 2: (12 - 11) / 2 = 0.5
    ASSERT_NEAR(delta_features(0, 0), 0.5F, 1e-6);
    ASSERT_NEAR(delta_features(0, 1), 0.5F, 1e-6);
    ASSERT_NEAR(delta_features(0, 2), 0.5F, 1e-6);

    // Expected delta for second frame:
    // Frame 1 delta: (Row 2 - Row 0) / 2
    // Col 0: (3 - 1) / 2 = 1.0
    // Col 1: (8 - 6) / 2 = 1.0
    // Col 2: (13 - 11) / 2 = 1.0
    ASSERT_NEAR(delta_features(1, 0), 1.0F, 1e-6);
    ASSERT_NEAR(delta_features(1, 1), 1.0F, 1e-6);
    ASSERT_NEAR(delta_features(1, 2), 1.0F, 1e-6);
}

TEST_F(LfccPipelineUtilsTest, ComputeDeltas_EmptyFeatures)
{
    nn::Tensor features(0, 3);
    LoadingAndProcessingParameters loading_params = dummy_loading_params;
    loading_params.audio_params.delta_window_span = 1;
    nn::Tensor delta_features = compute_deltas(features, loading_params);
    ASSERT_TRUE(delta_features.rows() == 0);
    ASSERT_TRUE(delta_features.cols() == 0);
}