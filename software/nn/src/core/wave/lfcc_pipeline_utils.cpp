/**
 * @file lfcc_pipeline_utils.cpp
 * @brief Implementation of the LFCC extraction pipeline.
 */

#define USE_MATH_DEFINES

#include "nn/wave/lfcc_pipeline_utils.h"

#include <fftw3.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/wave/audioFeatureExtraction.h"

using nn::dataLoaders::loadAudioFromMat;
using std::size_t;
using std::vector;

auto load_and_process_audio(const std::string& audio_file_path,
                            const LoadingAndProcessingParameters& loading_params)
    -> std::vector<nn::Tensor>
{
    auto [audio_samples, audio_stimulus, eeg_index] = loadAudioFromMat(audio_file_path, 0);
    (void)audio_stimulus;
    (void)eeg_index;

    audio_samples = audio_samples.transpose();
    vector<float> input_data(audio_samples.data_ptr(),
                             audio_samples.data_ptr() + audio_samples.size());

    vector<vector<float>> frames;
    nn::Tensor power_spectrum;
    nn::Tensor filterbank_local;
    vector<float> center_frequencies_local;
    nn::Tensor log_energies;
    nn::Tensor cepstral_coeff;
    nn::Tensor delta_coeff;
    nn::Tensor delta_delta_coeff;

    nn::core::wave::pre_emphasis_inplace(
        input_data, loading_params.audio_params.preemphasis_coefficient);

    FramingConfig framing_context = {
        .frame_length = 0,
        .frame_step = 0,
        .loading_params = loading_params,
    };
    frames = nn::core::wave::framing_and_window(input_data, framing_context);

    power_spectrum = nn::core::wave::rfft_power(frames, framing_context.frame_length);

    FilterbankConfig filterbank_context = {
        .filterbank = filterbank_local,
        .center_frequencies = center_frequencies_local,
        .loading_params = loading_params,
    };
    nn::core::wave::build_linear_filterbank(framing_context.frame_length, filterbank_context);

    PowerFilterbankConfig power_filterbank_context = {
        .filterbank = filterbank_local,
        .loading_params = loading_params,
    };
    log_energies = nn::core::wave::dot_power_filterbank(power_spectrum, power_filterbank_context);

    cepstral_coeff = nn::core::wave::dct2(log_energies, loading_params);
    delta_coeff = nn::core::wave::compute_deltas(cepstral_coeff, loading_params);
    delta_delta_coeff = nn::core::wave::compute_deltas(delta_coeff, loading_params);

    return {cepstral_coeff, delta_coeff, delta_delta_coeff};
}

void process_subject(const SubjectInfo& subject)
{
    std::cout << "Processing subject: " << subject.name << '\n';

    const AudioProcessingParams audio_processing_params = {
        .target_sampling_rate = 44100,
        .preemphasis_coefficient = 0.97F,
        .frame_duration_ms = 25.0,
        .frame_shift_ms = 10.0,
        .number_of_filters = 24,
        .number_of_cepstrals = 19,
        .delta_window_span = 2,
    };

    constexpr HammingWindowConfig hamming_window_config = {.alpha = 0.54F, .beta = 0.46F};
    constexpr DctConfig dct_config = {.normalization_factor_sqrt = 2.0F, .filter_index_offset = 0.5F};
    constexpr DeltaConfig delta_config = {.denominator_factor = 2.0F};
    constexpr GeneralConstants general_constants = {
        .ms_to_seconds_factor = 1000.0F,
        .min_log_energy = 1e-12F,
        .default_sampling_rate = 44100,
        .debug_frame_limit = 5,
    };

    const LoadingAndProcessingParameters loading_params = {
        .audio_params = audio_processing_params,
        .hamming_window_config = hamming_window_config,
        .dct_config = dct_config,
        .delta_config = delta_config,
        .constants = general_constants,
    };

    std::vector<nn::Tensor> audio_windows =
        load_and_process_audio(subject.audio_file_path, loading_params);
    std::cout << "  - Loaded and processed " << audio_windows.size() << " audio windows.\n";
}