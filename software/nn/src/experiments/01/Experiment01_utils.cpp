#define USE_MATH_DEFINES

#include "Experiment01_utils.h"
#include "core/wave/audioFeatureExtraction.h" // Include the new header

#include <iostream> // For std::cout

using std::cout;

/**
 * @brief Processa um sujeito específico, carregando e processando seu áudio.
 *
 * @param subject Informações do sujeito a ser processado.
 */
void processSubject(const SubjectInfo& subject)
{
    cout << "Processing subject: " << subject.name << '\n';

    const AudioProcessingParams audioProcessingParams = {.target_sampling_rate = 44100,
                                                         .preemphasis_coefficient = 0.97,
                                                         .frame_duration_ms = 25.0,
                                                         .frame_shift_ms = 10.0,
                                                         .number_of_filters = 24,
                                                         .number_of_cepstrals = 19,
                                                         .delta_window_span = 2};

    constexpr HammingWindowConfig hamming_window_config = {.alpha = 0.54F, .beta = 0.46F};

    constexpr DctConfig dct_config = {.normalization_factor_sqrt = 2.0F,
                                      .filter_index_offset = 0.5F};

    constexpr DeltaConfig delta_config = {.denominator_factor = 2.0F};

    constexpr GeneralConstants general_constants = {.ms_to_seconds_factor = 1000.0F,
                                                    .min_log_energy = 1e-12F,
                                                    .default_sampling_rate = 44100,
                                                    .debug_frame_limit = 5};

    const LoadingAndProcessingParameters loading_params = {
        .audio_params = audioProcessingParams,
        .hamming_window_config = hamming_window_config,
        .dct_config = dct_config,
        .delta_config = delta_config,
        .constants = general_constants};

    std::vector<Tensor> audioWindows = 
        nn::core::wave::loadAndProcessAudio(subject.audio_file_path, loading_params);
    cout << "  - Loaded and processed " << audioWindows.size() << " audio windows.\n";
}