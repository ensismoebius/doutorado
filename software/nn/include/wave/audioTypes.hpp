#ifndef NN_CORE_WAVE_AUDIOTYPES_H
#define NN_CORE_WAVE_AUDIOTYPES_H

#include <cstddef> // For size_t
#include <string>
#include <vector> // For std::vector

#include "tensor/Tensor.hpp" // For Tensor

// Centralized audio pipeline configuration structs.

/**
 * @file audioTypes.h
 * @brief Configuration structs for audio preprocessing/feature extraction.
 *
 * These small POD-like structs hold parameters and scratch buffers shared across
 * feature extraction steps. They are passed by reference to avoid copying large
 * intermediate tensors (e.g., filterbanks).
 */

/**
 * @brief Hamming window coefficients.
 */
struct HammingWindowConfig
{
    float alpha;
    float beta;
};

/**
 * @brief DCT (Discrete Cosine Transform) configuration.
 *
 */
struct DctConfig
{
    float normalization_factor_sqrt;
    float filter_index_offset;
};

/**
 * @brief Delta-feature computation configuration.
 */
struct DeltaConfig
{
    float denominator_factor;
};

/**
 * @brief General constants used by audio feature extraction.
 */
struct GeneralConstants
{
    float ms_to_seconds_factor;
    float min_log_energy;
    int default_sampling_rate;
    size_t debug_frame_limit;
};

/**
 * @brief Audio processing parameters.
 *
 */
struct AudioProcessingParams
{
    int target_sampling_rate;
    float preemphasis_coefficient;
    float frame_duration_ms;
    float frame_shift_ms;
    int number_of_filters;
    long number_of_cepstrals;
    long delta_window_span;
};

/**
 * @brief Aggregated loading and processing parameters.
 */
struct LoadingAndProcessingParameters
{
    AudioProcessingParams audio_params;
    HammingWindowConfig hamming_window_config;
    DctConfig dct_config;
    DeltaConfig delta_config;
    GeneralConstants constants;
};

/**
 * @brief Subject metadata used by dataset demos/pipelines.
 */
struct SubjectInfo
{
    std::string path;
    std::string name;
    std::string audio_file_path;
    std::string eeg_file_path;
};

/**
 * @brief Framing/windowing configuration.
 *
 */
struct FramingConfig
{
    int frame_length;
    int frame_step;
    const LoadingAndProcessingParameters& loading_params;
};

/**
 * @brief Filterbank construction context.
 *
 */
struct FilterbankConfig
{
    nn::Tensor& filterbank;
    std::vector<float>& center_frequencies;
    const LoadingAndProcessingParameters& loading_params;
};

/**
 * @brief Context for applying filterbanks to power spectra.
 *
 */
struct PowerFilterbankConfig
{
    const nn::Tensor& filterbank;
    const LoadingAndProcessingParameters& loading_params;
};

#endif // NN_CORE_WAVE_AUDIOTYPES_H
