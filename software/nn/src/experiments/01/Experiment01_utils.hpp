#ifndef EXPERIMENT01_UTILS_HPP
#define EXPERIMENT01_UTILS_HPP

#include <cstddef> // For size_t
#include <string>
#include <vector> // For std::vector

#include "core/tensor/Tensor.hpp" // For Tensor

struct HammingWindowConfig
{
    float alpha;
    float beta;
};

struct DctConfig
{
    float normalization_factor_sqrt;
    float filter_index_offset;
};

struct DeltaConfig
{
    float denominator_factor;
};

struct GeneralConstants
{
    float ms_to_seconds_factor;
    float min_log_energy;
    int default_sampling_rate;
    size_t debug_frame_limit;
};

struct AudioProcessingParams
{
    int target_sampling_rate;
    float preemphasis_coefficient;
    float frame_duration_ms;
    float frame_shift_ms;
    int number_of_filters;
    int number_of_cepstrals;
    long delta_window_span;
};

struct LoadingAndProcessingParameters
{
    const AudioProcessingParams& audio_params;
    HammingWindowConfig hamming_window_config;
    DctConfig dct_config;
    DeltaConfig delta_config;
    GeneralConstants constants;
};

struct SubjectInfo
{
    std::string path;
    std::string name;
    std::string audio_file_path;
    std::string eeg_file_path;
};

struct FramingContext
{
    int& frame_length;
    int& frame_step;
    const LoadingAndProcessingParameters& loading_params;
};

struct FilterbankContext
{
    Tensor& filterbank;
    std::vector<float>& center_frequencies;
    const LoadingAndProcessingParameters& loading_params;
};

struct PowerFilterbankContext
{
    const Tensor& filterbank;
    const LoadingAndProcessingParameters& loading_params;
};

void processSubject(const SubjectInfo& subject);

#endif // EXPERIMENT01_UTILS_HPP
