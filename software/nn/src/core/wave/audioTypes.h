#ifndef NN_CORE_WAVE_AUDIOTYPES_H
#define NN_CORE_WAVE_AUDIOTYPES_H

#include <cstddef> // For size_t
#include <string>
#include <vector> // For std::vector

#include "core/tensor/Tensor.hpp" // For Tensor

// All struct definitions moved from Experiment01_utils.h

/**
 * @brief Configurações da janela de Hamming.
 */
struct HammingWindowConfig
{
    float alpha;
    float beta;
};

/**
 * @brief Configurações da Transformada Discreta de Cosseno (DCT).
 *
 */
struct DctConfig
{
    float normalization_factor_sqrt;
    float filter_index_offset;
};

/**
 * @brief Configurações para o cálculo dos deltas.
 */
struct DeltaConfig
{
    float denominator_factor;
};

/**
 * @brief Constantes gerais usadas no processamento.
 */
struct GeneralConstants
{
    float ms_to_seconds_factor;
    float min_log_energy;
    int default_sampling_rate;
    size_t debug_frame_limit;
};

/**
 * @brief Parâmetros para o processamento de áudio.
 *
 */
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

/**
 * @brief Parâmetros de carregamento e processamento.
 */
struct LoadingAndProcessingParameters
{
    AudioProcessingParams audio_params; // Changed to by value
    HammingWindowConfig hamming_window_config;
    DctConfig dct_config;
    DeltaConfig delta_config;
    GeneralConstants constants;
};

/**
 * @brief Informações sobre o sujeito a ser processado.
 */
struct SubjectInfo
{
    std::string path;
    std::string name;
    std::string audio_file_path;
    std::string eeg_file_path;
};

/**
 * @brief Configurações para o janelamento do sinal.
 *
 */
struct FramingConfig
{
    int frame_length; // Changed to by value
    int frame_step;   // Changed to by value
    const LoadingAndProcessingParameters& loading_params;
};

/**
 * @brief Configurações para o filtro de banco de dados.
 *
 */
struct FilterbankConfig
{
    Tensor& filterbank;
    std::vector<float>& center_frequencies;
    const LoadingAndProcessingParameters& loading_params;
};

/**
 * @brief Configurações para o cálculo do espectro de potência após o filtro.
 *
 */
struct PowerFilterbankConfig
{
    const Tensor& filterbank;
    const LoadingAndProcessingParameters& loading_params;
};

#endif // NN_CORE_WAVE_AUDIOTYPES_H
