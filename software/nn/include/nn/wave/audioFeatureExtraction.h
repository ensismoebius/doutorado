#ifndef NN_CORE_WAVE_AUDIOFEATUREEXTRACTION_H
#define NN_CORE_WAVE_AUDIOFEATUREEXTRACTION_H

#include <vector>

#include "nn/tensor/Tensor.hpp"
#include "nn/wave/audioTypes.h" // Include the new audio types header

namespace nn::core::wave
{

// Function declarations moved from Experiment01_utils.cpp
void pre_emphasis_inplace(std::vector<float>& signal, float coefficient);
auto framing_and_window(const std::vector<float>& signal, FramingConfig& context)
    -> std::vector<std::vector<float>>;
auto rfft_power(const std::vector<std::vector<float>>& frames, int fft_points) -> nn::Tensor;
void build_linear_filterbank(int fft_points, FilterbankConfig& context);
auto dot_power_filterbank(const nn::Tensor& power_spectrum, const PowerFilterbankConfig& context)
    -> nn::Tensor;
auto dct2(const nn::Tensor& log_energies, const LoadingAndProcessingParameters& loading_params)
    -> nn::Tensor;
auto compute_deltas(const nn::Tensor& features,
                    const LoadingAndProcessingParameters& loading_params) -> nn::Tensor;

// Windowing functions
auto hanning_window(int length) -> std::vector<double>;
auto apply_window(const std::vector<double>& signal, const std::vector<double>& window)
    -> std::vector<double>;
} // namespace nn::core::wave

#endif // NN_CORE_WAVE_AUDIOFEATUREEXTRACTION_H
