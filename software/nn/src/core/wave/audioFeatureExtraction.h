#ifndef NN_CORE_WAVE_AUDIOFEATUREEXTRACTION_H
#define NN_CORE_WAVE_AUDIOFEATUREEXTRACTION_H

#include <vector>

#include "core/tensor/Tensor.hpp"
#include "core/wave/audioTypes.h" // Include the new audio types header

namespace nn::core::wave
{

// Function declarations moved from Experiment01_utils.cpp
void pre_emphasis_inplace(std::vector<float>& signal, float coefficient);
auto framing_and_window(const std::vector<float>& signal, FramingConfig& context)
    -> std::vector<std::vector<float>>;
auto rfft_power(const std::vector<std::vector<float>>& frames, int fft_points) -> Tensor;
void build_linear_filterbank(int fft_points, FilterbankConfig& context);
auto dot_power_filterbank(const Tensor& power_spectrum, const PowerFilterbankConfig& context)
    -> Tensor;
auto dct2(const Tensor& log_energies, const LoadingAndProcessingParameters& loading_params)
    -> Tensor;
auto compute_deltas(const Tensor& features, const LoadingAndProcessingParameters& loading_params)
    -> Tensor;
} // namespace nn::core::wave

#endif // NN_CORE_WAVE_AUDIOFEATUREEXTRACTION_H
