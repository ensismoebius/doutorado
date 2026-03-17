#ifndef NN_DATALOADERS_10_1117_SAMPLEPACKING_HPP
#define NN_DATALOADERS_10_1117_SAMPLEPACKING_HPP

#include <array>

#include "nn/tensor/Tensor.hpp"

auto buildInputTensor(const nn::Tensor& eeg, const nn::Tensor& audio) -> nn::Tensor;
auto mergeAudioAndEEGSignals(const nn::Tensor& eeg_matrix, const nn::Tensor& audio_column)
    -> nn::Tensor;
auto extractEegFromAssembledRows(const nn::Tensor& assembled_inputs) -> nn::Tensor;
auto extractAudioFromAssembledRows(const nn::Tensor& assembled_inputs) -> nn::Tensor;
auto buildTargetTensor(int subject_id, const std::array<int, 3>& eeg_labels, int eeg_index_label)
    -> nn::Tensor;

#endif // NN_DATALOADERS_10_1117_SAMPLEPACKING_HPP