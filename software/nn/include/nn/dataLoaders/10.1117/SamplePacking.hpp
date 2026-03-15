#ifndef NN_DATALOADERS_10_1117_SAMPLEPACKING_HPP
#define NN_DATALOADERS_10_1117_SAMPLEPACKING_HPP

#include <array>

#include "nn/tensor/Tensor.hpp"

auto buildInputTensor(const nn::Tensor& eeg, const nn::Tensor& audio) -> nn::Tensor;
auto buildInputTensorFromFlattenedRows(const nn::Tensor& eeg_row, const nn::Tensor& audio_row)
    -> nn::Tensor;
auto buildTargetTensor(int subject_id, const std::array<int, 3>& eeg_labels, int eeg_index_label)
    -> nn::Tensor;

#endif // NN_DATALOADERS_10_1117_SAMPLEPACKING_HPP