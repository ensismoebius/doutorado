#ifndef EXEC_LOADINGDATA_SAMPLEPACKING_HPP
#define EXEC_LOADINGDATA_SAMPLEPACKING_HPP

#include <array>

#include "nn/tensor/Tensor.hpp"

auto buildInputTensor(const nn::Tensor& eeg, const nn::Tensor& audio) -> nn::Tensor;
auto buildTargetTensor(int subject_id, const std::array<int, 3>& eeg_labels, int eeg_index_label)
    -> nn::Tensor;

#endif // EXEC_LOADINGDATA_SAMPLEPACKING_HPP
