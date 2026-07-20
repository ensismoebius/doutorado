#pragma once

#include <vector>

#include "tensor/Tensor.hpp"

namespace guayaquil
{

using Tensor = nn::Tensor;

struct DatasetSplit
{
    std::vector<Tensor> train_samples;
    std::vector<Tensor> val_samples;
    std::vector<int> val_labels;
};

} // namespace guayaquil
