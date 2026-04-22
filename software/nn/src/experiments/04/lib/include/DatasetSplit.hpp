#pragma once

#include <vector>

#include "nn/tensor/Tensor.hpp"

namespace comparative_autoencoder_experiment
{

using Tensor = nn::Tensor;

struct DatasetSplit
{
    std::vector<Tensor> train_samples;
    std::vector<Tensor> val_samples;
    std::vector<int> val_labels;
};

} // namespace comparative_autoencoder_experiment
