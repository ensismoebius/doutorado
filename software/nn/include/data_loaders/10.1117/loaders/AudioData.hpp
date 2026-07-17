/**
 * @file AudioData.hpp
 * @brief Lightweight POD type for a single audio record in the 10.1117 dataset loaders.
 *
 * This header intentionally contains *data-only* structures.
 * Downstream code typically converts these into `nn::Tensor` batches.
 */

#pragma once

#include <Eigen/Dense>

namespace nn::dataLoaders
{
struct AudioData
{
    Eigen::VectorXf audioSamples;
    int audioStimulus;
    long eegIndex;
};
} // namespace nn::dataLoaders