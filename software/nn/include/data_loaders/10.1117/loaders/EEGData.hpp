/**
 * @file EEGData.h
 * @brief Data-only holder types for EEG records in the 10.1117 dataset loaders.
 *
 * The loaders in this folder historically use both a packed matrix representation
 * (`eegSamplesMatrix`) and per-channel vectors (`eegChannels`) for convenience.
 */

#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace nn::dataLoaders
{
struct EEGData
{
    xt::xarray<float> eegSamplesMatrix;
    std::array<int, 3> eegInfo;
    std::vector<Eigen::VectorXf> eegChannels;
};
} // namespace nn::dataLoaders