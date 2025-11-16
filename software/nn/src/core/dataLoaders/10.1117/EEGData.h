#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace nn::dataLoaders
{
struct EEGData
{
    Eigen::MatrixXf eegSamplesMatrix;
    std::array<int, 3> eegInfo;
    std::vector<Eigen::VectorXf> eegChannels;
};
} // namespace nn::dataLoaders