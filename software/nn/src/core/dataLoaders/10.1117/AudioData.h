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