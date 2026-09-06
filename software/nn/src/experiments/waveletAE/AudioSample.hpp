/**
 * @file src/experiments/waveletAE/AudioSample.hpp
 * @brief AudioSample struct (extracted from WaveletAEData.hpp).
 */

#ifndef NN_EXPERIMENTS_02_AUDIOSAMPLE_HPP
#define NN_EXPERIMENTS_02_AUDIOSAMPLE_HPP

#include <vector>

struct AudioSample
{
    std::vector<double> signal;
    int stimulus;
    int eeg_index;
};

#endif // NN_EXPERIMENTS_02_AUDIOSAMPLE_HPP
