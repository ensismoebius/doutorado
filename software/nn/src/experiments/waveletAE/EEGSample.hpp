/**
 * @file src/experiments/waveletAE/EEGSample.hpp
 * @brief EEGSample struct (extracted from WaveletAEData.hpp).
 */

#ifndef NN_EXPERIMENTS_02_EEGSAMPLE_HPP
#define NN_EXPERIMENTS_02_EEGSAMPLE_HPP

#include <vector>

struct EEGSample
{
    std::vector<std::vector<double>> channels;
    int modality;
    int stimulus;
    int artifacts;
};

#endif // NN_EXPERIMENTS_02_EEGSAMPLE_HPP
