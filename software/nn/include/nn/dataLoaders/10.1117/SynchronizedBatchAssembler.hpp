#ifndef NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP
#define NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/SubjectDiscovery.hpp"
#include "nn/tensor/Tensor.hpp"

struct RowRequest
{
    std::size_t batch_row;
    std::size_t local_audio_row;
};

class SynchronizedBatchAssembler
{
   public:
    static void assembleGrouped(
        const std::vector<std::vector<RowRequest>>& grouped,
        const std::vector<SubjectFiles>& subjects,
        const std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>>& audio_sessions,
        const std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,
        nn::Tensor& inputs, nn::Tensor& targets);
};

#endif // NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP