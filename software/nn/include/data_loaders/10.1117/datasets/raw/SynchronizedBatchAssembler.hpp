/**
 * @file include/data_loaders/10.1117/datasets/raw/SynchronizedBatchAssembler.hpp
 * @brief SynchronizedBatchAssembler (migrated into datasets/raw layout).
 */

#ifndef NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP
#define NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP

#include <memory>
#include <vector>

#include "data_loaders/10.1117/datasets/raw/RowRequest.hpp"
#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"
#include "tensor/Tensor.hpp"

class SynchronizedBatchAssembler
{
   public:
    static void assembleGrouped(const std::vector<std::vector<RowRequest>>& grouped,
        const std::vector<SubjectFiles>& subjects,
        const std::vector<std::unique_ptr<nn::dataLoaders::AudioSession>>& audio_sessions,
        const std::vector<std::unique_ptr<nn::dataLoaders::EEGSession>>& eeg_sessions,
        nn::Tensor& inputs,
        nn::Tensor& targets);
};

#endif // NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP
