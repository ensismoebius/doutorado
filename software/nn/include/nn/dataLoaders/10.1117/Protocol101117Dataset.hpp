#ifndef NN_DATALOADERS_10_1117_PROTOCOL101117DATASET_HPP
#define NN_DATALOADERS_10_1117_PROTOCOL101117DATASET_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/SubjectDiscovery.hpp"
#include "nn/dataLoaders/Dataset.hpp"

struct Protocol101117Sample
{
    nn::Tensor inputs;
    nn::Tensor targets;
    nn::Tensor audio;
    nn::Tensor eeg;
    bool concatenated = true;
};

class Protocol101117Dataset : public Dataset
{
   public:
    explicit Protocol101117Dataset(std::vector<SubjectFiles> subjects,
                                   bool concatenate_modalities = true);

    void set_concatenate_modalities(bool concatenate_modalities);
    [[nodiscard]] auto concatenate_modalities() const -> bool;

    [[nodiscard]] auto get_sample(std::size_t idx,
                                  std::optional<bool> concatenate_override = std::nullopt) const
        -> Protocol101117Sample;

    [[nodiscard]] auto size() const -> std::size_t override;
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;
    [[nodiscard]] auto collate(const std::vector<std::size_t>& indices) const -> Batch override;

    [[nodiscard]] auto subjects() const -> const std::vector<SubjectFiles>&;

   private:
    auto loadSampleByLocalIndex(std::size_t subject_index, std::size_t audio_row) const -> Batch;
    void ensureSessions(std::size_t subject_index) const;

    std::vector<SubjectFiles> subjects_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>> audio_sessions_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>> eeg_sessions_;
    std::vector<std::size_t> prefix_audio_row_offsets_;
    bool concatenate_modalities_ = true;
};

#endif // NN_DATALOADERS_10_1117_PROTOCOL101117DATASET_HPP