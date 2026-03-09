#ifndef EXEC_LOADINGDATA_PROTOCOL101117DATASET_HPP
#define EXEC_LOADINGDATA_PROTOCOL101117DATASET_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/Dataset.hpp"
#include "subject_discovery.hpp"

class Protocol101117Dataset : public Dataset
{
   public:
    explicit Protocol101117Dataset(std::vector<SubjectFiles> subjects);

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
};

#endif // EXEC_LOADINGDATA_PROTOCOL101117DATASET_HPP
