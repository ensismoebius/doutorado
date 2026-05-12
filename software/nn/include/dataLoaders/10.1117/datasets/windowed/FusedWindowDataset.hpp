/**
 * @file include/nn/dataLoaders/10.1117/datasets/windowed/FusedWindowDataset.hpp
 * @brief Dataset that fuses synchronised EEG + audio windows into one input vector.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dataLoaders/10.1117/loaders/AudioLoader.h"
#include "dataLoaders/10.1117/loaders/EEGLoader.h"
#include "dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "dataLoaders/datasets/Dataset.hpp"
#include "windowing/WindowSpec.hpp"

class FusedWindowDataset : public Dataset
{
   public:
    explicit FusedWindowDataset(std::vector<SubjectFiles> subjects,
        nn::windowing::WindowSpec eeg_spec,
        nn::windowing::WindowSpec audio_spec);

    [[nodiscard]] auto size() const -> std::size_t override;
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;
    void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const override;
    void print(IDatasetPrinter& printer) const override;

    [[nodiscard]] auto eeg_spec() const noexcept -> const nn::windowing::WindowSpec&
    {
        return eeg_spec_;
    }
    [[nodiscard]] auto audio_spec() const noexcept -> const nn::windowing::WindowSpec&
    {
        return audio_spec_;
    }
    [[nodiscard]] auto windows_per_pair() const noexcept -> int
    {
        return windows_per_pair_;
    }
    [[nodiscard]] auto input_features() const noexcept -> int
    {
        return input_features_;
    }

   private:
    struct WindowIndex
    {
        std::size_t subject_idx;
        std::size_t audio_row;
        int window_k;
    };

    void ensure_sessions(std::size_t subject_idx) const;

    std::vector<SubjectFiles> subjects_;
    nn::windowing::WindowSpec eeg_spec_;
    nn::windowing::WindowSpec audio_spec_;
    int windows_per_pair_{0};
    int input_features_{0};

    std::vector<WindowIndex> index_table_;

    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>> audio_sessions_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>> eeg_sessions_;
};
