/**
 * @file include/nn/dataLoaders/10.1117/datasets/windowed/AudioWindowDataset.hpp
 * @brief Dataset that slices audio recordings into overlapping windows.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "nn/dataLoaders/10.1117/loaders/AudioLoader.h"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/Dataset.hpp"
#include "nn/windowing/WindowSpec.hpp"

class AudioWindowDataset : public Dataset
{
   public:
    explicit AudioWindowDataset(std::vector<SubjectFiles> subjects, nn::windowing::WindowSpec spec);

    [[nodiscard]] auto size() const -> std::size_t override;
    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override;
    void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const override;

    [[nodiscard]] auto spec() const noexcept -> const nn::windowing::WindowSpec&
    {
        return spec_;
    }
    [[nodiscard]] auto windows_per_row() const noexcept -> int
    {
        return windows_per_row_;
    }

   private:
    struct WindowIndex
    {
        std::size_t subject_idx;
        std::size_t row_idx;
        int window_start;
    };

    void ensure_session(std::size_t subject_idx) const;

    std::vector<SubjectFiles> subjects_;
    nn::windowing::WindowSpec spec_;
    int windows_per_row_{0};

    std::vector<WindowIndex> index_table_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>> audio_sessions_;
};
