#pragma once
// GuayaquilMitBih.hpp — minimal WFDB reader for the MIT-BIH Arrhythmia Database
// (PhysioNet mitdb), producing the same windowed representation as the FSDD
// loader so the Guayaquil nested-LOSO pipeline can treat ECG records exactly like
// spoken-digit recordings.
//
// Scope: format-212 records only (every mitdb record is 212). Reads signal 0
// (the MLII lead on all but a couple of records), converts to physical units via
// the header gain/baseline, slices non-overlapping windows, and z-score
// normalises each window in place.
//
// One record (.hea/.dat pair) == one "recording" == one leave-one-group-out
// unit; a handful of mitdb records share a subject, noted as a limitation.

#include <filesystem>
#include <string>
#include <vector>

#include "GuayaquilDatasetSplit.hpp" // guayaquil::WindowMetadata, guayaquil::Tensor
#include "tensor/Tensor.hpp"

namespace guayaquil
{

class MitBihWindowDataset
{
   public:
    // Discovers every *.hea under dataset_root, reads the matching .dat (format
    // 212), and slices signal 0 into non-overlapping windows of window_size.
    //
    // Throws std::invalid_argument if window_size <= 0.
    // Throws std::runtime_error if no records are found, a header is not
    // format 212, or a .dat size is inconsistent with its header.
    MitBihWindowDataset(const std::filesystem::path& dataset_root, int window_size);

    [[nodiscard]] auto windows() const -> const std::vector<nn::Tensor>&
    {
        return windows_;
    }
    [[nodiscard]] auto labels() const -> const std::vector<int>&
    {
        return labels_;
    }
    [[nodiscard]] auto metadata() const -> const std::vector<WindowMetadata>&
    {
        return metadata_;
    }
    [[nodiscard]] auto size() const -> std::size_t
    {
        return windows_.size();
    }

   private:
    std::vector<nn::Tensor> windows_;
    std::vector<int> labels_;
    std::vector<WindowMetadata> metadata_;
};

} // namespace guayaquil
