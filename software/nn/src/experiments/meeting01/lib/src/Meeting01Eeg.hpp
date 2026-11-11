#pragma once
// Meeting01Eeg.hpp — EDF (European Data Format, Kemp et al. 1992) reader for
// subject-organized public EEG datasets (PhysioNet Siena Scalp EEG, PhysioNet
// EEG Motor Movement/Imagery), producing the same windowed representation as
// the FSDD/MIT-BIH loaders so the Meeting01 nested-LOSO pipeline can treat an
// EEG recording exactly like a spoken-digit recording or an ECG record.
//
// Scope: reads signal 0 of every .edf file found (recursively) under
// dataset_root, converts to physical units via the header's digital/physical
// min-max, bandpass+notch filters the full recording, slices non-overlapping
// windows, z-score normalises each window. One .edf file == one "recording"; the file's
// immediate PARENT
// DIRECTORY name is the leave-one-group-out unit ("subject") — both target
// datasets (Siena: PNNN/PNNN-M.edf, PhysioNet eegmmidb: SNNN/SNNNRMM.edf)
// organize files one subdirectory per subject, so this needs no
// dataset-specific filename parsing. Assumes signal 0 is an EEG channel:
// verified 2026-09-23 against real downloaded headers — signal 0 is "EEG Fp1"
// for Siena and "Fc5." for eegmmidb, both genuine 10-20-system electrode
// labels, not a status/EKG channel.

#include <filesystem>
#include <string>
#include <vector>

#include "Meeting01DatasetSplit.hpp" // meeting01::WindowMetadata, meeting01::Tensor
#include "tensor/Tensor.hpp"

namespace meeting01
{

class EegWindowDataset
{
   public:
    // Discovers every *.edf under dataset_root (recursive), reads signal 0 of each, bandpass
    // (0.5-40 Hz, the standard EEG band -- see .wiki/Core/DataLoaders.md) + mains-notch
    // filters the FULL recording (nn::utility::bandpass_notch -- must run before windowing,
    // not per-window; see that function's own doc comment for why), then slices it into
    // non-overlapping windows of window_size.
    //
    // notch_hz is the recording site's mains frequency (50 Hz for Siena/Italy, 60 Hz for
    // eegmmidb/US) -- pass <= 0 to disable the notch.
    //
    // sampling_rate is the rate every recording ends up at, so that a window lasts one duration
    // and a frequency means one thing across the dataset. Each file's own rate comes from its
    // header (samples per data record / record duration): the filter runs at that rate, and a
    // file at another one is converted afterwards (nn::utility::resample_bandlimited). This is
    // not a formality: 36 of eegmmidb's 1526 files (S088, S092, S100) are at 128 Hz. Until
    // 2026-11-04 the profile's rate was assumed for every file, so their 40 Hz edge fell at
    // 32 Hz, their 60 Hz notch at 48 Hz and their windows lasted 2.0 s, with no error. Throws
    // std::invalid_argument when window_size <= 0, or when sampling_rate cannot hold the band or
    // sampling_rate/notch_hz form no valid filter (see nn::utility::bandpass_notch), and
    // std::runtime_error when no .edf file is found, a header fails to parse (including its
    // size self-check, or a non-positive record duration) or a signal's digital_max ==
    // digital_min.
    EegWindowDataset(const std::filesystem::path& dataset_root,
        int window_size,
        double sampling_rate,
        double notch_hz);

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
    // How many recordings arrived at another rate and were converted to sampling_rate.
    [[nodiscard]] auto resampled_recordings() const -> int
    {
        return resampled_recordings_;
    }

   private:
    std::vector<nn::Tensor> windows_;
    int resampled_recordings_ = 0;
    std::vector<int> labels_;
    std::vector<WindowMetadata> metadata_;
};

} // namespace meeting01
