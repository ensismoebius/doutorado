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
    // eegmmidb/US) -- pass <= 0 to disable the notch. sampling_rate must be the EDF's real
    // native rate (Siena 512 Hz, eegmmidb 160 Hz); it is NOT read from the EDF header itself
    // (the header's own per-signal rate isn't parsed by this loader), so a wrong value here
    // silently mis-targets the filter's cutoffs without any way for this constructor to catch
    // it -- verify against Meeting01Config::DatasetSource::sample_rate for the dataset in use.
    //
    // Throws std::invalid_argument if window_size <= 0 or if sampling_rate/notch_hz don't form
    // a valid filter configuration (see nn::utility::bandpass_notch).
    // Throws std::runtime_error if no .edf files are found, a header fails to
    // parse (including a header-size self-check against the parsed field
    // widths), or a signal's digital_max == digital_min (division by zero).
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

   private:
    std::vector<nn::Tensor> windows_;
    std::vector<int> labels_;
    std::vector<WindowMetadata> metadata_;
};

} // namespace meeting01
