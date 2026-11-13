#pragma once

#include <map>
#include <string>
#include <vector>

#include "TextSplit.hpp"
#include "ThesisConfig.hpp"
#include "ThesisSample.hpp"
#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"
#include "tensor/Tensor.hpp"

namespace thesis
{

// Per-subject flattened feature vectors, keyed by speaker label string.
using SpeakerFeatureMap = std::map<std::string, std::vector<std::vector<double>>>;

// Result of loading the full dataset.
struct ThesisDatasetView
{
    std::vector<ThesisSample> samples;
    std::vector<SubjectFiles> subject_files; ///< coretto2017 only (one entry per speaker)
    int n_subjects = 0;
    int n_stimuli = 0;
    /// The rates the signals were recorded at, set by the loader from the base (Hz). The
    /// extractors read them (frequency scales, jitter, the simulated whisper); 0 means "not
    /// loaded" and every reader refuses it. Until 2026-11-13 they were constants
    /// (44100 / 1024, the Coretto base), wrong for any other base without an error.
    double voice_sample_rate = 0.0;
    double eeg_sample_rate = 0.0;
};

/// The rate a reader of `view` needs, refused when the loader never set it.
auto require_rate(double rate_hz, const char* which) -> double;

// Load the base dataset.source names from dataset.root, return raw samples.
// coretto2017: voice + EEG of the condition asked (pronounced: same trial; imagined: drawn);
// biomex2021: the pairs written by scripts/data/biomex_to_sqlite.py (pronounced only).
auto load_dataset(const ThesisConfig::Dataset& dataset_cfg) -> ThesisDatasetView;

auto make_text_split(
    const std::vector<ThesisSample>& samples, const std::string& text_mode, uint32_t seed)
    -> TextSplit;

// Convert raw samples to speaker-keyed feature map (using pre-computed feature vectors).
// feature_dim must match the length of each feature vector.
auto build_speaker_map(const std::vector<ThesisSample>& samples,
    const std::vector<std::vector<double>>& feature_vectors) -> SpeakerFeatureMap;

} // namespace thesis
