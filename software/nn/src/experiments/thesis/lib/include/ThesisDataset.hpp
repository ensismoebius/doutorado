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
    std::vector<SubjectFiles> subject_files;
    int n_subjects = 0;
    int n_stimuli = 0;
};

// Load all subjects under dataset.root, return raw samples.
// Modality controls which signals are loaded ("voice", "eeg", "fused").
auto load_dataset(const ThesisConfig::Dataset& dataset_cfg) -> ThesisDatasetView;

auto make_text_split(
    const std::vector<ThesisSample>& samples, const std::string& text_mode, uint32_t seed)
    -> TextSplit;

// Convert raw samples to speaker-keyed feature map (using pre-computed feature vectors).
// feature_dim must match the length of each feature vector.
auto build_speaker_map(const std::vector<ThesisSample>& samples,
    const std::vector<std::vector<double>>& feature_vectors) -> SpeakerFeatureMap;

} // namespace thesis
