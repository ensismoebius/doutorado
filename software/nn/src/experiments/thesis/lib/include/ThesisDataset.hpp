#pragma once

#include <map>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"
#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"
#include "tensor/Tensor.hpp"

namespace thesis
{

// One raw sample from a single subject: audio + EEG tensors plus stimulus label.
struct ThesisSample
{
    nn::Tensor audio; // shape (N_audio_samples, 1)
    nn::Tensor eeg;   // shape (N_eeg_channels, N_eeg_samples)
    int stimulus = 0; // word/vowel index
    int subject_id = 0;
    std::string text_phrase; // e.g. "arriba", "a", etc.
};

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

// Split samples into text-dependent or text-independent train/test indices.
// text-dependent:   train and test use the same phrase set (split by utterance).
// text-independent: train and test use disjoint phrase sets.
struct TextSplit
{
    std::vector<size_t> train_indices;
    std::vector<size_t> test_indices;
};

auto make_text_split(
    const std::vector<ThesisSample>& samples, const std::string& text_mode, uint32_t seed)
    -> TextSplit;

// Convert raw samples to speaker-keyed feature map (using pre-computed feature vectors).
// feature_dim must match the length of each feature vector.
auto build_speaker_map(const std::vector<ThesisSample>& samples,
    const std::vector<std::vector<double>>& feature_vectors) -> SpeakerFeatureMap;

} // namespace thesis
