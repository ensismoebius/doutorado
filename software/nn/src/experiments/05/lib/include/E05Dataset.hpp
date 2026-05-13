#pragma once

#include <map>
#include <string>
#include <vector>

#include "E05Config.hpp"
#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"
#include "tensor/Tensor.hpp"

namespace e05
{

// One raw sample from a single subject: audio + EEG tensors plus stimulus label.
struct E05Sample
{
    nn::Tensor audio;       // shape (N_audio_samples, 1)
    nn::Tensor eeg;         // shape (N_eeg_channels, N_eeg_samples)
    int stimulus = 0;       // word/vowel index
    int subject_id = 0;
    std::string text_phrase; // e.g. "arriba", "a", etc.
};

// Per-subject flattened feature vectors, keyed by speaker label string.
using SpeakerFeatureMap = std::map<std::string, std::vector<std::vector<double>>>;

// Result of loading the full dataset.
struct E05DatasetView
{
    std::vector<E05Sample> samples;
    std::vector<SubjectFiles> subject_files;
    int n_subjects = 0;
    int n_stimuli = 0;
};

// Load all subjects under dataset.root, return raw samples.
// Modality controls which signals are loaded ("voice", "eeg", "fused").
auto load_dataset(const E05Config::Dataset& dataset_cfg) -> E05DatasetView;

// Split samples into text-dependent or text-independent train/test indices.
// text-dependent:   train and test use the same phrase set (split by utterance).
// text-independent: train and test use disjoint phrase sets.
struct TextSplit
{
    std::vector<size_t> train_indices;
    std::vector<size_t> test_indices;
};

auto make_text_split(const std::vector<E05Sample>& samples,
    const std::string& text_mode,
    uint32_t seed) -> TextSplit;

// Convert raw samples to speaker-keyed feature map (using pre-computed feature vectors).
// feature_dim must match the length of each feature vector.
auto build_speaker_map(const std::vector<E05Sample>& samples,
    const std::vector<std::vector<double>>& feature_vectors) -> SpeakerFeatureMap;

} // namespace e05
