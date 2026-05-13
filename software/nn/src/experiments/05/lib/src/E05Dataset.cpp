#include "E05Dataset.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"

namespace e05
{

namespace
{
// Stimulus → text phrase mapping for the 10.1117 dataset.
// Stimuli 1-5: vowels /a/ /e/ /i/ /o/ /u/
// Stimuli 6-10: directional words
static const std::vector<std::string> kStimulToPhrase = {
    "",         // index 0 unused
    "a", "e", "i", "o", "u",
    "arriba", "abajo", "izquierda", "derecha", "adelante"
};

std::string stimulus_to_phrase(int stim)
{
    if (stim >= 1 && stim < static_cast<int>(kStimulToPhrase.size()))
        return kStimulToPhrase[static_cast<size_t>(stim)];
    return "unknown";
}
} // namespace

auto load_dataset(const E05Config::Dataset& dataset_cfg) -> E05DatasetView
{
    auto subjects = discoverSubjects(dataset_cfg.root, "^S(\\d+)$");
    if (subjects.empty())
        throw std::runtime_error("E05Dataset: no subjects found in " + dataset_cfg.root);

    E05DatasetView view;
    view.subject_files = subjects;
    view.n_subjects = static_cast<int>(subjects.size());

    bool load_audio = (dataset_cfg.modality == "voice" || dataset_cfg.modality == "fused");
    bool load_eeg   = (dataset_cfg.modality == "eeg"   || dataset_cfg.modality == "fused");

    std::unordered_set<int> stimuli_seen;

    for (const auto& sf : subjects)
    {
        if (load_audio && !sf.audio_mat_path.empty())
        {
            nn::dataLoaders::AudioMatSession audio_session(sf.audio_mat_path, sf.subject_id);
            for (size_t row = 0; row < audio_session.rowCount(); ++row)
            {
                auto [audio_tensor, stimulus, eeg_index] = audio_session.readRow(row);
                E05Sample sample;
                sample.audio = std::move(audio_tensor);
                sample.stimulus = stimulus;
                sample.subject_id = sf.subject_id;
                sample.text_phrase = stimulus_to_phrase(stimulus);
                stimuli_seen.insert(stimulus);
                view.samples.push_back(std::move(sample));
            }
        }
        if (load_eeg && !sf.eeg_mat_path.empty())
        {
            nn::dataLoaders::EEGMatSession eeg_session(sf.eeg_mat_path, sf.subject_id);
            for (size_t row = 0; row < eeg_session.rowCount(); ++row)
            {
                auto [eeg_tensor, labels] = eeg_session.readRow(row);
                E05Sample sample;
                sample.eeg = std::move(eeg_tensor);
                sample.stimulus = labels[1]; // stimulus is index 1
                sample.subject_id = sf.subject_id;
                sample.text_phrase = stimulus_to_phrase(labels[1]);
                stimuli_seen.insert(labels[1]);
                view.samples.push_back(std::move(sample));
            }
        }
    }

    // For fused: second pass loads EEG by matching eeg_index from audio rows.
    // (If fused and both passes ran, EEG was loaded from eeg-only pass above.
    //  Proper fused loading requires correlation via eeg_index; simplified here
    //  by loading audio and eeg independently and pairing by stimulus order.)

    // Apply max_samples limit if set (for debug runs).
    if (dataset_cfg.max_samples > 0 &&
        static_cast<int>(view.samples.size()) > dataset_cfg.max_samples)
    {
        view.samples.resize(static_cast<size_t>(dataset_cfg.max_samples));
    }

    view.n_stimuli = static_cast<int>(stimuli_seen.size());
    return view;
}

auto make_text_split(const std::vector<E05Sample>& samples,
    const std::string& text_mode,
    uint32_t seed) -> TextSplit
{
    if (text_mode == "dependent")
    {
        // All phrases available at both train and test; split by utterance index.
        std::vector<size_t> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
        size_t split_point = indices.size() * 8 / 10; // 80/20
        TextSplit ts;
        ts.train_indices = {indices.begin(), indices.begin() + static_cast<ptrdiff_t>(split_point)};
        ts.test_indices  = {indices.begin() + static_cast<ptrdiff_t>(split_point), indices.end()};
        return ts;
    }

    if (text_mode == "independent")
    {
        // Collect unique phrases, split phrases into train-set and test-set halves.
        std::vector<std::string> phrases;
        for (const auto& s : samples)
        {
            if (std::find(phrases.begin(), phrases.end(), s.text_phrase) == phrases.end())
                phrases.push_back(s.text_phrase);
        }
        std::mt19937 rng(seed);
        std::shuffle(phrases.begin(), phrases.end(), rng);
        size_t phrase_split = phrases.size() / 2;
        std::unordered_set<std::string> train_phrases(phrases.begin(),
            phrases.begin() + static_cast<ptrdiff_t>(phrase_split));

        TextSplit ts;
        for (size_t i = 0; i < samples.size(); ++i)
        {
            if (train_phrases.count(samples[i].text_phrase))
                ts.train_indices.push_back(i);
            else
                ts.test_indices.push_back(i);
        }
        return ts;
    }

    throw std::invalid_argument("E05Dataset: unknown text_mode " + text_mode);
}

auto build_speaker_map(const std::vector<E05Sample>& samples,
    const std::vector<std::vector<double>>& feature_vectors) -> SpeakerFeatureMap
{
    if (samples.size() != feature_vectors.size())
        throw std::invalid_argument("E05Dataset: samples/feature_vectors size mismatch");

    SpeakerFeatureMap map;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        std::string key = "subject_" + std::to_string(samples[i].subject_id);
        map[key].push_back(feature_vectors[i]);
    }
    return map;
}

} // namespace e05
