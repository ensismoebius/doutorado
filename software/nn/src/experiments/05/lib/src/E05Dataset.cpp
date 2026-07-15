#include "E05Dataset.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"
#include "progress/ProgressManager.hpp"

namespace e05
{

namespace
{
// Stimulus → text phrase mapping for the 10.1117 dataset.
// Stimuli 1-5: vowels /a/ /e/ /i/ /o/ /u/
// Stimuli 6-10: directional words
static const std::vector<std::string> kStimulToPhrase = {"", // index 0 unused
    "a",
    "e",
    "i",
    "o",
    "u",
    "arriba",
    "abajo",
    "izquierda",
    "derecha",
    "adelante"};

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

    std::unordered_set<int> stimuli_seen;
    int subjects_with_both = 0;

    const auto n_subjects = static_cast<float>(subjects.size());
    uint32_t load_bar =
        nn::progress::ProgressManager::instance().create_bar("Loading dataset", n_subjects);
    nn::progress::ProgressManager::instance().set_description(
        load_bar, "Pairing audio+EEG trials per subject");

    int subjects_processed = 0;

    // Always load paired audio+EEG so every modality run operates on the
    // same set of trials. The modality field controls feature extraction, not
    // which samples are included. Samples where either audio or EEG data is
    // absent are dropped to guarantee comparability across modality runs.
    for (const auto& sf : subjects)
    {
        if (sf.audio_path.empty() || sf.eeg_path.empty())
            continue; // skip subjects missing either modality

        nn::dataLoaders::AudioSession audio_session(sf.audio_path, sf.subject_id);
        nn::dataLoaders::EEGSession eeg_session(sf.eeg_path, sf.subject_id);

        const size_t n_audio = audio_session.rowCount();
        const size_t n_eeg = eeg_session.rowCount();
        if (n_audio == 0 || n_eeg == 0) continue;

        int paired = 0;
        for (size_t row = 0; row < n_audio; ++row)
        {
            auto [audio_tensor, stimulus, eeg_index] = audio_session.readRow(row);

            // eeg_index is 1-based in the dataset; convert to 0-based.
            const size_t eeg_row = (eeg_index > 0) ? static_cast<size_t>(eeg_index - 1)
                                                   : row; // fallback: same-row pairing

            if (eeg_row >= n_eeg) continue; // eeg_index out of range — drop trial

            auto [eeg_tensor, eeg_labels] = eeg_session.readRow(eeg_row);

            E05Sample sample;
            sample.audio = std::move(audio_tensor);
            sample.eeg = std::move(eeg_tensor);
            sample.stimulus = stimulus;
            sample.subject_id = sf.subject_id;
            sample.text_phrase = stimulus_to_phrase(stimulus);
            stimuli_seen.insert(stimulus);
            view.samples.push_back(std::move(sample));
            ++paired;
        }

        if (paired > 0) ++subjects_with_both;

        nn::progress::ProgressManager::instance().update_bar(
            load_bar, static_cast<float>(++subjects_processed));
    }

    nn::progress::ProgressManager::instance().complete_bar(load_bar);

    if (view.samples.empty())
        throw std::runtime_error(
            "E05Dataset: no paired audio+EEG samples found. "
            "Check that each subject has both audio and EEG .mat files.");

    view.n_subjects = subjects_with_both;
    view.n_stimuli = static_cast<int>(stimuli_seen.size());

    // Apply max_samples limit (debug/smoke runs). Samples are stored
    // subject-contiguous, so a plain resize() to the first N would keep only the
    // first 2-3 speakers — which breaks speaker-disjoint (GroupKFold) folds,
    // especially nested CV. Instead round-robin across subjects so the truncated
    // set spans every speaker.
    if (dataset_cfg.max_samples > 0 &&
        static_cast<int>(view.samples.size()) > dataset_cfg.max_samples)
    {
        std::map<int, std::vector<size_t>> by_subject;
        for (size_t i = 0; i < view.samples.size(); ++i)
            by_subject[view.samples[i].subject_id].push_back(i);

        std::vector<size_t> keep;
        keep.reserve(static_cast<size_t>(dataset_cfg.max_samples));
        size_t round = 0;
        bool added = true;
        while (static_cast<int>(keep.size()) < dataset_cfg.max_samples && added)
        {
            added = false;
            for (auto& [sid, idxs] : by_subject)
            {
                if (round < idxs.size())
                {
                    keep.push_back(idxs[round]);
                    added = true;
                    if (static_cast<int>(keep.size()) >= dataset_cfg.max_samples) break;
                }
            }
            ++round;
        }
        std::sort(keep.begin(), keep.end());

        std::vector<E05Sample> trimmed;
        trimmed.reserve(keep.size());
        for (size_t i : keep) trimmed.push_back(std::move(view.samples[i]));
        view.samples = std::move(trimmed);
    }

    return view;
}

auto make_text_split(
    const std::vector<E05Sample>& samples, const std::string& text_mode, uint32_t seed) -> TextSplit
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
        ts.test_indices = {indices.begin() + static_cast<ptrdiff_t>(split_point), indices.end()};
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
        std::unordered_set<std::string> train_phrases(
            phrases.begin(), phrases.begin() + static_cast<ptrdiff_t>(phrase_split));

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
