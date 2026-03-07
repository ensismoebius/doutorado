/**
 * @file loadingData.cpp
 * @brief PyTorch-style loading/feeding demo for the 10.1117 EEG+Audio dataset.
 *
 * Pipeline implemented here:
 *   Subject directories (S01, S02, ...) -> Dataset -> DataLoader -> Batch -> Model
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lib/include/cli.hpp"
#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/Dataset.hpp"

using nn::dataLoaders::loadAudioFromMat;
using nn::dataLoaders::loadEEGFromMat;

using nn::dataLoaders::ARTIFACT_NAMES;
using nn::dataLoaders::AUDIO_SAMPLES_COUNT;
using nn::dataLoaders::EEG_CHANNELS_NAMES;
using nn::dataLoaders::ESTIMULUS_NAMES;
using nn::dataLoaders::MODALITY_NAMES;

using std::cerr;
using std::cout;
using std::exception;
using std::size_t;
using std::string;

namespace
{

constexpr size_t EEG_CHANNELS = 6;
constexpr size_t EEG_SAMPLES_PER_CHANNEL = 4096;
constexpr size_t EEG_FEATURES = EEG_CHANNELS * EEG_SAMPLES_PER_CHANNEL;
constexpr size_t INPUT_FEATURES = EEG_FEATURES + AUDIO_SAMPLES_COUNT;

struct SubjectFiles
{
    int subject_id = 0;
    string subject_name;
    string eeg_mat_path;
    string audio_mat_path;
    size_t eeg_rows = 0;
    size_t audio_rows = 0;
};

auto discoverSubjects(const string& root_dir, string& subject_regex_pattern)
    -> std::vector<SubjectFiles>
{
    namespace fs = std::filesystem;

    // Checks if root path exists and is a directory
    fs::path root_path(root_dir);
    if (!fs::exists(root_path) || !fs::is_directory(root_path))
    {
        throw std::runtime_error("Dataset root does not exist or is not a directory: " + root_dir);
    }

    // List subject directories matching the regex
    // pattern and check for required MAT files
    std::vector<SubjectFiles> subjects;

    // Create regex for subject selection.
    std::regex selection_pattern(subject_regex_pattern);

    // Iterate over entries in the root directory
    for (const auto& entry : fs::directory_iterator(root_path))
    {
        if (!entry.is_directory())
        {
            continue;
        }

        // Extract the directory name
        const string dir_name = entry.path().filename().string();

        // Check if the directory name matches the subject selection pattern
        std::smatch regex_groups_matches;

        if (!std::regex_match(dir_name, regex_groups_matches, selection_pattern))
        {
            continue;
        }

        // Extract subject ID from the regex match
        // (assuming it's in the first capture group)
        const int subject_id = std::stoi(regex_groups_matches[1].str());

        const fs::path eeg_path = entry.path() / (dir_name + "_EEG.mat");
        const fs::path audio_path = entry.path() / (dir_name + "_Audio.mat");

        if (!fs::exists(eeg_path) || !fs::exists(audio_path))
        {
            continue;
        }

        SubjectFiles info{};
        info.subject_id = subject_id;
        info.subject_name = dir_name;
        info.eeg_mat_path = eeg_path.string();
        info.audio_mat_path = audio_path.string();

        subjects.push_back(std::move(info));
    }

    if (subjects.empty())
    {
        throw std::runtime_error(
            "No valid subject directories found. "
            "Expected S01/S01_EEG.mat and S01/S01_Audio.mat" //
        );
    }

    std::sort(            //
        subjects.begin(), //
        subjects.end(),   //
        [](const SubjectFiles& a, const SubjectFiles& b)
        {
            return a.subject_id < b.subject_id; // sort by subject ID ascending
        } //
    );

    return subjects;
}

auto resolveEegRowIndex(int eeg_index_label, size_t eeg_rows) -> size_t
{
    if (eeg_rows == 0)
    {
        throw std::runtime_error("Invalid EEG row count: 0");
    }

    // Protocol files are MATLAB-oriented and usually 1-based row indexing.
    if (eeg_index_label >= 1 && static_cast<size_t>(eeg_index_label) <= eeg_rows)
    {
        return static_cast<size_t>(eeg_index_label - 1);
    }

    if (eeg_index_label >= 0 && static_cast<size_t>(eeg_index_label) < eeg_rows)
    {
        return static_cast<size_t>(eeg_index_label);
    }

    throw std::runtime_error("Audio->EEG index label is out of range: " +
                             std::to_string(eeg_index_label));
}

auto makeInputTensor(const nn::Tensor& eeg, const nn::Tensor& audio) -> nn::Tensor
{
    if (eeg.rows() != EEG_CHANNELS || eeg.cols() != EEG_SAMPLES_PER_CHANNEL)
    {
        throw std::runtime_error("Unexpected EEG shape. Expected [6x4096].");
    }

    if (audio.rows() != AUDIO_SAMPLES_COUNT || audio.cols() != 1)
    {
        throw std::runtime_error("Unexpected Audio shape. Expected [176400x1].");
    }

    nn::Tensor input(1, INPUT_FEATURES);

    size_t col = 0;
    for (size_t ch = 0; ch < EEG_CHANNELS; ++ch)
    {
        for (size_t s = 0; s < EEG_SAMPLES_PER_CHANNEL; ++s)
        {
            input.at(0, col++) = eeg.at(ch, s);
        }
    }

    for (size_t i = 0; i < AUDIO_SAMPLES_COUNT; ++i)
    {
        input.at(0, col++) = audio.at(i, 0);
    }

    return input;
}

auto makeTargetTensor(int subject_id, const std::array<int, 3>& eeg_labels, int eeg_index_label)
    -> nn::Tensor
{
    // [subject_id, modality, stimulus, artifact, eeg_index_label]
    nn::Tensor target(1, 5);
    target.at(0, 0) = static_cast<float>(subject_id);
    target.at(0, 1) = static_cast<float>(eeg_labels[0]);
    target.at(0, 2) = static_cast<float>(eeg_labels[1]);
    target.at(0, 3) = static_cast<float>(eeg_labels[2]);
    target.at(0, 4) = static_cast<float>(eeg_index_label);
    return target;
}

class Protocol101117Dataset : public Dataset
{
   public:
    explicit Protocol101117Dataset(std::vector<SubjectFiles> subjects)
        : subjects_(std::move(subjects))
    {
        prefix_offsets_.reserve(subjects_.size() + 1);
        prefix_offsets_.push_back(0);

        for (const auto& subject : subjects_)
        {
            const size_t next = prefix_offsets_.back() + subject.audio_rows;
            prefix_offsets_.push_back(next);
        }
    }

    [[nodiscard]] auto size() const -> size_t override
    {
        return prefix_offsets_.empty() ? 0 : prefix_offsets_.back();
    }

    [[nodiscard]] auto get_item(size_t idx) const -> Batch override
    {
        if (idx >= size())
        {
            throw std::out_of_range("Dataset index out of range: " + std::to_string(idx));
        }

        const auto upper = std::upper_bound(prefix_offsets_.begin(), prefix_offsets_.end(), idx);
        const size_t subject_pos =
            static_cast<size_t>(std::distance(prefix_offsets_.begin(), upper) - 1);
        const size_t local_audio_row = idx - prefix_offsets_[subject_pos];
        const SubjectFiles& subject = subjects_.at(subject_pos);

        const auto [audio_tensor, audio_stimulus, eeg_index_label] =
            loadAudioFromMat(subject.audio_mat_path, local_audio_row);

        const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
        const auto [eeg_tensor, eeg_labels] = loadEEGFromMat(subject.eeg_mat_path, eeg_row);

        if (audio_stimulus != eeg_labels[1])
        {
            throw std::runtime_error("Stimulus mismatch between audio and EEG for subject " +
                                     subject.subject_name + " at audio row " +
                                     std::to_string(local_audio_row));
        }

        nn::Tensor input = makeInputTensor(eeg_tensor, audio_tensor);
        nn::Tensor target = makeTargetTensor(subject.subject_id, eeg_labels, eeg_index_label);

        return {.inputs = std::move(input), .targets = std::move(target)};
    }

    [[nodiscard]] auto subjects() const -> const std::vector<SubjectFiles>&
    {
        return subjects_;
    }

   private:
    std::vector<SubjectFiles> subjects_;
    std::vector<size_t> prefix_offsets_;
};

class DemoProbeModel
{
   public:
    [[nodiscard]] auto forward(const nn::Tensor& batch_inputs) const -> nn::Tensor
    {
        // Forward pass emits two simple modality-agnostic features per sample:
        // mean(|EEG|) and mean(|Audio|). This keeps focus on loader/feeder flow.
        nn::Tensor features(batch_inputs.rows(), 2);

        for (size_t r = 0; r < batch_inputs.rows(); ++r)
        {
            float eeg_abs_sum = 0.0f;
            for (size_t c = 0; c < EEG_FEATURES; ++c)
            {
                eeg_abs_sum += std::abs(batch_inputs.at(r, c));
            }

            float audio_abs_sum = 0.0f;
            for (size_t c = EEG_FEATURES; c < INPUT_FEATURES; ++c)
            {
                audio_abs_sum += std::abs(batch_inputs.at(r, c));
            }

            features.at(r, 0) = eeg_abs_sum / static_cast<float>(EEG_FEATURES);
            features.at(r, 1) = audio_abs_sum / static_cast<float>(AUDIO_SAMPLES_COUNT);
        }

        return features;
    }
};

auto printData(const Batch& batch) -> void
{
    for (size_t i = 0; i < batch.inputs.rows(); ++i)
    {
        cout << "Sample " << i << ":\n";
        cout << "  - Target subject ID: " << batch.targets.at(i, 0) << '\n';
        cout << "  - Target modality: "
             << MODALITY_NAMES.at(static_cast<int>(batch.targets.at(i, 1))) << '\n';
        cout << "  - Target stimulus: "
             << ESTIMULUS_NAMES.at(static_cast<int>(batch.targets.at(i, 2))) << '\n';
        cout << "  - Target artifact: "
             << ARTIFACT_NAMES.at(static_cast<int>(batch.targets.at(i, 3))) << '\n';
        cout << "  - Target EEG index label: " << batch.targets.at(i, 4) << "\n\n";
    }
}

} // namespace

auto main(int argc, char* argv[]) -> int
{
    const Config default_config{
        .subject_regex_pattern = "^S(\\d+)$",
        .dataset_root =
            "/home/ensismoebius/Documentos"
            "/UNESP/doutorado/databases/"
            "BaseDeDatosHablaImaginada",
        .batch_size = 4,
        .max_batches = 2,
        .shuffle = true,
        .seed = 42U,
    };

    Config config{};
    parseCliParams(argc, argv, config, default_config);

    try
    {
        auto discovered = discoverSubjects(config.dataset_root, config.subject_regex_pattern);
        auto dataset = std::make_shared<Protocol101117Dataset>(std::move(discovered));

        DataLoader loader(dataset, config.batch_size, config.shuffle, config.seed);

        DemoProbeModel model;

        cout << "Dataset root: " << config.dataset_root << '\n';
        cout << "Subjects discovered: " << dataset->subjects().size() << '\n';
        for (const auto& s : dataset->subjects())
        {
            cout << "  - " << s.subject_name << '\n';
        }
        cout << "Total synchronized samples: " << dataset->size() << "\n\n";

        size_t seen_batches = 0;
        for (const auto& batch : loader)
        {
            ++seen_batches;
            nn::Tensor probe = model.forward(batch.inputs);

            printData(batch);

            if (seen_batches >= config.max_batches)
            {
                break;
            }
        }

        if (seen_batches == 0)
        {
            cout << "No batches produced. Check dataset files and row counts.\n";
        }

        cout << "Pipeline completed: Dataset -> DataLoader -> Batch -> Model\n";
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
