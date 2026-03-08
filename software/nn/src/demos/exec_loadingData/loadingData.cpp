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
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lib/include/batch_util.hpp"
#include "lib/include/cli.hpp"
#include "lib/include/subject_discovery.hpp"
#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/Dataset.hpp"

using nn::dataLoaders::loadAudioFromMat;
using nn::dataLoaders::loadEEGFromMat;

using nn::dataLoaders::ARTIFACT_NAMES;
using nn::dataLoaders::AUDIO_SAMPLES_COUNT;
using nn::dataLoaders::EEG_CHANNELS;
using nn::dataLoaders::EEG_CHANNELS_NAMES;
using nn::dataLoaders::EEG_SAMPLE_COUNT;
using nn::dataLoaders::ESTIMULUS_NAMES;
using nn::dataLoaders::MODALITY_NAMES;

using std::cerr;
using std::cout;
using std::exception;
using std::size_t;
using std::string;

namespace
{

constexpr size_t EEG_FEATURES = EEG_CHANNELS * EEG_SAMPLE_COUNT;
constexpr size_t INPUT_FEATURES = EEG_FEATURES + AUDIO_SAMPLES_COUNT;

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
    if (eeg.rows() != EEG_CHANNELS || eeg.cols() != EEG_SAMPLE_COUNT)
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
        for (size_t s = 0; s < EEG_SAMPLE_COUNT; ++s)
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
        // Build prefix-sum offsets for per-subject audio rows.
        // After this loop `prefix_audio_row_offsets_` contains cumulative
        // counts such that prefix_audio_row_offsets_[i] is the start index
        // (in the flattened dataset space) of subject i's audio rows.
        // Example: subjects audio_rows = {A, B, C} -> offsets = {0, A, A+B, A+B+C}
        prefix_audio_row_offsets_.reserve(subjects_.size() + 1);
        prefix_audio_row_offsets_.emplace_back(0);

        for (const auto& subject : subjects_)
        {
            const size_t next = prefix_audio_row_offsets_.back() + subject.audio_rows;
            prefix_audio_row_offsets_.emplace_back(next);
        }
    }

    [[nodiscard]] auto size() const -> size_t override
    {
        // Total number of synchronized (audio) samples across all subjects
        return prefix_audio_row_offsets_.empty() ? 0 : prefix_audio_row_offsets_.back();
    }

    [[nodiscard]] auto get_item(size_t idx) const -> Batch override
    {
        if (idx >= size())
        {
            throw std::out_of_range("Dataset index out of range: " + std::to_string(idx));
        }

        // Find the subject that owns the global flattened index `idx`.
        // `upper_it` points to the first offset > idx, so the subject index
        // is one before that iterator.
        const auto upper_it = std::upper_bound(
            prefix_audio_row_offsets_.begin(), prefix_audio_row_offsets_.end(), idx);
        const size_t subject_index =
            static_cast<size_t>(std::distance(prefix_audio_row_offsets_.begin(), upper_it) - 1);

        // Convert global index to per-subject (local) audio row index.
        const size_t local_audio_row_index = idx - prefix_audio_row_offsets_[subject_index];
        const SubjectFiles& subject = subjects_.at(subject_index);

        const auto [audio_tensor, audio_stimulus, eeg_index_label] =
            loadAudioFromMat(subject.audio_mat_path, local_audio_row_index);

        const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
        const auto [eeg_tensor, eeg_labels] = loadEEGFromMat(subject.eeg_mat_path, eeg_row);

        if (audio_stimulus != eeg_labels[1])
        {
            throw std::runtime_error("Stimulus mismatch between audio and EEG for subject " +
                                     subject.subject_name + " at audio row " +
                                     std::to_string(local_audio_row_index));
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
    // Cumulative start offsets (prefix sums) of audio rows for each subject.
    // Use this to map a flattened dataset index -> (subject index, local row).
    std::vector<size_t> prefix_audio_row_offsets_;
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

} // namespace

auto main(int argc, char* argv[]) -> int
{
    const Config default_config{
        .subject_regex_pattern = "^S(\\d+)$",
        .dataset_root =
            "/home/ensismoebius/Documentos"
            "/UNESP/doutorado/databases/"
            "BaseDeDatosHablaImaginada",
        .batch_size = 20,
        .max_batches = 20,
        .shuffle = true,
        .seed = 42U,
    };

    Config config{};
    parseCliParams(argc, argv, config, default_config);

    try
    {
        auto discovered = discoverSubjects( //
            config.dataset_root,            //
            config.subject_regex_pattern    //
        );
        auto dataset = std::make_shared<Protocol101117Dataset>(discovered);

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
