/**
 * @file loadingData.cpp
 * @brief PyTorch-style loading/feeding demo for the 10.1117 EEG+Audio dataset.
 *
 * Pipeline implemented here:
 *   Subject directories (S01, S02, ...) -> Dataset -> DataLoader -> Batch -> Model
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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

using nn::dataLoaders::ARTIFACT_NAMES;
using nn::dataLoaders::EEG_CHANNELS_NAMES;
using nn::dataLoaders::ESTIMULUS_NAMES;
using nn::dataLoaders::MODALITY_NAMES;

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using std::cerr;
using std::cout;
using std::exception;
using std::size_t;
using std::string;

namespace
{

constexpr size_t EEG_FEATURES = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns();
constexpr size_t INPUT_FEATURES =
    EEG_FEATURES + nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();

auto resolveEegRowIndex(int eeg_index_label, size_t eeg_rows) -> size_t
{
    if (eeg_rows == 0)
    {
        throw std::runtime_error("Invalid EEG row count: 0");
    }

    // Protocol files are MATLAB-oriented and usually 1-based row indexing.
    if (eeg_index_label >= 1 && static_cast<size_t>(eeg_index_label) <= eeg_rows)
    {
        return static_cast<size_t>(eeg_index_label -
                                   1); // TODO - Check if this off-by-one adjustment is actually
                                       // needed for the provided dataset.
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
    if (eeg.rows() != ImaginedSpeechSchema_10_1117.eeg_channels ||
        eeg.cols() != ImaginedSpeechSchema_10_1117.eegSamplesPerChannel())
    {
        throw std::runtime_error("Unexpected EEG shape. Expected [6x4096].");
    }

    if (audio.rows() != ImaginedSpeechSchema_10_1117.audioSamples() || audio.cols() != 1)
    {
        throw std::runtime_error("Unexpected Audio shape. Expected [176400x1].");
    }

    nn::Tensor input(1, INPUT_FEATURES);

    size_t col = 0;
    for (size_t ch = 0; ch < ImaginedSpeechSchema_10_1117.eeg_channels; ++ch)
    {
        for (size_t s = 0; s < ImaginedSpeechSchema_10_1117.eegSamplesPerChannel(); ++s)
        {
            input.at(0, col++) = eeg.at(ch, s);
        }
    }

    for (size_t i = 0; i < ImaginedSpeechSchema_10_1117.audioSamples(); ++i)
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

auto makeSamplerOptions(const Config& config) -> DataLoader::DefaultSamplerOptions
{
    DataLoader::DefaultSamplerOptions options{};
    options.seed = config.seed;

    // Backward compatibility: when no explicit sampler is requested,
    // preserve legacy --shuffle/--no-shuffle behavior.
    if (config.sampler_type.empty())
    {
        options.type = config.shuffle ? DataLoader::DefaultSamplerType::Random
                                      : DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "sequential")
    {
        options.type = DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "random")
    {
        options.type = DataLoader::DefaultSamplerType::Random;
        return options;
    }

    if (config.sampler_type == "weighted")
    {
        options.type = DataLoader::DefaultSamplerType::WeightedRandom;
        options.weights = config.sampler_weights;
        options.weighted_num_samples = config.weighted_num_samples;
        return options;
    }

    if (config.sampler_type == "distributed")
    {
        options.type = DataLoader::DefaultSamplerType::Distributed;
        options.num_replicas = config.distributed_num_replicas;
        options.rank = config.distributed_rank;
        options.distributed_shuffle = config.distributed_shuffle;
        options.distributed_drop_last = config.distributed_drop_last;
        return options;
    }

    throw std::runtime_error("Unknown sampler type: " + config.sampler_type);
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

        audio_sessions_.resize(subjects_.size());
        eeg_sessions_.resize(subjects_.size());
    }

    [[nodiscard]] auto size() const -> size_t override
    {
        // Total number of synchronized (audio) samples across all subjects
        return prefix_audio_row_offsets_.empty() ? 0 : prefix_audio_row_offsets_.back();
    }

    [[nodiscard]] auto get_item(size_t idx) const -> Batch override
    {
        const auto t0 = std::chrono::steady_clock::now();
        if (idx >= size())
        {
            throw std::out_of_range("Dataset index out of range: " + std::to_string(idx));
        }

        // Find the subject that owns the global flattened index `idx`.
        // `upper_it` points to the first offset > idx, so the subject index
        // is one before that iterator.
        const auto upper_it = std::upper_bound( //
            prefix_audio_row_offsets_.begin(),  //
            prefix_audio_row_offsets_.end(),    //
            idx                                 //
        );

        const size_t subject_index = static_cast<size_t>( //
            std::distance(                                //
                prefix_audio_row_offsets_.begin(),        //
                upper_it) -
            1 //
        );

        const size_t audio_row = idx - prefix_audio_row_offsets_[subject_index];
        Batch out = loadSampleByLocalIndex(subject_index, audio_row);

        const auto t1 = std::chrono::steady_clock::now();
        ++perf_.get_item_calls;
        perf_.get_item_total_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return out;
    }

    [[nodiscard]] auto collate(const std::vector<std::size_t>& indices) const -> Batch override
    {
        const auto t0 = std::chrono::steady_clock::now();
        if (indices.empty())
        {
            return Dataset::collate(indices);
        }

        nn::Tensor inputs(indices.size(), INPUT_FEATURES);
        nn::Tensor targets(indices.size(), 5);

        struct RowRequest
        {
            size_t batch_row;
            size_t local_audio_row;
        };

        std::vector<std::vector<RowRequest>> grouped(subjects_.size());
        for (size_t row = 0; row < indices.size(); ++row)
        {
            const size_t idx = indices[row];
            if (idx >= size())
            {
                throw std::out_of_range("Dataset index out of range in collate: " +
                                        std::to_string(idx));
            }

            const auto upper_it = std::upper_bound(
                prefix_audio_row_offsets_.begin(), prefix_audio_row_offsets_.end(), idx);
            const size_t subject_index =
                static_cast<size_t>(std::distance(prefix_audio_row_offsets_.begin(), upper_it) - 1);
            grouped[subject_index].push_back(
                RowRequest{row, idx - prefix_audio_row_offsets_[subject_index]});
        }

        const auto t_reads_start = std::chrono::steady_clock::now();
        for (size_t subject_index = 0; subject_index < grouped.size(); ++subject_index)
        {
            if (grouped[subject_index].empty())
            {
                continue;
            }

            ensureSessions(subject_index);
            auto& requests = grouped[subject_index];
            std::sort(requests.begin(),
                      requests.end(),
                      [](const RowRequest& a, const RowRequest& b)
                      { return a.local_audio_row < b.local_audio_row; });

            size_t pos = 0;
            while (pos < requests.size())
            {
                size_t run_end = pos + 1;
                while (run_end < requests.size() && requests[run_end].local_audio_row ==
                                                        requests[run_end - 1].local_audio_row + 1)
                {
                    ++run_end;
                }

                const size_t run_start_row = requests[pos].local_audio_row;
                const size_t run_count = run_end - pos;
                const auto t_audio_start = std::chrono::steady_clock::now();
                const auto audio_rows_flat =
                    audio_sessions_.at(subject_index)->readRowsFlat(run_start_row, run_count);
                const auto t_audio_end = std::chrono::steady_clock::now();

                struct BatchTask
                {
                    size_t batch_row;
                    size_t audio_index;
                    int audio_stimulus;
                    int eeg_index_label;
                    size_t eeg_row;
                };

                std::vector<BatchTask> tasks;
                tasks.reserve(run_count);
                const SubjectFiles& subject = subjects_.at(subject_index);
                for (size_t k = 0; k < run_count; ++k)
                {
                    const RowRequest& req = requests[pos + k];
                    const int audio_stimulus = audio_rows_flat.stimuli[k];
                    const int eeg_index_label = audio_rows_flat.eegIndices[k];
                    const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
                    tasks.push_back(
                        BatchTask{req.batch_row, k, audio_stimulus, eeg_index_label, eeg_row});
                }

                std::vector<size_t> order(tasks.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(),
                          order.end(),
                          [&tasks](size_t a, size_t b)
                          { return tasks[a].eeg_row < tasks[b].eeg_row; });

                const auto t_eeg_start = std::chrono::steady_clock::now();
                size_t op = 0;
                while (op < order.size())
                {
                    size_t op_end = op + 1;
                    while (op_end < order.size() &&
                           tasks[order[op_end]].eeg_row == tasks[order[op_end - 1]].eeg_row + 1)
                    {
                        ++op_end;
                    }

                    const size_t eeg_run_start = tasks[order[op]].eeg_row;
                    const size_t eeg_run_count = op_end - op;
                    const auto eeg_rows_flat =
                        eeg_sessions_.at(subject_index)->readRowsFlat(eeg_run_start, eeg_run_count);

                    const auto t_pack_start = std::chrono::steady_clock::now();
                    nn::Tensor eegRow(1, EEG_FEATURES);
                    nn::Tensor audioRow(1, ImaginedSpeechSchema_10_1117.audioSamples());
                    float* eegDst = eegRow.mutable_data_ptr();
                    float* audioDst = audioRow.mutable_data_ptr();
                    const size_t audioCols = ImaginedSpeechSchema_10_1117.audioSamples();

                    for (size_t j = 0; j < eeg_run_count; ++j)
                    {
                        const BatchTask& task = tasks[order[op + j]];
                        const size_t audioOffset =
                            task.audio_index * ImaginedSpeechSchema_10_1117.audioSamples();
                        const size_t eegOffset = j * EEG_FEATURES;
                        const auto& eeg_labels = eeg_rows_flat.labels[j];

                        const int stimulus_label = eeg_labels[1];
                        if (task.audio_stimulus != stimulus_label)
                        {
                            throw std::runtime_error(
                                "Stimulus mismatch between audio and EEG in collate");
                        }

                        for (size_t c = 0; c < EEG_FEATURES; ++c)
                        {
                            eegDst[c] = eeg_rows_flat.signals[eegOffset + c];
                        }
                        inputs.setBlock(task.batch_row, 0, eegRow);

                        for (size_t c = 0; c < audioCols; ++c)
                        {
                            audioDst[c] = audio_rows_flat.samples[audioOffset + c];
                        }
                        inputs.setBlock(task.batch_row, EEG_FEATURES, audioRow);

                        targets.at(task.batch_row, 0) = static_cast<float>(subject.subject_id);
                        targets.at(task.batch_row, 1) = static_cast<float>(eeg_labels[0]);
                        targets.at(task.batch_row, 2) = static_cast<float>(eeg_labels[1]);
                        targets.at(task.batch_row, 3) = static_cast<float>(eeg_labels[2]);
                        targets.at(task.batch_row, 4) = static_cast<float>(task.eeg_index_label);
                    }
                    const auto t_pack_end = std::chrono::steady_clock::now();
                    perf_.collate_pack_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                                 t_pack_end - t_pack_start)
                                                 .count();

                    op = op_end;
                }
                const auto t_eeg_end = std::chrono::steady_clock::now();

                perf_.collate_audio_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                              t_audio_end - t_audio_start)
                                              .count();
                perf_.collate_eeg_us +=
                    std::chrono::duration_cast<std::chrono::microseconds>(t_eeg_end - t_eeg_start)
                        .count();

                pos = run_end;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();

        ++perf_.collate_calls;
        perf_.collate_total_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        perf_.collate_reads_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t_reads_start).count();

        if ((perf_.collate_calls % 5U) == 0U)
        {
            const double avg_collate_ms = static_cast<double>(perf_.collate_total_us) /
                                          static_cast<double>(perf_.collate_calls) / 1000.0;
            const double avg_reads_ms = static_cast<double>(perf_.collate_reads_us) /
                                        static_cast<double>(perf_.collate_calls) / 1000.0;
            const double avg_get_item_ms =
                perf_.get_item_calls == 0U ? 0.0
                                           : static_cast<double>(perf_.get_item_total_us) /
                                                 static_cast<double>(perf_.get_item_calls) / 1000.0;
            const double avg_audio_ms = static_cast<double>(perf_.collate_audio_us) /
                                        static_cast<double>(perf_.collate_calls) / 1000.0;
            const double avg_eeg_ms = static_cast<double>(perf_.collate_eeg_us) /
                                      static_cast<double>(perf_.collate_calls) / 1000.0;
            const double avg_pack_ms = static_cast<double>(perf_.collate_pack_us) /
                                       static_cast<double>(perf_.collate_calls) / 1000.0;

            std::cout << "[loader-perf] collate_calls=" << perf_.collate_calls
                      << " avg_collate_ms=" << avg_collate_ms
                      << " avg_read_loop_ms=" << avg_reads_ms << " avg_audio_ms=" << avg_audio_ms
                      << " avg_eeg_ms=" << avg_eeg_ms << " avg_pack_ms=" << avg_pack_ms
                      << " avg_get_item_ms=" << avg_get_item_ms << '\n';
        }

        return {.inputs = std::move(inputs), .targets = std::move(targets)};
    }

    [[nodiscard]] auto subjects() const -> const std::vector<SubjectFiles>&
    {
        return subjects_;
    }

   private:
    struct PerfStats
    {
        size_t collate_calls = 0;
        size_t get_item_calls = 0;
        long long collate_total_us = 0;
        long long collate_reads_us = 0;
        long long collate_audio_us = 0;
        long long collate_eeg_us = 0;
        long long collate_pack_us = 0;
        long long get_item_total_us = 0;
    };

    auto loadSampleByLocalIndex(size_t subject_index, size_t audio_row) const -> Batch
    {
        const SubjectFiles& subject = subjects_.at(subject_index);
        ensureSessions(subject_index);

        const auto [audio_tensor, audio_stimulus, eeg_index_label] =
            audio_sessions_.at(subject_index)->readRow(audio_row);

        const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
        const auto [    //
            eeg_tensor, //
            eeg_labels  //
        ] = eeg_sessions_.at(subject_index)->readRow(eeg_row);

        const int stimulus_label = eeg_labels[1];
        if (audio_stimulus != stimulus_label)
        {
            throw std::runtime_error("Stimulus mismatch between audio and EEG for subject " +
                                     subject.subject_name + " at audio row " +
                                     std::to_string(audio_row));
        }

        nn::Tensor input = makeInputTensor(eeg_tensor, audio_tensor);
        nn::Tensor target = makeTargetTensor(subject.subject_id, eeg_labels, eeg_index_label);
        return {.inputs = std::move(input), .targets = std::move(target)};
    }

    void ensureSessions(size_t subject_index) const
    {
        if (audio_sessions_.at(subject_index) && eeg_sessions_.at(subject_index))
        {
            return;
        }

        const SubjectFiles& subject = subjects_.at(subject_index);
        if (!audio_sessions_.at(subject_index))
        {
            audio_sessions_.at(subject_index) =
                std::make_unique<nn::dataLoaders::AudioMatSession>(subject.audio_mat_path);
        }
        if (!eeg_sessions_.at(subject_index))
        {
            eeg_sessions_.at(subject_index) =
                std::make_unique<nn::dataLoaders::EEGMatSession>(subject.eeg_mat_path);
        }
    }

    std::vector<SubjectFiles> subjects_;
    mutable PerfStats perf_{};
    mutable std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>> audio_sessions_;
    mutable std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>> eeg_sessions_;
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
            features.at(r, 1) =
                audio_abs_sum /
                static_cast<float>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples());
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
        .batch_size = 4,
        .max_batches = 10,
        .shuffle = true,
        .seed = 42U,
        .sampler_type = "",
        .sampler_weights = {},
        .weighted_num_samples = std::nullopt,
        .distributed_num_replicas = 1,
        .distributed_rank = 0,
        .distributed_shuffle = true,
        .distributed_drop_last = false,
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

        DataLoader loader(dataset, config.batch_size, makeSamplerOptions(config));

        DemoProbeModel model;

        cout << "Dataset root: " << config.dataset_root << '\n';
        cout << "Subjects discovered: " << dataset->subjects().size() << '\n';
        for (const auto& s : dataset->subjects())
        {
            cout << "  - " << s.subject_name << '\n';
        }
        cout << "Total synchronized samples: " << dataset->size() << "\n\n";

        size_t seen_batches = 0;
        auto it = loader.begin();
        auto end = loader.end();

        std::optional<std::future<Batch>> next_batch_future;
        if (it != end)
        {
            auto it_snapshot = it;
            next_batch_future =
                std::async(std::launch::async, [it_snapshot]() mutable { return *it_snapshot; });
        }

        while (it != end && seen_batches < config.max_batches)
        {
            if (!next_batch_future.has_value())
            {
                break;
            }

            Batch batch = next_batch_future->get();

            ++it;
            if (it != end && (seen_batches + 1) < config.max_batches)
            {
                auto it_snapshot = it;
                next_batch_future = std::async(std::launch::async,
                                               [it_snapshot]() mutable { return *it_snapshot; });
            }
            else
            {
                next_batch_future.reset();
            }

            ++seen_batches;
            nn::Tensor probe = model.forward(batch.inputs);

            printData(batch);
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
