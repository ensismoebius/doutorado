#include "../include/GuayaquilDataset.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

#include "GuayaquilMitBih.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "utility/SignalPreprocessing.hpp"

namespace guayaquil
{

auto to_window_tensor(const nn::Tensor& signal, int window_size) -> std::vector<nn::Tensor>
{
    std::vector<nn::Tensor> windows;
    if (window_size <= 0 || signal.size() == 0) return windows;

    std::size_t offset = 0;
    const std::size_t signal_len = static_cast<std::size_t>(signal.rows());
    while (offset < signal_len)
    {
        const std::size_t remaining = signal_len - offset;
        const std::size_t take =
            std::min<std::size_t>(remaining, static_cast<std::size_t>(window_size));

        nn::Tensor sample(static_cast<nn::Index>(window_size), 1);
        for (int t = 0; t < window_size; ++t)
        {
            if (static_cast<std::size_t>(t) < take)
            {
                sample.at(static_cast<nn::Index>(t), 0) =
                    signal.at(static_cast<nn::Index>(offset + static_cast<std::size_t>(t)), 0);
            }
        }

        nn::utility::zscore_inplace(sample);
        windows.push_back(std::move(sample));
        offset += static_cast<std::size_t>(window_size);
    }

    return windows;
}

auto collect_signal_files(const GuayaquilConfig& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(cfg.dataset.dataset_root);
    if (!fs::exists(root))
        throw std::runtime_error("Dataset root does not exist: " + root.string());

    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file()) continue;
        const std::string path_str = entry.path().string();
        const std::string ext = entry.path().extension().string();

        if ((ext == ".csv" || ext == ".txt") &&
            (path_str.find("physionet") != std::string::npos ||
                path_str.find("PhysioNet") != std::string::npos))
        {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

auto assign_speaker_fold(std::span<const WindowMetadata> meta, int cv_fold, int num_folds)
    -> SpeakerFoldAssignment
{
    if (num_folds < 2)
        throw std::runtime_error(
            "assign_speaker_fold: num_folds must be >= 2 (got " + std::to_string(num_folds) + ")");
    if (cv_fold < 0 || cv_fold >= num_folds)
        throw std::runtime_error("assign_speaker_fold: cv_fold " + std::to_string(cv_fold) +
                                 " out of range [0, " + std::to_string(num_folds) + ")");

    std::set<int> distinct;
    for (const auto& m : meta) distinct.insert(m.speaker_id);
    if (static_cast<int>(distinct.size()) < num_folds)
        throw std::runtime_error("assign_speaker_fold: metadata has " +
                                 std::to_string(distinct.size()) +
                                 " distinct speakers but num_folds=" + std::to_string(num_folds) +
                                 " — cv_num_folds must not exceed the speaker count.");

    // Grouped leave-one-group-out: partition the sorted speaker ids into
    // `num_folds` contiguous groups. Fold f's test group is group f, its
    // validation group is group (f+1) mod num_folds, and everything else trains.
    // When distinct == num_folds each group holds exactly one speaker and this
    // reduces to the original leave-one-speaker-out behaviour (FSDD).
    const std::vector<int> sorted_ids(distinct.begin(), distinct.end()); // std::set is ordered
    const int n = static_cast<int>(sorted_ids.size());
    auto group_of = [&](int rank) { return (rank * num_folds) / n; };

    std::map<int, int> group_by_sid;
    for (int rank = 0; rank < n; ++rank)
        group_by_sid.emplace(sorted_ids[static_cast<std::size_t>(rank)], group_of(rank));

    const int test_group = cv_fold;
    const int val_group = (cv_fold + 1) % num_folds;

    SpeakerFoldAssignment out;
    std::set<std::string> train_spk;
    std::set<std::string> val_spk;
    std::set<std::string> test_spk;
    for (std::size_t i = 0; i < meta.size(); ++i)
    {
        const WindowMetadata& m = meta[i];
        const int g = group_by_sid.at(m.speaker_id);
        if (g == test_group)
        {
            out.test_idx.push_back(i);
            test_spk.insert(m.speaker);
        }
        else if (g == val_group)
        {
            out.val_idx.push_back(i);
            val_spk.insert(m.speaker);
        }
        else
        {
            out.train_idx.push_back(i);
            train_spk.insert(m.speaker);
        }
    }
    auto join = [](const std::set<std::string>& s)
    {
        std::string r;
        for (const auto& x : s) r += (r.empty() ? "" : ",") + x;
        return r;
    };
    out.train_speakers.assign(train_spk.begin(), train_spk.end());
    out.val_speaker = join(val_spk);
    out.test_speaker = join(test_spk);

    if (out.train_idx.empty() || out.val_idx.empty() || out.test_idx.empty())
        throw std::runtime_error(
            "assign_speaker_fold: fold " + std::to_string(cv_fold) +
            " produced an empty train/val/test partition (num_folds too small?).");

    return out;
}

namespace
{

// Legacy pooled split: shuffle every window and slice off the validation tail.
// This CANNOT protect against speaker/recording leakage and is kept only for the
// non-FSDD (physionet CSV) experiments and for ad-hoc runs that pass cv_fold < 0.
// The article pipeline always passes cv_fold >= 0 → build_loso_split below.
auto build_legacy_split(const GuayaquilConfig& cfg, const std::string& dataset) -> DatasetSplit
{
    DatasetSplit split;
    std::vector<Tensor> all_samples;
    std::vector<int> all_labels;

    if (dataset == "fsdd")
    {
        nn::dataLoaders::fsdd::FsddWindowDataset ds(
            cfg.dataset.dataset_root, cfg.dataset.window_size);
        all_samples = ds.windows();
        all_labels = ds.labels();
    }
    else
    {
        const auto files = collect_signal_files(cfg, dataset);
        if (files.empty()) throw std::runtime_error("No files found for dataset token: " + dataset);

        for (const auto& file : files)
        {
            const auto signal = nn::utility::read_csv_signal(file);
            const auto windows = to_window_tensor(signal, cfg.dataset.window_size);
            all_samples.insert(all_samples.end(), windows.begin(), windows.end());
        }
        all_labels.assign(all_samples.size(), 0);
    }

    if (all_samples.empty())
        throw std::runtime_error("No windows created for dataset token: " + dataset);

    {
        std::vector<std::size_t> idx(all_samples.size());
        std::iota(idx.begin(), idx.end(), 0u);
        std::mt19937 rng(cfg.experiment.seed != 0u ? cfg.experiment.seed : 42u);
        std::shuffle(idx.begin(), idx.end(), rng);

        std::vector<Tensor> s(all_samples.size());
        std::vector<int> l(all_labels.size());
        for (std::size_t i = 0; i < idx.size(); ++i)
        {
            s[i] = std::move(all_samples[idx[i]]);
            l[i] = all_labels[idx[i]];
        }
        all_samples = std::move(s);
        all_labels = std::move(l);
    }

    const std::size_t max_total = static_cast<std::size_t>(
        cfg.dataset.max_loaded_train_samples + cfg.dataset.max_validation_samples);
    if (all_samples.size() > max_total)
    {
        all_samples.resize(max_total);
        all_labels.resize(max_total);
    }

    const std::size_t val_count =
        std::min<std::size_t>(cfg.dataset.max_validation_samples, all_samples.size());
    const std::size_t train_count = all_samples.size() - val_count;

    split.train_samples.assign(
        all_samples.begin(), all_samples.begin() + static_cast<long>(train_count));
    split.val_samples.assign(
        all_samples.begin() + static_cast<long>(train_count), all_samples.end());
    split.val_labels.assign(all_labels.begin() + static_cast<long>(train_count), all_labels.end());

    // Legacy path has no held-out test speaker; downstream evaluates on val.
    return split;
}

struct GroupedWindows
{
    std::vector<Tensor> windows;
    std::vector<WindowMetadata> meta;
};

// Load every window + parallel metadata for one dataset source. FSDD and the
// (offline-resampled, 8 kHz) AudioMNIST corpus share the FSDD WAV loader; MIT-BIH
// uses the format-212 WFDB reader. A per-recording window cap keeps long
// recordings (ECG, silence-padded speech) from dominating the pooled counts;
// cap 0 = keep all (FSDD).
auto load_grouped_windows(const std::string& dataset, const GuayaquilConfig::DatasetSource& src)
    -> GroupedWindows
{
    GroupedWindows g;
    if (dataset == "fsdd" || dataset == "audiomnist")
    {
        nn::dataLoaders::fsdd::FsddWindowDataset ds(src.root, src.window_size);
        g.windows = ds.windows();
        g.meta = ds.metadata();
    }
    else if (dataset == "mitbih")
    {
        MitBihWindowDataset ds(src.root, src.window_size);
        g.windows = ds.windows();
        g.meta = ds.metadata();
    }
    else
    {
        throw std::runtime_error("load_grouped_windows: unknown grouped dataset '" + dataset +
                                 "' (expected fsdd | audiomnist | mitbih)");
    }

    if (src.max_windows_per_recording > 0)
    {
        std::vector<Tensor> w;
        std::vector<WindowMetadata> m;
        for (std::size_t i = 0; i < g.meta.size(); ++i)
        {
            if (g.meta[i].source_window_index >= src.max_windows_per_recording) continue;
            w.push_back(std::move(g.windows[i]));
            m.push_back(g.meta[i]);
        }
        g.windows = std::move(w);
        g.meta = std::move(m);
    }
    return g;
}

// Nested leave-one-group-out fold. Windows are partitioned BY SPEAKER/GROUP
// before any pooling, so no speaker/record and no source recording crosses a
// boundary. The SNN hyperparameter sweep selects on `val` only; the winning
// config is retrained on train ∪ val and evaluated once on `test`.
// Deterministic stratified subsample to <= cap windows, round-robin across recordings
// (ordered by recording_id), so every recording and every speaker keeps representation and
// per-recording counts stay as even as the cap allows. cap <= 0 or already under → no-op.
// Bounds per-epoch training cost (batch_size 1) without collapsing the LOSO structure or
// the recording-level statistical unit.
void stratified_window_cap(std::vector<Tensor>& samples,
    std::vector<WindowMetadata>& meta,
    std::vector<int>* labels,
    int cap)
{
    if (cap <= 0 || static_cast<int>(samples.size()) <= cap) return;

    std::map<int, std::vector<std::size_t>> by_rec;
    for (std::size_t i = 0; i < meta.size(); ++i) by_rec[meta[i].recording_id].push_back(i);

    std::vector<std::size_t> keep;
    keep.reserve(static_cast<std::size_t>(cap));
    bool progress = true;
    while (static_cast<int>(keep.size()) < cap && progress)
    {
        progress = false;
        for (auto& [rid, idxs] : by_rec)
        {
            if (idxs.empty()) continue;
            keep.push_back(idxs.front());
            idxs.erase(idxs.begin());
            progress = true;
            if (static_cast<int>(keep.size()) >= cap) break;
        }
    }
    std::sort(keep.begin(), keep.end());

    std::vector<Tensor> s;
    std::vector<WindowMetadata> m;
    std::vector<int> l;
    s.reserve(keep.size());
    m.reserve(keep.size());
    for (std::size_t k : keep)
    {
        s.push_back(std::move(samples[k]));
        m.push_back(meta[k]);
        if (labels != nullptr) l.push_back((*labels)[k]);
    }
    samples = std::move(s);
    meta = std::move(m);
    if (labels != nullptr) *labels = std::move(l);
}

auto build_loso_split(const GuayaquilConfig& cfg, const std::string& dataset, int cv_fold)
    -> DatasetSplit
{
    const GuayaquilConfig::DatasetSource src = cfg.dataset.resolve(dataset);
    const GroupedWindows loaded = load_grouped_windows(dataset, src);
    const auto& windows = loaded.windows;
    const auto& meta = loaded.meta;
    if (windows.empty())
        throw std::runtime_error(
            "build_loso_split: no windows loaded for dataset '" + dataset + "'");

    const SpeakerFoldAssignment fold = assign_speaker_fold(meta, cv_fold, src.cv_num_folds);

    DatasetSplit split;
    split.train_speakers = fold.train_speakers;
    split.val_speaker = fold.val_speaker;
    split.test_speaker = fold.test_speaker;

    for (std::size_t i : fold.test_idx)
    {
        split.test_samples.push_back(windows[i]);
        split.test_meta.push_back(meta[i]);
        split.test_labels.push_back(meta[i].digit);
    }
    for (std::size_t i : fold.val_idx)
    {
        split.val_samples.push_back(windows[i]);
        split.val_meta.push_back(meta[i]);
        split.val_labels.push_back(meta[i].digit);
    }
    for (std::size_t i : fold.train_idx)
    {
        split.train_samples.push_back(windows[i]);
        split.train_meta.push_back(meta[i]);
    }

    // Per-fold stratified window caps (bound training cost; 0 = unlimited).
    stratified_window_cap(
        split.test_samples, split.test_meta, &split.test_labels, src.loso_max_test_windows);
    stratified_window_cap(
        split.val_samples, split.val_meta, &split.val_labels, src.loso_max_val_windows);
    stratified_window_cap(
        split.train_samples, split.train_meta, nullptr, src.loso_max_train_windows);

    // Deterministic shuffle of the train windows, carrying the parallel metadata.
    {
        std::vector<std::size_t> idx(split.train_samples.size());
        std::iota(idx.begin(), idx.end(), 0u);
        std::mt19937 rng(cfg.experiment.seed != 0u ? cfg.experiment.seed : 42u);
        std::shuffle(idx.begin(), idx.end(), rng);

        std::vector<Tensor> s;
        std::vector<WindowMetadata> mm;
        s.reserve(idx.size());
        mm.reserve(idx.size());
        for (std::size_t j : idx)
        {
            s.push_back(std::move(split.train_samples[j]));
            mm.push_back(split.train_meta[j]);
        }
        split.train_samples = std::move(s);
        split.train_meta = std::move(mm);
    }

    if (split.train_samples.empty() || split.val_samples.empty() || split.test_samples.empty())
        throw std::runtime_error("build_loso_split: fold " + std::to_string(cv_fold) +
                                 " produced an empty train/val/test partition.");

    return split;
}

} // namespace

auto build_split(const GuayaquilConfig& cfg, const std::string& dataset, int cv_fold)
    -> DatasetSplit
{
    if (cv_fold >= 0 && (dataset == "fsdd" || dataset == "audiomnist" || dataset == "mitbih"))
        return build_loso_split(cfg, dataset, cv_fold);
    return build_legacy_split(cfg, dataset);
}

} // namespace guayaquil
