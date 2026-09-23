#include "../include/Meeting01Dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

#include "Meeting01Eeg.hpp"
#include "Meeting01MitBih.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"

namespace meeting01
{

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
auto load_grouped_windows(const std::string& dataset, const Meeting01Config::DatasetSource& src)
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
    else if (dataset == "eegmmidb" || dataset == "siena")
    {
        EegWindowDataset ds(src.root, src.window_size);
        g.windows = ds.windows();
        g.meta = ds.metadata();
    }
    else
    {
        throw std::runtime_error("load_grouped_windows: unknown grouped dataset '" + dataset +
                                 "' (expected fsdd | audiomnist | mitbih | eegmmidb | siena)");
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

auto build_loso_split(const Meeting01Config& cfg, const std::string& dataset, int cv_fold)
    -> DatasetSplit
{
    const Meeting01Config::DatasetSource src = cfg.dataset.resolve(dataset);
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

auto build_split(const Meeting01Config& cfg, const std::string& dataset, int cv_fold)
    -> DatasetSplit
{
    if (cv_fold < 0)
        throw std::runtime_error(
            "build_split: dataset.cv_fold must be >= 0. The legacy pooled/shuffled split "
            "(cv_fold < 0) was removed 2026-09-23 -- it pooled every window across speakers "
            "then shuffled, so the same speaker/recording could land in both train and "
            "validation (the leakage defect a reviewer flagged as strong-reject on "
            "submission 71; see .wiki/Experiments/Meeting01.md). Remedy: set dataset.cv_fold "
            "(and dataset.cv_num_folds) in the profile, or pass --cv-fold on the CLI.");
    if (dataset != "fsdd" && dataset != "audiomnist" && dataset != "mitbih" &&
        dataset != "eegmmidb" && dataset != "siena")
        throw std::runtime_error("build_split: unknown dataset '" + dataset +
                                 "' (expected fsdd | audiomnist | mitbih | eegmmidb | siena)");
    return build_loso_split(cfg, dataset, cv_fold);
}

} // namespace meeting01
