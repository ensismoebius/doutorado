#include "../include/Meeting01Dataset.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

#include "Meeting01Eeg.hpp"
#include "Meeting01MitBih.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "utility/RandomIndexCrop.hpp"

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

// Deterministic stratified subsample to <= cap windows.
//
// THE PROBLEM. A cap keeps a sample of a larger pool, and the pool is sorted: FSDD's files run
// 0_george_0, 0_george_1, ... 9_yweweler_49. Until 2026-11-04 this function walked the
// recordings in that order, one window each, and stopped at the cap. On FSDD fold 0 that was
//
//   train: digits 0 1 2 3 4              (1000 windows, cap reached in the middle of the list)
//   val:   digits 0 1 2                  (150 windows)
//   test:  digits 0 1 2 3 4 5 6 7 8 9    (1500 windows, above its 500 recordings: the walk wrapped)
//
// EEGMMIDB's validation saw 11 of its 18 subjects, and AudioMNIST's training would have seen 2
// of 40 speakers. Nothing failed: the count was exactly the cap, and the docstring said
// "stratified".
//
// THE RULE. Who appears is balanced level by level, from the unit that generalizes outwards:
//
//   speaker -> digit -> recording -> window
//
// Each pass visits every speaker once, in an order drawn anew per pass (the speakers cut off by
// the cap in the last pass are therefore a random few, not the last of the list). A speaker
// gives its next digit in rotation, the digit its next recording in rotation, the recording its
// next window. Digits, recordings and windows rotate in shuffled orders, so the start of a
// sorted list has no advantage at any level. A dataset with no digit (EEG, digit -1) has one
// digit per subject, and the subject simply rotates through its recordings.
//
// `seed` fixes the draw. build_loso_split passes the profile's experiment.seed, not the run
// seed: every repeat of a fold sees the same windows, which is why the PCA and mean references,
// fitted once, score the same test set as the networks. The draw's own variance is therefore
// not sampled by the repeats. Measured on FSDD fold 0: PCA k=16 moved 0.047 between two balanced
// draws.
//
// Defined here (in meeting01:: proper, not the anonymous namespace below) because the header
// declares it with external linkage, so it is unit-testable on synthetic metadata
// (split_audit_gtest.cpp).
void stratified_window_cap(std::vector<Tensor>& samples,
    std::vector<WindowMetadata>& meta,
    std::vector<int>* labels,
    int cap,
    unsigned int seed)
{
    if (cap <= 0 || static_cast<int>(samples.size()) <= cap) return;

    // One rotation: `items` in a shuffled order, `next` the one whose turn it is.
    struct Recording
    {
        std::vector<std::size_t> windows;
    };
    struct Digit
    {
        std::vector<Recording> recordings;
        std::size_t next = 0;
    };
    struct Speaker
    {
        std::vector<Digit> digits;
        std::size_t next = 0;
    };

    std::map<int, std::map<int, std::map<int, std::vector<std::size_t>>>> tree;
    for (std::size_t i = 0; i < meta.size(); ++i)
        tree[meta[i].speaker_id][meta[i].digit][meta[i].recording_id].push_back(i);

    const nn::transforms::RandomIndexCrop crop(seed);
    std::mt19937 rng(seed);
    std::vector<Speaker> speakers;
    speakers.reserve(tree.size());
    for (auto& [speaker_id, by_digit] : tree)
    {
        Speaker speaker;
        for (auto& [digit, by_recording] : by_digit)
        {
            Digit d;
            for (auto& [recording_id, windows] : by_recording)
                d.recordings.push_back(Recording{crop(std::move(windows))});
            std::shuffle(d.recordings.begin(), d.recordings.end(), rng);
            speaker.digits.push_back(std::move(d));
        }
        std::shuffle(speaker.digits.begin(), speaker.digits.end(), rng);
        speakers.push_back(std::move(speaker));
    }

    // The next window of the entry whose turn it is, or false when `list` has none left. An
    // emptied entry leaves the rotation, so the others keep their turns evenly.
    const auto take_recording = [](Digit& d, std::size_t& out) -> bool
    {
        while (!d.recordings.empty())
        {
            d.next %= d.recordings.size();
            Recording& r = d.recordings[d.next];
            if (r.windows.empty())
            {
                d.recordings.erase(d.recordings.begin() + static_cast<std::ptrdiff_t>(d.next));
                continue;
            }
            out = r.windows.back();
            r.windows.pop_back();
            ++d.next;
            return true;
        }
        return false;
    };
    const auto take_digit = [&take_recording](Speaker& s, std::size_t& out) -> bool
    {
        while (!s.digits.empty())
        {
            s.next %= s.digits.size();
            if (take_recording(s.digits[s.next], out))
            {
                ++s.next;
                return true;
            }
            s.digits.erase(s.digits.begin() + static_cast<std::ptrdiff_t>(s.next));
        }
        return false;
    };

    std::vector<std::size_t> keep;
    keep.reserve(static_cast<std::size_t>(cap));
    std::vector<std::size_t> order(speakers.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    bool progress = true;
    while (static_cast<int>(keep.size()) < cap && progress)
    {
        progress = false;
        std::shuffle(order.begin(), order.end(), rng);
        for (std::size_t s : order)
        {
            std::size_t window = 0;
            if (!take_digit(speakers[s], window)) continue;
            keep.push_back(window);
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

// Keeps at most `cap` windows of each recording, drawn across the whole recording.
//
// The rule used to be "the first cap windows": EEGMMIDB's first 64 s of each run (with 40 windows
// of 1.6 s), Siena's first 20 s of each file, AudioMNIST's windows 0 and 1. A recording's start
// is not a sample of it: an EEG run's task, a Siena file's content and a spoken digit's voiced
// part all lie elsewhere. The draw is uniform and does not look at the signal: choosing windows
// by their content (voice activity, say) would be a selection the protocol would have to defend.
void cap_windows_per_recording(
    std::vector<Tensor>& windows, std::vector<WindowMetadata>& meta, int cap, unsigned int seed)
{
    if (cap <= 0) return;

    std::map<int, std::vector<std::size_t>> by_recording;
    for (std::size_t i = 0; i < meta.size(); ++i) by_recording[meta[i].recording_id].push_back(i);

    const nn::transforms::RandomIndexCrop crop(seed);
    std::vector<std::size_t> keep;
    keep.reserve(by_recording.size() * static_cast<std::size_t>(cap));
    for (auto& [recording_id, idxs] : by_recording)
    {
        std::vector<std::size_t> drawn = crop(std::move(idxs));
        if (drawn.size() > static_cast<std::size_t>(cap))
            drawn.resize(static_cast<std::size_t>(cap));
        keep.insert(keep.end(), drawn.begin(), drawn.end());
    }
    std::sort(keep.begin(), keep.end());

    std::vector<Tensor> w;
    std::vector<WindowMetadata> m;
    w.reserve(keep.size());
    m.reserve(keep.size());
    for (std::size_t k : keep)
    {
        w.push_back(std::move(windows[k]));
        m.push_back(meta[k]);
    }
    windows = std::move(w);
    meta = std::move(m);
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
// cap 0 = keep all (FSDD). `seed` draws which windows; the draw happens before the fold is
// assigned, so a recording keeps the same windows in every fold.
auto load_grouped_windows(
    const std::string& dataset, const Meeting01Config::DatasetSource& src, unsigned int seed)
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
        // Mains frequency is a property of the recording SITE, not the signal itself: Siena
        // was recorded in Italy (50 Hz mains), eegmmidb at PhysioNet/US sites (60 Hz) -- see
        // .wiki/Core/DataLoaders.md for the literature this bandpass+notch choice is based on.
        const double notch_hz = (dataset == "siena") ? 50.0 : 60.0;
        EegWindowDataset ds(src.root, src.window_size, src.sample_rate, notch_hz);
        g.windows = ds.windows();
        g.meta = ds.metadata();
    }
    else
    {
        throw std::runtime_error("load_grouped_windows: unknown grouped dataset '" + dataset +
                                 "' (expected fsdd | audiomnist | mitbih | eegmmidb | siena)");
    }

    cap_windows_per_recording(g.windows, g.meta, src.max_windows_per_recording, seed);
    return g;
}

// Nested leave-one-group-out fold. Windows are partitioned BY SPEAKER/GROUP
// before any pooling, so no speaker/record and no source recording crosses a
// boundary. The SNN hyperparameter sweep selects on `val` only; the winning
// config is retrained on train ∪ val and evaluated once on `test`.
auto build_loso_split(const Meeting01Config& cfg, const std::string& dataset, int cv_fold)
    -> DatasetSplit
{
    const Meeting01Config::DatasetSource src = cfg.dataset.resolve(dataset);
    // Distinct +0/+1/+2/+3 seed offsets so the draws of test/val/train and of the per-recording
    // cap don't repeat one another.
    const unsigned int base_seed = cfg.experiment.seed != 0u ? cfg.experiment.seed : 42u;
    const GroupedWindows loaded = load_grouped_windows(dataset, src, base_seed + 3);
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
    stratified_window_cap(split.test_samples,
        split.test_meta,
        &split.test_labels,
        src.loso_max_test_windows,
        base_seed);
    stratified_window_cap(split.val_samples,
        split.val_meta,
        &split.val_labels,
        src.loso_max_val_windows,
        base_seed + 1);
    stratified_window_cap(
        split.train_samples, split.train_meta, nullptr, src.loso_max_train_windows, base_seed + 2);

    // Deterministic shuffle of the train windows, carrying the parallel metadata.
    {
        std::vector<std::size_t> idx(split.train_samples.size());
        std::iota(idx.begin(), idx.end(), 0u);
        std::mt19937 rng(base_seed);
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

auto split_composition_json(const std::vector<WindowMetadata>& meta) -> nlohmann::json
{
    std::map<std::string, int> speakers;
    std::map<std::string, int> digits;
    std::map<std::string, int> window_indices;
    for (const auto& m : meta)
    {
        ++speakers[m.speaker];
        ++digits[std::to_string(m.digit)];
        ++window_indices[std::to_string(m.source_window_index)];
    }
    return {{"speakers", speakers}, {"digits", digits}, {"source_window_index", window_indices}};
}

auto checkpoint_hash(std::size_t config_hash, const DatasetSplit& split) -> std::size_t
{
    // FNV-1a over the id and the bytes of every value, starting from config_hash.
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    auto hash = static_cast<std::uint64_t>(config_hash) ^ 14695981039346656037ULL;
    const auto mix = [&hash](std::uint32_t word)
    {
        for (int shift = 0; shift < 32; shift += 8)
        {
            hash ^= (word >> static_cast<unsigned>(shift)) & 0xFFU;
            hash *= kPrime;
        }
    };
    const auto mix_part =
        [&mix](const std::vector<Tensor>& samples, const std::vector<WindowMetadata>& meta)
    {
        mix(static_cast<std::uint32_t>(samples.size()));
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            mix(static_cast<std::uint32_t>(meta[i].window_id));
            for (nn::Index k = 0; k < static_cast<nn::Index>(samples[i].size()); ++k)
                mix(std::bit_cast<std::uint32_t>(samples[i].at(k)));
        }
    };
    mix_part(split.train_samples, split.train_meta);
    mix_part(split.val_samples, split.val_meta);
    mix_part(split.test_samples, split.test_meta);
    return static_cast<std::size_t>(hash);
}

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
