#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01DatasetSplit.hpp"
#include "nlohmann/json.hpp"

namespace meeting01
{

// Pure speaker-disjoint fold assignment — the leakage-safety core, factored out so
// it can be unit-tested on synthetic metadata without a WAV corpus.
//
// Given per-window metadata, `cv_fold` in [0, num_folds), and num_folds <= the
// number of distinct speaker_ids present: the sorted speaker ids are partitioned
// into `num_folds` contiguous groups (a group holds exactly one speaker when
// distinct == num_folds, reducing to plain leave-one-speaker-out; a group holds
// several speakers otherwise, e.g. AudioMNIST's 60 speakers over 6 folds).
//   test  = group[cv_fold]
//   val   = group[(cv_fold + 1) mod num_folds]   (rotating validation)
//   train = every other group
// Returns window indices (into `meta`) per split, in ascending order.
//
// Throws std::runtime_error if cv_fold is out of range, if the distinct speaker
// count < num_folds, or if any of the three partitions would be empty.
struct SpeakerFoldAssignment
{
    std::vector<std::size_t> train_idx;
    std::vector<std::size_t> val_idx;
    std::vector<std::size_t> test_idx;
    std::vector<std::string> train_speakers; // sorted, unique
    std::string val_speaker;
    std::string test_speaker;
};

auto assign_speaker_fold(std::span<const WindowMetadata> meta, int cv_fold, int num_folds)
    -> SpeakerFoldAssignment;

// Keeps at most `cap` windows, balanced by speaker, then digit, then recording, in a seeded
// order, so no sorted-list prefix survives a cap (rationale in Meeting01Dataset.cpp). Same seed
// -> same result. cap <= 0, or a pool within the cap, keeps everything.
void stratified_window_cap(std::vector<Tensor>& samples,
    std::vector<WindowMetadata>& meta,
    std::vector<int>* labels,
    int cap,
    unsigned int seed);

// Keeps at most `cap` windows of each recording, drawn uniformly across it (not its first
// `cap`; rationale in Meeting01Dataset.cpp). Same seed -> same result. cap <= 0 keeps everything.
void cap_windows_per_recording(
    std::vector<Tensor>& windows, std::vector<WindowMetadata>& meta, int cap, unsigned int seed);

// What a split part is made of: windows per speaker, per digit and per source window index.
// Written into every fold's split manifest, because a cap's bias is invisible in a window count.
auto split_composition_json(const std::vector<WindowMetadata>& meta) -> nlohmann::json;

// config_hash extended by the windows themselves (their ids and values, every split). The GA
// checkpoints are stamped with it: a validation score is a function of the data, and a code
// change that alters the windows changes neither the profile nor, unless someone remembers to
// bump it, results_format.
auto checkpoint_hash(std::size_t config_hash, const DatasetSplit& split) -> std::size_t;

// cv_fold >= 0 is REQUIRED — nested leave-one-speaker/group-out fold, speaker- and
// recording-disjoint across train/val/test. There is no pooled/shuffled fallback
// (removed 2026-09-23; it leaked speakers/recordings across the split). Throws if
// cv_fold < 0 or dataset is not one of the grouped loaders (fsdd | audiomnist |
// mitbih | eegmmidb | siena).
auto build_split(const Meeting01Config& cfg, const std::string& dataset, int cv_fold)
    -> DatasetSplit;

} // namespace meeting01
