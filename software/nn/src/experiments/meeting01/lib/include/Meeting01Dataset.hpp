#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01DatasetSplit.hpp"

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

// Deterministic stratified subsample to <= cap windows, round-robin across recordings
// (grouped by WindowMetadata::recording_id) so every recording keeps representation and
// per-recording counts stay as even as the cap allows. cap <= 0 or already under cap → no-op.
// Factored out (like assign_speaker_fold above) so it can be unit-tested on synthetic
// metadata without a WAV corpus.
//
// `seed` shuffles each recording's own candidate windows before the round-robin picks its
// next one — this is what fixes the "AudioMNIST window degeneracy" bug (see
// .wiki/Experiments/Meeting01.md): without it, the round-robin always took a recording's
// LOWEST source_window_index first, and a corpus with more recordings than the cap needs
// (true for AudioMNIST) exhausts the cap during the very first pass, so every kept window is
// index 0 — a near-silent recording lead-in for AudioMNIST specifically, not the spoken
// digit. Same seed → same result (reproducible); different seed → different window-index mix
// while the per-recording fairness guarantee is unchanged either way.
void stratified_window_cap(std::vector<Tensor>& samples,
    std::vector<WindowMetadata>& meta,
    std::vector<int>* labels,
    int cap,
    unsigned int seed);

// cv_fold >= 0 is REQUIRED — nested leave-one-speaker/group-out fold, speaker- and
// recording-disjoint across train/val/test. There is no pooled/shuffled fallback
// (removed 2026-09-23; it leaked speakers/recordings across the split). Throws if
// cv_fold < 0 or dataset is not one of the grouped loaders (fsdd | audiomnist |
// mitbih | eegmmidb | siena).
auto build_split(const Meeting01Config& cfg, const std::string& dataset, int cv_fold)
    -> DatasetSplit;

} // namespace meeting01
