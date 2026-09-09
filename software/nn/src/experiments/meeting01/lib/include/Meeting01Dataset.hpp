#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "GuayaquilConfig.hpp"
#include "GuayaquilDatasetSplit.hpp"

namespace guayaquil
{

auto to_window_tensor(const nn::Tensor& signal, int window_size) -> std::vector<nn::Tensor>;

auto collect_signal_files(const GuayaquilConfig& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>;

// Pure speaker-disjoint fold assignment — the leakage-safety core, factored out so
// it can be unit-tested on synthetic metadata without a WAV corpus.
//
// Given per-window metadata, `cv_fold` in [0, num_folds), and num_folds == the
// number of distinct speaker_ids present:
//   test  = { sorted_speaker_id[cv_fold] }
//   val   = { sorted_speaker_id[(cv_fold + 1) mod num_folds] }   (rotating validation)
//   train = every other speaker
// Returns window indices (into `meta`) per split, in ascending order.
//
// Throws std::runtime_error if cv_fold is out of range, if the distinct speaker
// count != num_folds, or if any of the three partitions would be empty.
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

// cv_fold < 0 → legacy pooled split; cv_fold >= 0 → nested leave-one-speaker-out
// fold (FSDD only), speaker- and recording-disjoint across train/val/test.
auto build_split(const GuayaquilConfig& cfg, const std::string& dataset, int cv_fold = -1)
    -> DatasetSplit;

} // namespace guayaquil
