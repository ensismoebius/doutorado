#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01DatasetSplit.hpp"
#include "Meeting01EpochHistory.hpp"
#include "Meeting01ResultRow.hpp"

namespace meeting01
{

// Version of the results this binary writes, stamped as "results_format" into every JSON
// manifest of a fold and every checkpoint. The post-processing scripts accept exactly this
// value (RESULTS_FORMAT in scripts/pipeline/meeting01/meeting01_results.py) and refuse any
// other, so results whose numbers cannot be compared never reach a paper table; a
// checkpoint of another format is not restored (Meeting01Checkpoint.cpp). Bump BOTH
// whenever a change makes new numbers incomparable with old ones.
//   absent  binaries before 2026-10-06. Up to commit 6f332734 a zero-padded window (the
//           last of a FSDD / AudioMNIST recording) was z-scored together with its padding
//           and scored over it; from that commit the LSTM/GRU/Transformer-AE path crashed
//           on its first batch. Per-window rows had no encoding, baseline test rows said
//           train_ms 0, and there were no reference target dumps.
//   2       padding excluded from normalization, every loss and every error
//           (valid_length); target dumps for the references; per-window encoding and
//           baseline train_ms written.
inline constexpr int kResultsFormat = 2;

// Writes the inputs of the PCA / mean-frame reference baselines
// (scripts/pipeline/meeting01/03_meeting01_pca_mean_baselines.py) for one fold, per split
// part p in {train, val, test}:
//
//   <dir>/<stem>_target_<p>_windows.npy       float32 (N_p, window_size), one row per window
//   <dir>/<stem>_target_<p>_windows_meta.csv  speaker_id,recording_id,window_id,
//                                             source_window_index,valid_length
//
// A row is the window itself, in sample order: the analog signal every model family is
// scored against. Those families reconstruct make_reconstruction_target(window), the
// window held constant over T steps; a reference's per-window MSE against that T-fold copy
// equals its MSE against the window, so one dump serves every encoding. valid_length lets
// the reference use the same activity mask as the families (zero-padded tails excluded).
//
// Throws std::runtime_error on an empty part, on windows of unequal width, on a
// valid_length outside [0, width], or on a file it cannot write.
void write_reference_inputs(
    const std::filesystem::path& dir, const std::string& stem, const DatasetSplit& split);

void write_rows_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_publication_table(const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_summary_json(const std::filesystem::path& path,
    const Meeting01Config& cfg,
    std::size_t cfg_hash,
    const std::vector<ResultRow>& rows);

void write_latex_exports(const std::filesystem::path& dir,
    const std::string& run_tag,
    const Meeting01Config& cfg,
    const std::vector<ResultRow>& rows);

void validate_repeat_determinism(const Meeting01Config& cfg, const std::vector<ResultRow>& rows);

void write_pgfplots_summary_dat(
    const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_pgfplots_sweep_dat(
    const std::filesystem::path& path, const std::vector<ResultRow>& rows);

void write_epoch_history_dat(const std::filesystem::path& path,
    const std::string& model,
    const std::string& encoding,
    const std::string& architecture,
    float v_th,
    float alpha,
    int run_id,
    const EpochHistory& history);

void write_batch_convergence_dat(const std::filesystem::path& path,
    const std::string& model,
    const std::string& encoding,
    const std::string& architecture,
    float v_th,
    float alpha,
    int run_id,
    const EpochHistory& history);

} // namespace meeting01
