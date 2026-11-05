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
// other, so results whose numbers cannot be compared never reach a paper table; a GA
// checkpoint of another format is refused on resume (Meeting01GaCheckpoint.hpp). Bump BOTH
// whenever a change makes new numbers incomparable with old ones.
//   absent  binaries before 2026-10-06. Up to commit 6f332734 a zero-padded window (the
//           last of a FSDD / AudioMNIST recording) was z-scored together with its padding
//           and scored over it; from that commit the LSTM/GRU/Transformer-AE path crashed
//           on its first batch. Per-window rows had no encoding, baseline test rows said
//           train_ms 0, and there were no reference target dumps.
//   2       interim, superseded 2026-10-07 (no production fold was run with it): padding
//           excluded from normalization, every loss and every error (valid_length); target
//           dumps for the references; per-window encoding and baseline train_ms written.
//           But the sequence models' frames were a strided gather (one window sample at
//           alternate steps per frame) and the LSTM/GRU cost counted one stack of two.
//   3       to_lstm_frames cuts consecutive samples of one step; the LSTM/GRU `macs` count
//           the encoder and decoder stacks (and so steer the architecture search).
//   4       the LSTM/GRU/Transformer-AE read the window at model.recurrent_time_steps (1 in
//           production: S = 32 frames) instead of repeating it at the SNN's time_steps
//           (S = 512), where they did not learn (loss ~0.99). Their encoding gene is `direct`
//           only. No production fold ran with format 3.
//   5       2026-11-01. The SNN-AE gains model.snn_time_pool / model.snn_decoder_neuron
//           (production: the mean over T of the latent, an analog decoder run once, T = 4),
//           from the gap diagnosis in .wiki/Experiments/Meeting01.md. `spike_rate` is the
//           spiking fraction of the hidden LIF neurons (was the fraction of output values
//           > 0, ~0.5 for any model) and NaN for non-spiking models; `energy` (spikes +
//           10 x MACs, unitless) is replaced by `acs` and `arith_energy_nj` (4.6 pJ/MAC,
//           0.9 pJ/AC); the SNN's `macs` is counted from its forward passes (was the dense
//           estimate x T). No production fold ran with format 4.
//   6       2026-11-02. T (model.time_steps) is a gene of the SNN-AE search
//           (evaluation.ga.snn.time_steps_min/max), so each selection manifest and each
//           GA candidate records its own `time_steps`. No production fold ran with format 5.
//   7       2026-11-03. Every training's patience is armed only once the monitor loss is
//           (1 - training.early_stop_min_skill) x the mean predictor's, and the best monitor
//           epoch's network is kept (Meeting01Stopping.hpp): format 6 stopped networks still on
//           the mean-predictor plateau and kept the last epoch's weights. Rows gain the
//           memory-aware energy (`energy_nj`, its ops/memory split, `mem_accesses`, and the
//           one-memory-size repricings `energy_nj_sram8k/32k/1m`; Meeting01Energy.hpp), the
//           paper's primary energy; `arith_energy_nj` stays as the arithmetic-only figure.
//           No production fold ran with format 6.
//   8       2026-11-03. An SNN-AE input that is the same at every step (direct code through the
//           dense or conv1d transform; snn_input_repeats_every_step) has its transform and first
//           Linear counted once per window in `energy_nj` and its repricings, `macs`, `acs` and
//           `arith_energy_nj` (format 7 counted them T times, as the code runs); rows gain
//           `energy_nj_as_executed`, that T-times count, as a check (Meeting01Energy.hpp). GA
//           evaluations log their own identity in the events file (format 7 logged every
//           evaluation after a final fit under that fit's identity). No production fold ran with
//           format 7; the pre-validation job 5315335 did.
//   9       2026-11-04. The run seed reaches every training: the LSTM/GRU/Transformer-AE draw
//           their weights from it (format 8 started all three from weights fixed in the model
//           code, so their seeds sampled no initialisation) and every family's window order
//           comes from it (TrainerConfig::sampler_shuffle_seed; format 8 used 42 for every
//           seed). The SNN GA's cost objective counts a repeated input's first layer once,
//           as energy_nj does (format 8 charged it T times there), so the search changes.
//           No production fold ran with format 8; T1 (results/meeting01_t1) did, locally.
//   10      2026-11-04. The data changed, for every dataset. (1) Speech: Wav::read decoded the
//           44 header bytes as 22 samples, so window 0 of every FSDD and AudioMNIST recording
//           was mostly header and the last 22 real samples were lost; the networks could score
//           by memorizing it (T1, FSDD: error 0.113 on window 0, 1.000 on the others in a
//           replica). (2) stratified_window_cap walked the sorted recordings and stopped at the
//           cap (FSDD fold 0 trained on digits 0-4); it now balances speaker, digit, recording.
//           (3) max_windows_per_recording kept each recording's first windows; it now draws
//           across the recording. (4) The 36 EEGMMIDB files at 128 Hz were filtered and windowed
//           as 160 Hz; each file is now filtered at its own rate and converted to the dataset's.
//           Blocks are still assigned on all 109 subjects, so no subject moved fold. GA
//           checkpoints are also stamped with the windows (checkpoint_hash). The split manifest
//           records each part's composition. No production fold ran with format 9; the
//           pre-validation job 5316016 and T1 did.
//   11      2026-11-05. The GA's re-score of the Pareto front uses winner_seeds fresh seeds
//           (format 10 averaged in the search's own score, which is the one that selected the
//           member, so the reported val_mse stayed biased low and the pick could change).
//           Rows gain energy_nj_encoder, hidden_silent_share, hidden_always_share,
//           lif_v_th_mean, lif_v_th_min, lif_beta_mean and train_mse (RunMetrics). Windows and
//           training are unchanged. No production fold ran with format 10; the stopping calibration
//           did.
inline constexpr int kResultsFormat = 11;

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

// The per-fold `<tag>_comparative_metrics.csv`. It doubles as 01_meeting01_run_loso.sh's
// RESUME=1 "this fold finished" marker (a fold whose file is non-empty is skipped), so it is
// written to `<path>.tmp` and renamed into place: a process killed mid-write (a cluster time
// limit, a scancel) leaves the previous state or the complete file, never a truncated one that
// RESUME would take for a finished fold. Throws std::runtime_error on a file it cannot write.
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
