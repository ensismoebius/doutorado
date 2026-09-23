#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <string>

#include "../include/Meeting01Checkpoint.hpp"
#include "../include/Meeting01Cli.hpp"
#include "../include/Meeting01Dataset.hpp"
#include "../include/Meeting01Evaluation.hpp"
#include "../include/Meeting01EventCallback.hpp"
#include "../include/Meeting01Events.hpp"
#include "../include/Meeting01GaGenome.hpp"
#include "../include/Meeting01GaSearch.hpp"
#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Output.hpp"
#include "../include/Meeting01PerWindow.hpp"
#include "../include/Meeting01RecurrentGaFitness.hpp"
#include "../include/Meeting01RecurrentGaGenome.hpp"
#include "../include/Meeting01Runner.hpp"
#include "../include/Meeting01Training.hpp"
#include "../include/Meeting01TransformerGaFitness.hpp"
#include "../include/Meeting01TransformerGaGenome.hpp"
#include "Meeting01AeCommon.hpp"
#include "cnpy.h"
#include "logging/Logger.hpp" // IWYU pragma: keep — provides NN_LOG_* macros
#include "nlohmann/json.hpp"
#include "progress/ProgressManager.hpp"
#include "serialization/NetworkSerializer.hpp"
#include "utility/progress.hpp"

using nn::models::autoencoder::AutoencoderConfig;
using nn::models::autoencoder::ProtocolSpikingAutoencoder;

// Helper to extract sizes from layer specs
auto extract_layer_sizes(const std::vector<std::string>& specs)
{
    std::vector<int> sizes;
    for (const auto& spec : specs)
    {
        std::stringstream ss(spec);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ':'))
        {
            parts.push_back(token);
        }
        if (parts.size() >= 2 && parts[0] == "linear")
        {
            try
            {
                sizes.push_back(std::stoi(parts[1]));
            }
            catch (...)
            {
            }
        }
    }
    return sizes;
}

auto active_backend_name() -> std::string
{
#if defined(NN_BACKEND_DEVICE)
    return "device";
#else
    return "xtensor";
#endif
}

auto sanitize_name(const std::string& raw) -> std::string
{
    std::string out;
    out.reserve(raw.size());
    for (char c : raw)
    {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '-' || c == '_')
        {
            out.push_back(c);
        }
        else
        {
            out.push_back('_');
        }
    }
    return out.empty() ? std::string("artifact") : out;
}

auto save_state_dict_text(
    const std::filesystem::path& path, const std::map<std::string, nn::Tensor>& state_dict) -> bool
{
    std::ofstream out(path);
    if (!out.is_open())
    {
        return false;
    }

    out << "# LSTM state_dict dump (name rows cols values...)\n";
    for (const auto& [name, tensor] : state_dict)
    {
        out << name << ' ' << tensor.rows() << ' ' << tensor.cols();
        for (nn::Index r = 0; r < tensor.rows(); ++r)
        {
            for (nn::Index c = 0; c < tensor.cols(); ++c)
            {
                out << ' ' << tensor.at(r, c);
            }
        }
        out << '\n';
    }

    return out.good();
}

auto save_parameter_list_text(
    const std::filesystem::path& path, std::span<nn::Tensor*> parameters, const std::string& prefix)
    -> bool
{
    std::ofstream out(path);
    if (!out.is_open())
    {
        return false;
    }

    out << "# Parameter dump (name rows cols values...)\n";
    for (std::size_t i = 0; i < parameters.size(); ++i)
    {
        const nn::Tensor* tensor = parameters[i];
        if (tensor == nullptr)
        {
            continue;
        }

        out << prefix << '.' << i << ' ' << tensor->rows() << ' ' << tensor->cols();
        for (nn::Index r = 0; r < tensor->rows(); ++r)
        {
            for (nn::Index c = 0; c < tensor->cols(); ++c)
            {
                out << ' ' << tensor->at(r, c);
            }
        }
        out << '\n';
    }

    return out.good();
}

namespace meeting01
{

namespace
{

// Per-fold output filename stem: "<run_tag>_<dataset>_fold<f>" under nested LOSO,
// "<run_tag>" otherwise. The dataset segment keeps folds of FSDD / AudioMNIST /
// MIT-BIH from clobbering each other when each (dataset, fold) runs as its own
// process.
auto fold_output_tag(const Meeting01Config& c, const std::string& dataset) -> std::string
{
    std::string t = c.experiment.run_tag;
    if (c.dataset.cv_fold >= 0)
    {
        if (!dataset.empty()) t += "_" + dataset;
        t += "_fold" + std::to_string(c.dataset.cv_fold);
    }
    return t;
}

// Serializes a completed ResultRow as a `config_end` event for the live monitor.
// `config_id` / `role` identify the training; every RunMetrics field is passed
// through at full precision (NaN → null). No-op when the events sink is closed.
void emit_config_end(const ResultRow& row, const std::string& config_id, const std::string& role)
{
    ExperimentEvents::instance().emit("config_end",
        {{"config_id", config_id},
            {"model", row.model},
            {"encoding", row.encoding},
            {"role", role},
            {"architecture", row.architecture},
            {"hyperparams",
                row.model == "snn-ae" ? nlohmann::json{{"v_th", row.v_th},
                                            {"alpha", row.alpha},
                                            {"architecture", row.architecture}}
                                      : nlohmann::json::object()},
            {"run_id", row.run_id},
            {"seed", row.seed},
            {"split", row.split},
            {"cv_fold", row.cv_fold},
            {"metrics",
                {{"mse", jnum(row.metrics.mse)},
                    {"mae", jnum(row.metrics.mae)},
                    {"r2", jnum(row.metrics.r2)},
                    {"precision", jnum(row.metrics.precision)},
                    {"recall", jnum(row.metrics.recall)},
                    {"f1", jnum(row.metrics.f1)},
                    {"spike_rate", jnum(row.metrics.spike_rate)},
                    {"train_ms", jnum(row.metrics.train_ms)},
                    {"infer_ms", jnum(row.metrics.infer_ms)},
                    {"param_count", row.metrics.parameter_count},
                    {"macs", row.metrics.macs}}}});
}

// Resolved output locations for one experiment run: raw metrics/summary files, saved model
// dumps, and resume checkpoints.
struct OutputDirs
{
    std::filesystem::path out_dir;
    std::filesystem::path models_dir;
    std::filesystem::path chk_dir;
};

// Resolves `out_dir` (falling back to plain "results" when the implicit source-tree
// location doesn't exist in this checkout — only the *implicit*, empty results_dir path
// may fall back; an explicit results_dir is always honored), then creates it and the
// models/checkpoints subdirectories.
OutputDirs resolve_output_dirs(const Meeting01Config& config)
{
    std::filesystem::path out_dir = config.dataset.results_dir.empty()
                                        ? source_results_dir()
                                        : std::filesystem::path(config.dataset.results_dir);

    // Only the *implicit* (empty results_dir) path may fall back: source_results_dir()
    // points into the source tree and may not exist in this checkout. An explicit
    // results_dir (e.g. "results/meeting01") is always honored and created below —
    // otherwise a fresh checkout without that subdir would silently divert output to
    // plain "results/".
    if (config.dataset.results_dir.empty() && !std::filesystem::exists(out_dir))
    {
        out_dir = std::filesystem::path("results");
    }

    std::filesystem::create_directories(out_dir);
    const std::filesystem::path models_dir =
        out_dir / "models" / sanitize_name(config.experiment.run_tag);
    if (config.dataset.save_models)
    {
        std::filesystem::create_directories(models_dir);
    }

    const auto chk_dir = out_dir / "checkpoints";
    std::filesystem::create_directories(chk_dir);

    return {out_dir, models_dir, chk_dir};
}

// Maps an evaluation.baselines family token to (model config, arch label used in the
// result row / checkpoint key).
struct BaselineFamily
{
    std::string token; // "lstm-ae" | "gru-ae" | "transformer-ae"
    std::string arch;  // "lstm"    | "gru"    | "transformer"
};

// Builds one ResultRow for a baseline family on a given split partition.
auto make_baseline_row(const Meeting01Config& config,
    const std::string& backend_name,
    const std::string& dataset_name,
    const BaselineFamily& fam,
    const std::string& encoding,
    int run_id,
    std::uint32_t run_seed,
    std::size_t cfg_hash,
    const std::string& split_name,
    const RunMetrics& metrics) -> ResultRow
{
    ResultRow row{backend_name,
        config.experiment.run_tag,
        dataset_name,
        fam.token,
        encoding,
        fam.arch,
        1,
        0.0f,
        0.0f,
        run_id + 1,
        run_seed,
        cfg_hash,
        metrics};
    row.split = split_name;
    row.cv_fold = config.dataset.cv_fold;
    return row;
}

// Summary of one family's GA winner, returned by run_snn_ga_search /
// run_lstm_ga_search / run_gru_ga_search / run_transformer_ga_search so the per-run_id
// loop (run_comparative_experiment) can compare the 4 winners at the end and record
// which family is best overall — same ordering pick_winner already uses within a single
// family's Pareto front: val_mse first, inference_cost as the tie-break.
struct FamilyWinnerSummary
{
    std::string family_token; // "snn-ae" | "lstm-ae" | "gru-ae" | "transformer-ae"
    float val_mse;
    std::size_t inference_cost;
};

/** Writes the encoder/decoder parameter dumps for one SNN model, when model saving is
 *  configured. `role_tag` ("combo" for a GA-evaluated candidate genome, "final" for the
 *  retrained winner) plus the fold index keep the ~18 nested-LOSO processes from
 *  colliding on the same filename. */
void save_snn_combo_models(const Meeting01Config& config,
    const std::string& dataset_name,
    const std::string& encoding,
    const std::string& architecture,
    float voltage_threshold,
    float alpha,
    int run_id,
    const std::filesystem::path& models_dir,
    ProtocolSpikingAutoencoder& snn_model,
    const std::string& role_tag = "combo")
{
    if (!config.dataset.save_models)
    {
        return;
    }

    const std::string base_name = sanitize_name(
        config.experiment.run_tag + "_snn_" + role_tag + "_" + dataset_name + "_" + encoding + "_" +
        architecture + "_vth" + std::to_string(voltage_threshold) + "_a" + std::to_string(alpha) +
        "_fold" + std::to_string(config.dataset.cv_fold) + "_run" + std::to_string(run_id + 1));
    const std::filesystem::path encoder_txt = models_dir / (base_name + "_encoder_params.txt");
    const std::filesystem::path decoder_txt = models_dir / (base_name + "_decoder_params.txt");
    const bool enc_ok =
        save_parameter_list_text(encoder_txt, snn_model.encoder_.params(), "encoder");
    const bool dec_ok =
        save_parameter_list_text(decoder_txt, snn_model.decoder_.params(), "decoder");
    if (!enc_ok || !dec_ok)
    {
        NN_LOG_WARN("[comparative] failed to save SNN model artifacts for " + base_name);
    }

    // Binary .npz alongside the text dumps: the SNN AE encoder_/decoder_ are plain
    // nn::Sequential of Linear/Lif, which NetworkSerializer round-trips exactly. This is
    // the artifact nn_microscope.meeting01.snn_ae_forward reloads to reproduce a window's
    // latent + reconstruction for the inspection GUI without retraining (FIXME §19, §20).
    const bool enc_npz = NetworkSerializer::saveNetwork(
        snn_model.encoder_, (models_dir / (base_name + "_encoder.npz")).string());
    const bool dec_npz = NetworkSerializer::saveNetwork(
        snn_model.decoder_, (models_dir / (base_name + "_decoder.npz")).string());
    if (!enc_npz || !dec_npz)
    {
        NN_LOG_WARN("[comparative] failed to save SNN model .npz for " + base_name);
    }
}

// Recording-disjoint early-stopping monitor carved from `train` for the nested-LOSO
// final fit. The selected config is retrained on (train \ monitor) ∪ val and
// early-stopped on `monitor`, so the test speaker still never influences any weight or
// stopping decision. Whole recordings move together (adjacent same-recording windows
// are correlated); selection is seeded for reproducibility.
struct MonitorCarve
{
    std::vector<Tensor> fit_samples;     // (train \ monitor)
    std::vector<Tensor> monitor_samples; // carved early-stopping set
};

auto carve_recording_disjoint_monitor(const std::vector<Tensor>& train_samples,
    const std::vector<WindowMetadata>& train_meta,
    std::size_t target_monitor_count,
    std::uint32_t seed) -> MonitorCarve
{
    std::map<int, std::vector<std::size_t>> by_recording;
    for (std::size_t i = 0; i < train_meta.size(); ++i)
        by_recording[train_meta[i].recording_id].push_back(i);

    std::vector<int> recordings;
    recordings.reserve(by_recording.size());
    for (const auto& [rid, idx] : by_recording) recordings.push_back(rid);
    std::mt19937 rng(seed != 0u ? seed : 42u);
    std::shuffle(recordings.begin(), recordings.end(), rng);

    std::vector<char> is_monitor(train_samples.size(), 0);
    std::size_t taken = 0;
    for (int rid : recordings)
    {
        if (taken >= target_monitor_count) break;
        for (std::size_t i : by_recording[rid])
        {
            is_monitor[i] = 1;
            ++taken;
        }
    }

    MonitorCarve out;
    for (std::size_t i = 0; i < train_samples.size(); ++i)
    {
        if (is_monitor[i] != 0)
            out.monitor_samples.push_back(train_samples[i]);
        else
            out.fit_samples.push_back(train_samples[i]);
    }
    return out;
}

// Shared nested-LOSO final-fit + test-evaluate + manifest routine for one baseline
// family's GA winner (LSTM-AE / GRU-AE / Transformer-AE). All three are frame-consuming
// AEs sharing train_ae/evaluate_ae/per_window_errors_ae (Meeting01AeCommon.hpp), so only
// the model itself and its genome-derived cost differ between families — the caller
// builds `model` from the winning genome (to_lstm_cfg/to_gru_cfg/to_transformer_cfg) and
// passes its true MAC estimate; this function is everything IDENTICAL across the three
// (retrain on train ∪ val, early-stop on a recording-disjoint monitor carve, test-
// evaluate, checkpoint, save, manifest) — the baseline-arm analogue of
// finalize_snn_selection, minus the SNN's architecture-family/encoder-widths axes since
// a baseline's shape is fully described by `model` and `selected_hyperparams`.
template <typename Model>
void finalize_baseline_selection(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    const BaselineFamily& fam,
    const std::string& encoding,
    int run_id,
    std::uint32_t run_seed,
    const std::string& backend_name,
    std::size_t cfg_hash,
    const std::filesystem::path& chk_dir,
    const std::filesystem::path& models_dir,
    Model& model,
    std::size_t macs,
    const nlohmann::json& selected_hyperparams,
    const nlohmann::json& candidates_json,
    float val_mse,
    std::vector<ResultRow>& all_rows,
    std::vector<PerWindowError>& pw_rows)
{
    if (split.test_samples.empty()) return;

    const std::string final_config_id =
        make_config_id(fam.token, encoding, fam.arch + "_final", 0.0f, 0.0f, run_seed, run_id + 1);
    ExperimentEvents::instance().emit("config_selected",
        {{"config_id", final_config_id},
            {"encoding", encoding},
            {"run_id", run_id + 1},
            {"seed", run_seed},
            {"selection_metric", "val_mse"},
            {"selected", selected_hyperparams}});

    CheckpointKey test_key{config.experiment.run_tag,
        backend_name,
        dataset_name,
        fam.token,
        encoding,
        fam.arch,
        0.0f,
        0.0f,
        run_id + 1,
        "test",
        config.dataset.cv_fold};
    const auto test_chk = checkpoint_path(chk_dir, test_key);
    if (checkpoint_is_valid(test_chk, cfg_hash))
    {
        all_rows.push_back(checkpoint_load(test_chk));
        emit_config_end(all_rows.back(), final_config_id, fam.arch + "_final");
        return;
    }

    {
        EventContext evctx;
        evctx.config_id = final_config_id;
        evctx.model = fam.token;
        evctx.encoding = encoding;
        evctx.role = fam.arch + "_final";
        evctx.hyperparams = selected_hyperparams;
        evctx.run_id = run_id + 1;
        evctx.seed = run_seed;
        evctx.max_epochs = config.training.epochs;
        evctx.lr = config.training.learning_rate;
        evctx.lr_biophysical = config.training.learning_rate_biophysical;
        evctx.early_stop_patience = config.training.early_stop_patience;
        ExperimentEvents::instance().set_pending_context(evctx);
    }

    // Retrain the selected genome on (train \ monitor) ∪ val, early-stopping on the
    // carved recording-disjoint monitor — identical discipline to finalize_snn_selection:
    // the test speaker never enters any fit.
    const MonitorCarve carve = carve_recording_disjoint_monitor(
        split.train_samples, split.train_meta, split.val_samples.size(), run_seed);

    std::vector<Tensor> fit_samples = carve.fit_samples;
    fit_samples.insert(fit_samples.end(), split.val_samples.begin(), split.val_samples.end());

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const TrainResult final_train = train_ae(model,
        config,
        fit_samples,
        carve.monitor_samples,
        encoding,
        run_seed,
        static_cast<std::size_t>(run_id),
        static_cast<std::size_t>(config.experiment.repeats),
        fam.token + " final: encoding=" + encoding,
        fam.token + " (final)",
        macs,
        train_ms,
        infer_ms);

    const RunMetrics test_metrics = evaluate_ae(model,
        split.test_samples,
        std::vector<int>(split.test_samples.size(), 0),
        config.training.max_reconstruct_mean_deviation,
        macs,
        parameter_count(model.params()),
        encoding,
        run_seed,
        infer_ms,
        config.model.lstm_frame_size,
        config.model.time_steps);

    all_rows.push_back(make_baseline_row(config,
        backend_name,
        dataset_name,
        fam,
        encoding,
        run_id,
        run_seed,
        cfg_hash,
        "test",
        test_metrics));
    checkpoint_save(test_chk, all_rows.back(), final_train.history, cfg_hash);
    emit_config_end(all_rows.back(), final_config_id, fam.arch + "_final");

    if (config.dataset.save_models)
    {
        const std::string base_name =
            sanitize_name(config.experiment.run_tag + "_" + fam.arch + "_" + dataset_name + "_" +
                          encoding + "_fold" + std::to_string(config.dataset.cv_fold) + "_run" +
                          std::to_string(run_id + 1) + "_final");
        if (!save_state_dict_text(models_dir / (base_name + "_state_dict.txt"), model.state_dict()))
        {
            NN_LOG_WARN(
                "[comparative] failed to save " + fam.token + " state_dict for " + base_name);
        }
    }

    {
        PerWindowError proto;
        proto.model = fam.token;
        proto.architecture = fam.arch;
        proto.run_id = run_id + 1;
        proto.seed = run_seed;
        proto.cv_fold = config.dataset.cv_fold;
        proto.split = "test";
        auto pw = per_window_errors_ae(model,
            split.test_samples,
            split.test_meta,
            encoding,
            run_seed,
            config.model.lstm_frame_size,
            config.model.time_steps,
            proto);
        pw_rows.insert(pw_rows.end(), pw.begin(), pw.end());
    }

    // Model-selection provenance manifest — same rationale as finalize_snn_selection's:
    // proves the choice used inner-val only, and records every genome the GA evaluated.
    if (!config.dataset.results_dir.empty())
    {
        nlohmann::json man;
        man["dataset"] = dataset_name;
        man["cv_fold"] = config.dataset.cv_fold;
        man["encoding"] = encoding;
        man["run_id"] = run_id + 1;
        man["seed"] = run_seed;
        man["selection_split"] = "val (speaker " + split.val_speaker + ")";
        man["selection_metric"] = "val_mse";
        man["test_speaker"] = split.test_speaker;
        man["selected"] = selected_hyperparams;
        man["selected"]["encoding"] = encoding;
        man["selected"]["val_mse"] = val_mse;
        man["candidates"] = candidates_json;
        const std::filesystem::path man_path =
            std::filesystem::path(config.dataset.results_dir) /
            (fold_output_tag(config, dataset_name) + "_" + fam.token + "_run" +
                std::to_string(run_id + 1) + "_model_selection_manifest.json");
        std::ofstream mf(man_path);
        if (mf.is_open()) mf << man.dump(2);
    }
}

// The GA-selected SNN-AE candidate, ready for the nested-LOSO final fit.
// `encoder_widths` is the winning genome's free-form layer widths (see
// Meeting01GaGenome::to_ae_config) — always non-empty, since the GA is the only SNN
// architecture search mechanism (no grid/sweep path exists).
struct SnnSelection
{
    std::string architecture;
    // Which encoding this candidate was trained under. `encoding` is a GA gene, so
    // candidates in one search differ in it; omitting it from the provenance manifest
    // made every candidate look like it shared the winner's encoding, and made the
    // GA's encoding-selection frequency — the SNN-side replacement for the old
    // per-encoding comparison — impossible to audit.
    std::string encoding;
    float v_th;
    float alpha;
    float val_mse;
    std::vector<int> encoder_widths;
};

// Nested-LOSO final fit for the GA-selected SNN-AE: retrain `best` on
// (train \ monitor) ∪ val, early-stop on the carved recording-disjoint monitor,
// evaluate once on the held-out test speaker, checkpoint/save/manifest. `candidates` is
// only used for the model-selection provenance manifest (every genome the GA
// evaluated, not just the winner).
void finalize_snn_selection(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    const std::string& encoding,
    int run_id,
    std::uint32_t run_seed,
    const std::string& backend_name,
    std::size_t cfg_hash,
    const std::filesystem::path& chk_dir,
    const std::filesystem::path& models_dir,
    const SnnSelection& best,
    const std::vector<SnnSelection>& candidates,
    std::vector<ResultRow>& all_rows,
    std::vector<PerWindowError>& pw_rows)
{
    if (split.test_samples.empty()) return;

    const std::string final_config_id = make_config_id(
        "snn-ae", encoding, "snn_final", best.v_th, best.alpha, run_seed, run_id + 1);
    ExperimentEvents::instance().emit("config_selected",
        {{"config_id", final_config_id},
            {"encoding", encoding},
            {"run_id", run_id + 1},
            {"seed", run_seed},
            {"selection_metric", "val_mse"},
            {"selected",
                {{"architecture", best.architecture},
                    {"v_th", best.v_th},
                    {"alpha", best.alpha},
                    {"val_score", jnum(best.val_mse)}}}});

    // Test row for the winner may already be checkpointed.
    CheckpointKey test_key{config.experiment.run_tag,
        backend_name,
        dataset_name,
        "snn-ae",
        encoding,
        best.architecture,
        best.v_th,
        best.alpha,
        run_id + 1,
        "test",
        config.dataset.cv_fold};
    const auto test_chk = checkpoint_path(chk_dir, test_key);
    if (checkpoint_is_valid(test_chk, cfg_hash))
    {
        all_rows.push_back(checkpoint_load(test_chk));
        emit_config_end(all_rows.back(), final_config_id, "snn_final");
        return;
    }

    {
        EventContext evctx;
        evctx.config_id = final_config_id;
        evctx.model = "snn-ae";
        evctx.encoding = encoding;
        evctx.role = "snn_final";
        evctx.hyperparams = {
            {"v_th", best.v_th}, {"alpha", best.alpha}, {"architecture", best.architecture}};
        evctx.run_id = run_id + 1;
        evctx.seed = run_seed;
        evctx.max_epochs = config.training.epochs;
        evctx.lr = config.training.learning_rate;
        evctx.lr_biophysical = config.training.learning_rate_biophysical;
        evctx.early_stop_patience = config.training.early_stop_patience;
        ExperimentEvents::instance().set_pending_context(evctx);
    }

    // Retrain the selected config on (train \ monitor) ∪ val, early-stopping on the
    // carved recording-disjoint monitor. The test speaker never enters any fit.
    const MonitorCarve carve = carve_recording_disjoint_monitor(
        split.train_samples, split.train_meta, split.val_samples.size(), run_seed);

    std::vector<Tensor> fit_samples = carve.fit_samples;
    fit_samples.insert(fit_samples.end(), split.val_samples.begin(), split.val_samples.end());

    AutoencoderConfig snn_config = make_snn_cfg(config, best.alpha, best.v_th, best.encoder_widths);
    snn_config.initializer_seed = run_seed;
    snn_config.initializer_sampler_type =
        "comparative-final|" + dataset_name + "|" + encoding + "|" + best.architecture + "|" +
        std::to_string(best.v_th) + "|" + std::to_string(best.alpha);
    ProtocolSpikingAutoencoder snn_model(snn_config);

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const TrainResult final_train = train_with_early_stopping_snn(snn_model,
        config,
        fit_samples,
        carve.monitor_samples,
        std::vector<int>(carve.monitor_samples.size(), 0),
        encoding,
        best.architecture,
        best.alpha,
        best.v_th,
        run_seed,
        static_cast<std::size_t>(run_id),
        static_cast<std::size_t>(config.experiment.repeats),
        train_ms,
        infer_ms);

    const std::vector<int> sizes = best.encoder_widths.empty()
                                       ? extract_layer_sizes(config.model.encoder_layer_spec)
                                       : best.encoder_widths;

    RunMetrics test_metrics = evaluate_snn(snn_model,
        split.test_samples,
        std::vector<int>(split.test_samples.size(), 0),
        config.training.max_reconstruct_mean_deviation,
        estimate_snn_macs(static_cast<std::size_t>(config.dataset.window_size),
            sizes.empty() ? 0 : sizes.front(),
            static_cast<int>(sizes.size())),
        parameter_count(snn_model.params()),
        encoding,
        best.architecture,
        best.alpha,
        best.v_th,
        run_seed,
        infer_ms,
        config.model.time_steps);
    test_metrics.train_ms = train_ms;

    ResultRow test_row{backend_name,
        config.experiment.run_tag,
        dataset_name,
        "snn-ae",
        encoding,
        best.architecture,
        static_cast<int>(sizes.size()),
        best.v_th,
        best.alpha,
        run_id + 1,
        run_seed,
        cfg_hash,
        test_metrics};
    test_row.split = "test";
    test_row.cv_fold = config.dataset.cv_fold;
    all_rows.push_back(test_row);
    checkpoint_save(test_chk, all_rows.back(), final_train.history, cfg_hash);
    emit_config_end(all_rows.back(), final_config_id, "snn_final");

    // Persist the retrained winner: this is the model whose held-out test metrics feed
    // the paper, and the one the inspection GUI most needs.
    save_snn_combo_models(config,
        dataset_name,
        encoding,
        best.architecture,
        best.v_th,
        best.alpha,
        run_id,
        models_dir,
        snn_model,
        "final");

    {
        PerWindowError proto;
        proto.model = "snn-ae";
        proto.architecture = best.architecture;
        proto.v_th = best.v_th;
        proto.alpha = best.alpha;
        proto.run_id = run_id + 1;
        proto.seed = run_seed;
        proto.cv_fold = config.dataset.cv_fold;
        proto.split = "test";
        auto pw = per_window_errors_snn(snn_model,
            split.test_samples,
            split.test_meta,
            encoding,
            best.architecture,
            best.alpha,
            best.v_th,
            run_seed,
            config.model.time_steps,
            proto);
        pw_rows.insert(pw_rows.end(), pw.begin(), pw.end());
    }

    // Model-selection provenance manifest: proves the choice used inner-val only.
    if (!config.dataset.results_dir.empty())
    {
        nlohmann::json man;
        man["dataset"] = dataset_name;
        man["cv_fold"] = config.dataset.cv_fold;
        man["encoding"] = encoding;
        man["run_id"] = run_id + 1;
        man["seed"] = run_seed;
        man["selection_split"] = "val (speaker " + split.val_speaker + ")";
        man["selection_metric"] = "val_mse";
        man["time_steps"] = config.model.time_steps;
        man["test_speaker"] = split.test_speaker;
        // `encoder_widths` and `encoding` are what actually make this manifest a
        // reproducibility record rather than a log line: the GA searches a free-form
        // architecture AND its encoding, so without both the published network cannot
        // be rebuilt from the manifest. The top-level "encoding" key above is the
        // per-run loop label, NOT the winning genome's gene — they can differ.
        man["selected"] = {{"architecture", best.architecture},
            {"encoding", best.encoding},
            {"encoder_widths", best.encoder_widths},
            {"v_th", best.v_th},
            {"alpha", best.alpha},
            {"val_mse", best.val_mse}};
        for (const auto& c : candidates)
            man["candidates"].push_back({{"architecture", c.architecture},
                {"encoding", c.encoding},
                {"encoder_widths", c.encoder_widths},
                {"v_th", c.v_th},
                {"alpha", c.alpha},
                {"val_mse", c.val_mse}});
        const std::filesystem::path man_path =
            std::filesystem::path(config.dataset.results_dir) /
            (fold_output_tag(config, dataset_name) + "_" + encoding + "_run" +
                std::to_string(run_id + 1) + "_model_selection_manifest.json");
        std::ofstream mf(man_path);
        if (mf.is_open()) mf << man.dump(2);
    }
}

// Runs NSGA-II over the SNN-AE's own architecture (Meeting01GaSearch.hpp) for one
// (dataset, run_id) — the ONLY SNN architecture search mechanism (no grid/sweep exists).
// Evolves encoder_widths/encoding/architecture/v_th/alpha jointly. Called once per
// run_id, OUTSIDE the per-encoding baseline loop: the SNN's own encoding is a gene
// rather than an externally fixed sweep dimension, so there is no per-encoding SNN cell
// to loop over (baselines still run per encoding, unaffected — see Meeting01.md).
FamilyWinnerSummary run_snn_ga_search(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    int run_id,
    std::uint32_t run_seed,
    const std::string& backend_name,
    std::size_t cfg_hash,
    const std::filesystem::path& chk_dir,
    const std::filesystem::path& models_dir,
    std::uint32_t run_bar,
    int& completed_runs,
    std::vector<ResultRow>& all_rows,
    std::vector<PerWindowError>& pw_rows)
{
    const auto& ga_cfg = config.evaluation.ga.snn;

    meeting01::ga::GenomeBounds bounds;
    bounds.min_layers = ga_cfg.min_layers;
    bounds.max_layers = ga_cfg.max_layers;
    bounds.min_width = ga_cfg.min_width;
    bounds.max_width = ga_cfg.max_width;
    bounds.voltage_threshold_min = ga_cfg.voltage_threshold_min;
    bounds.voltage_threshold_max = ga_cfg.voltage_threshold_max;
    bounds.alpha_min = ga_cfg.alpha_min;
    bounds.alpha_max = ga_cfg.alpha_max;
    bounds.encoding_choices = config.evaluation.encodings;
    bounds.architecture_choices = config.evaluation.snn_architectures;
    // Bottleneck fixed at the profile's latent_dim (never a gene — same rule as the 3
    // baseline families, see GenomeBounds::latent_dim's comment), NOT bounds.max_width:
    // build_snn_decoder's first layer always expects exactly cfg.model.latent_dim
    // input features regardless of what the genome draws, so leaving this unset (the
    // struct default of 32) would silently diverge from a profile using a different
    // latent_dim.
    bounds.latent_dim = config.model.latent_dim;

    meeting01::ga::GaSearchConfig search_cfg;
    search_cfg.population_size = ga_cfg.population_size;
    search_cfg.generations = ga_cfg.generations;
    search_cfg.crossover_prob = ga_cfg.crossover_prob;
    search_cfg.mutation_prob = ga_cfg.mutation_prob;
    search_cfg.tournament_k = ga_cfg.tournament_k;
    search_cfg.winner_seeds = ga_cfg.winner_seeds;
    search_cfg.seed = ga_cfg.seed;
    search_cfg.bounds = bounds;
    search_cfg.results_dir = config.dataset.results_dir;
    search_cfg.run_tag = config.experiment.run_tag + "_" + dataset_name + "_fold" +
                         std::to_string(config.dataset.cv_fold) + "_run" +
                         std::to_string(run_id + 1);
    search_cfg.checkpoint_every_generations = ga_cfg.checkpoint_every_generations;

    // `Ind` is given explicitly: it cannot be deduced from the callback argument alone
    // (a raw lambda converting to std::function<void(const Ind&)> is a non-deduced
    // context in template argument deduction), so every run_ga_search call site names
    // its individual type — the same reason each family's orchestration function below
    // does too.
    const auto ga_result =
        meeting01::ga::run_ga_search<meeting01::ga::Meeting01GaIndividual>(config,
            split,
            search_cfg,
            run_seed,
            [&](const meeting01::ga::Meeting01GaIndividual&)
            {
                nn::progress::ProgressManager::instance().update_bar(
                    run_bar, static_cast<float>(++completed_runs));
            });

    const auto& winner = meeting01::ga::pick_winner(ga_result);

    const SnnSelection best{winner.genome.architecture,
        winner.genome.encoding,
        winner.genome.voltage_threshold,
        winner.genome.alpha,
        winner.val_mse,
        winner.genome.encoder_widths};

    std::vector<SnnSelection> candidates;
    candidates.reserve(ga_result.history.size());
    for (const auto& ind : ga_result.history)
        candidates.push_back({ind.genome.architecture,
            ind.genome.encoding,
            ind.genome.voltage_threshold,
            ind.genome.alpha,
            ind.val_mse,
            ind.genome.encoder_widths});

    finalize_snn_selection(config,
        split,
        dataset_name,
        winner.genome.encoding,
        run_id,
        run_seed,
        backend_name,
        cfg_hash,
        chk_dir,
        models_dir,
        best,
        candidates,
        all_rows,
        pw_rows);

    return FamilyWinnerSummary{"snn-ae", winner.val_mse, winner.inference_cost};
}

// Runs NSGA-II over the LSTM-AE's own architecture (hidden_size/num_layers/encoding)
// for one (dataset, run_id) — the baseline-arm analogue of run_snn_ga_search, now that
// every family (SNN and all three baselines) searches its own architecture instead of
// training at a profile-fixed shape (2026-09-22 scope change: the experiment's goal is
// "the best autoencoder, period", not "SNN vs three fixed baselines").
FamilyWinnerSummary run_lstm_ga_search(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    int run_id,
    std::uint32_t run_seed,
    const std::string& backend_name,
    std::size_t cfg_hash,
    const std::filesystem::path& chk_dir,
    const std::filesystem::path& models_dir,
    std::uint32_t run_bar,
    int& completed_runs,
    std::vector<ResultRow>& all_rows,
    std::vector<PerWindowError>& pw_rows)
{
    const auto& ga_cfg = config.evaluation.ga.lstm;

    meeting01::ga::RecurrentGenomeBounds bounds;
    bounds.min_hidden = ga_cfg.min_hidden;
    bounds.max_hidden = ga_cfg.max_hidden;
    bounds.min_layers = ga_cfg.min_layers;
    bounds.max_layers = ga_cfg.max_layers;
    bounds.encoding_choices = config.evaluation.encodings;

    meeting01::ga::GaSearchConfigT<meeting01::ga::RecurrentGenomeBounds> search_cfg;
    search_cfg.population_size = ga_cfg.population_size;
    search_cfg.generations = ga_cfg.generations;
    search_cfg.crossover_prob = ga_cfg.crossover_prob;
    search_cfg.mutation_prob = ga_cfg.mutation_prob;
    search_cfg.tournament_k = ga_cfg.tournament_k;
    search_cfg.winner_seeds = ga_cfg.winner_seeds;
    search_cfg.seed = ga_cfg.seed;
    search_cfg.bounds = bounds;
    search_cfg.results_dir = config.dataset.results_dir;
    search_cfg.run_tag = config.experiment.run_tag + "_" + dataset_name + "_fold" +
                         std::to_string(config.dataset.cv_fold) + "_run" +
                         std::to_string(run_id + 1) + "_lstm";
    search_cfg.checkpoint_every_generations = ga_cfg.checkpoint_every_generations;

    const auto ga_result = meeting01::ga::run_ga_search<meeting01::ga::LstmGaIndividual>(config,
        split,
        search_cfg,
        run_seed,
        [&](const meeting01::ga::LstmGaIndividual&)
        {
            nn::progress::ProgressManager::instance().update_bar(
                run_bar, static_cast<float>(++completed_runs));
        });

    const auto& winner = meeting01::ga::pick_winner(ga_result);

    nlohmann::json candidates_json;
    for (const auto& ind : ga_result.history)
        candidates_json.push_back({{"hidden_size", ind.genome.hidden_size},
            {"num_layers", ind.genome.num_layers},
            {"encoding", ind.genome.encoding},
            {"val_mse", ind.val_mse}});

    const nlohmann::json selected_hyperparams{
        {"hidden_size", winner.genome.hidden_size}, {"num_layers", winner.genome.num_layers}};

    const auto lstm_cfg = meeting01::ga::to_lstm_cfg(winner.genome, config);
    nn::models::lstm::LSTMAutoencoder model(lstm_cfg);
    const std::size_t macs = estimate_lstm_macs(lstm_cfg);

    finalize_baseline_selection(config,
        split,
        dataset_name,
        BaselineFamily{"lstm-ae", "lstm"},
        winner.genome.encoding,
        run_id,
        run_seed,
        backend_name,
        cfg_hash,
        chk_dir,
        models_dir,
        model,
        macs,
        selected_hyperparams,
        candidates_json,
        winner.val_mse,
        all_rows,
        pw_rows);

    return FamilyWinnerSummary{"lstm-ae", winner.val_mse, winner.inference_cost};
}

// GRU-AE analogue of run_lstm_ga_search — same RecurrentGenome axes, distinct type
// (GruGaIndividual) so evaluate_individual overload resolution trains the right cell
// (Meeting01RecurrentGaFitness.hpp).
FamilyWinnerSummary run_gru_ga_search(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    int run_id,
    std::uint32_t run_seed,
    const std::string& backend_name,
    std::size_t cfg_hash,
    const std::filesystem::path& chk_dir,
    const std::filesystem::path& models_dir,
    std::uint32_t run_bar,
    int& completed_runs,
    std::vector<ResultRow>& all_rows,
    std::vector<PerWindowError>& pw_rows)
{
    const auto& ga_cfg = config.evaluation.ga.gru;

    meeting01::ga::RecurrentGenomeBounds bounds;
    bounds.min_hidden = ga_cfg.min_hidden;
    bounds.max_hidden = ga_cfg.max_hidden;
    bounds.min_layers = ga_cfg.min_layers;
    bounds.max_layers = ga_cfg.max_layers;
    bounds.encoding_choices = config.evaluation.encodings;

    meeting01::ga::GaSearchConfigT<meeting01::ga::RecurrentGenomeBounds> search_cfg;
    search_cfg.population_size = ga_cfg.population_size;
    search_cfg.generations = ga_cfg.generations;
    search_cfg.crossover_prob = ga_cfg.crossover_prob;
    search_cfg.mutation_prob = ga_cfg.mutation_prob;
    search_cfg.tournament_k = ga_cfg.tournament_k;
    search_cfg.winner_seeds = ga_cfg.winner_seeds;
    search_cfg.seed = ga_cfg.seed;
    search_cfg.bounds = bounds;
    search_cfg.results_dir = config.dataset.results_dir;
    search_cfg.run_tag = config.experiment.run_tag + "_" + dataset_name + "_fold" +
                         std::to_string(config.dataset.cv_fold) + "_run" +
                         std::to_string(run_id + 1) + "_gru";
    search_cfg.checkpoint_every_generations = ga_cfg.checkpoint_every_generations;

    const auto ga_result = meeting01::ga::run_ga_search<meeting01::ga::GruGaIndividual>(config,
        split,
        search_cfg,
        run_seed,
        [&](const meeting01::ga::GruGaIndividual&)
        {
            nn::progress::ProgressManager::instance().update_bar(
                run_bar, static_cast<float>(++completed_runs));
        });

    const auto& winner = meeting01::ga::pick_winner(ga_result);

    nlohmann::json candidates_json;
    for (const auto& ind : ga_result.history)
        candidates_json.push_back({{"hidden_size", ind.genome.hidden_size},
            {"num_layers", ind.genome.num_layers},
            {"encoding", ind.genome.encoding},
            {"val_mse", ind.val_mse}});

    const nlohmann::json selected_hyperparams{
        {"hidden_size", winner.genome.hidden_size}, {"num_layers", winner.genome.num_layers}};

    const auto gru_cfg = meeting01::ga::to_gru_cfg(winner.genome, config);
    nn::models::gru::GRUAutoencoder model(gru_cfg);
    const std::size_t macs = estimate_gru_macs(gru_cfg);

    finalize_baseline_selection(config,
        split,
        dataset_name,
        BaselineFamily{"gru-ae", "gru"},
        winner.genome.encoding,
        run_id,
        run_seed,
        backend_name,
        cfg_hash,
        chk_dir,
        models_dir,
        model,
        macs,
        selected_hyperparams,
        candidates_json,
        winner.val_mse,
        all_rows,
        pw_rows);

    return FamilyWinnerSummary{"gru-ae", winner.val_mse, winner.inference_cost};
}

// Transformer-AE analogue of run_lstm_ga_search. `n_heads` is drawn from
// bounds.head_choices and reconciled against d_model by repair_transformer
// (Meeting01TransformerGaGenome.cpp) — the one family whose genome carries a hard
// structural constraint (d_model % n_heads == 0).
FamilyWinnerSummary run_transformer_ga_search(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    int run_id,
    std::uint32_t run_seed,
    const std::string& backend_name,
    std::size_t cfg_hash,
    const std::filesystem::path& chk_dir,
    const std::filesystem::path& models_dir,
    std::uint32_t run_bar,
    int& completed_runs,
    std::vector<ResultRow>& all_rows,
    std::vector<PerWindowError>& pw_rows)
{
    const auto& ga_cfg = config.evaluation.ga.transformer;

    meeting01::ga::TransformerGenomeBounds bounds;
    bounds.min_d_model = ga_cfg.min_d_model;
    bounds.max_d_model = ga_cfg.max_d_model;
    bounds.head_choices = ga_cfg.head_choices;
    bounds.min_layers = ga_cfg.min_layers;
    bounds.max_layers = ga_cfg.max_layers;
    bounds.min_d_ff = ga_cfg.min_d_ff;
    bounds.max_d_ff = ga_cfg.max_d_ff;
    bounds.encoding_choices = config.evaluation.encodings;

    meeting01::ga::GaSearchConfigT<meeting01::ga::TransformerGenomeBounds> search_cfg;
    search_cfg.population_size = ga_cfg.population_size;
    search_cfg.generations = ga_cfg.generations;
    search_cfg.crossover_prob = ga_cfg.crossover_prob;
    search_cfg.mutation_prob = ga_cfg.mutation_prob;
    search_cfg.tournament_k = ga_cfg.tournament_k;
    search_cfg.winner_seeds = ga_cfg.winner_seeds;
    search_cfg.seed = ga_cfg.seed;
    search_cfg.bounds = bounds;
    search_cfg.results_dir = config.dataset.results_dir;
    search_cfg.run_tag = config.experiment.run_tag + "_" + dataset_name + "_fold" +
                         std::to_string(config.dataset.cv_fold) + "_run" +
                         std::to_string(run_id + 1) + "_transformer";
    search_cfg.checkpoint_every_generations = ga_cfg.checkpoint_every_generations;

    const auto ga_result =
        meeting01::ga::run_ga_search<meeting01::ga::TransformerGaIndividual>(config,
            split,
            search_cfg,
            run_seed,
            [&](const meeting01::ga::TransformerGaIndividual&)
            {
                nn::progress::ProgressManager::instance().update_bar(
                    run_bar, static_cast<float>(++completed_runs));
            });

    const auto& winner = meeting01::ga::pick_winner(ga_result);

    nlohmann::json candidates_json;
    for (const auto& ind : ga_result.history)
        candidates_json.push_back({{"d_model", ind.genome.d_model},
            {"n_heads", ind.genome.n_heads},
            {"n_layers", ind.genome.n_layers},
            {"d_ff", ind.genome.d_ff},
            {"encoding", ind.genome.encoding},
            {"val_mse", ind.val_mse}});

    const nlohmann::json selected_hyperparams{{"d_model", winner.genome.d_model},
        {"n_heads", winner.genome.n_heads},
        {"n_layers", winner.genome.n_layers},
        {"d_ff", winner.genome.d_ff}};

    const auto tf_cfg = meeting01::ga::to_transformer_cfg(winner.genome, config);
    nn::models::transformer::TransformerAutoencoder model(tf_cfg);
    const std::size_t macs = estimate_transformer_macs(tf_cfg);

    finalize_baseline_selection(config,
        split,
        dataset_name,
        BaselineFamily{"transformer-ae", "transformer"},
        winner.genome.encoding,
        run_id,
        run_seed,
        backend_name,
        cfg_hash,
        chk_dir,
        models_dir,
        model,
        macs,
        selected_hyperparams,
        candidates_json,
        winner.val_mse,
        all_rows,
        pw_rows);

    return FamilyWinnerSummary{"transformer-ae", winner.val_mse, winner.inference_cost};
}

// Hard leakage gate + split manifest. Aborts the run (named exception, no fallback) if
// any speaker or any source recording appears in more than one of train/val/test.
void assert_split_disjoint_and_manifest(
    const Meeting01Config& config, const DatasetSplit& split, const std::string& dataset_name)
{
    // cv_fold is always >= 0 post-validate() (the pooled/shuffled legacy split was
    // removed 2026-09-23); this guard is now unreachable dead code, kept only so a
    // future caller that somehow bypasses validate() still fails safe by skipping
    // the assertion rather than crashing on an unset fold.
    if (config.dataset.cv_fold < 0) return;

    auto speakers = [](const std::vector<WindowMetadata>& m)
    {
        std::set<std::string> s;
        for (const auto& w : m) s.insert(w.speaker);
        return s;
    };
    auto recordings = [](const std::vector<WindowMetadata>& m)
    {
        std::set<int> s;
        for (const auto& w : m) s.insert(w.recording_id);
        return s;
    };

    const std::array<std::pair<std::string, const std::vector<WindowMetadata>*>, 3> parts{
        {{"train", &split.train_meta}, {"val", &split.val_meta}, {"test", &split.test_meta}}};

    for (std::size_t i = 0; i < parts.size(); ++i)
        for (std::size_t k = i + 1; k < parts.size(); ++k)
        {
            const auto si = speakers(*parts[i].second);
            const auto sk = speakers(*parts[k].second);
            for (const auto& sp : si)
                if (sk.count(sp) != 0)
                    throw std::runtime_error("LEAKAGE: speaker '" + sp + "' in both " +
                                             parts[i].first + " and " + parts[k].first + " (fold " +
                                             std::to_string(config.dataset.cv_fold) + ")");
            const auto ri = recordings(*parts[i].second);
            const auto rk = recordings(*parts[k].second);
            for (int rid : ri)
                if (rk.count(rid) != 0)
                    throw std::runtime_error("LEAKAGE: recording " + std::to_string(rid) +
                                             " in both " + parts[i].first + " and " +
                                             parts[k].first + " (fold " +
                                             std::to_string(config.dataset.cv_fold) + ")");
        }

    if (config.dataset.results_dir.empty()) return;
    nlohmann::json man;
    man["dataset"] = dataset_name;
    man["cv_fold"] = config.dataset.cv_fold;
    man["cv_num_folds"] = config.dataset.cv_num_folds;
    man["seed"] = config.experiment.seed;
    const std::set<std::string> train_speakers = speakers(split.train_meta);
    man["speakers"]["train"] =
        std::vector<std::string>(train_speakers.begin(), train_speakers.end());
    man["speakers"]["val"] = split.val_speaker;
    man["speakers"]["test"] = split.test_speaker;
    man["window_counts"] = {{"train", split.train_samples.size()},
        {"val", split.val_samples.size()},
        {"test", split.test_samples.size()}};
    auto rec_list = [&](const std::vector<WindowMetadata>& m)
    {
        std::map<int, int> counts;
        for (const auto& w : m) ++counts[w.recording_id];
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& [rid, c] : counts)
            arr.push_back({{"recording_id", rid}, {"window_count", c}});
        return arr;
    };
    man["recordings"]["train"] = rec_list(split.train_meta);
    man["recordings"]["val"] = rec_list(split.val_meta);
    man["recordings"]["test"] = rec_list(split.test_meta);

    const std::filesystem::path man_path =
        std::filesystem::path(config.dataset.results_dir) /
        (fold_output_tag(config, dataset_name) + "_split_manifest.json");
    std::ofstream mf(man_path);
    if (mf.is_open()) mf << man.dump(2);
}

// Dumps the framed-encoded train/test window matrices (row = window, col = flattened
// (T, frame_size)) for one fold+encoding so the Python PCA / mean-frame reference
// baselines can be fitted on train and scored on test in the exact representation the
// trained AEs reconstruct. Uses the seed-0 encoding realization (poisson is stochastic;
// the linear references are reported as one representative realization — direct and
// latency are deterministic). Once per fold+encoding, not per seed/model.
void dump_analytic_baseline_inputs(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    const std::string& encoding,
    const std::filesystem::path& out_dir)
{
    if (config.dataset.cv_fold < 0 || split.test_samples.empty()) return;
    if (config.dataset.results_dir.empty()) return;

    const int frame = config.model.lstm_frame_size;
    const int steps = config.model.time_steps;
    const std::uint32_t seed = config.experiment.seed;

    auto encode_matrix =
        [&](const std::vector<Tensor>& samples) -> std::pair<std::vector<float>, std::size_t>
    {
        std::vector<float> flat;
        std::size_t cols = 0;
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const Tensor framed = to_lstm_frames(
                encode_sample(samples[i], encoding, seed + static_cast<std::uint32_t>(i), steps),
                frame);
            const Tensor row = flatten_time_series(framed);
            cols = static_cast<std::size_t>(row.size());
            for (nn::Index k = 0; k < row.size(); ++k) flat.push_back(row.at(k));
        }
        return {flat, cols};
    };

    const std::string stem = fold_output_tag(config, dataset_name) + "_" + encoding;
    const std::filesystem::path dir(config.dataset.results_dir);

    NN_LOG_INFO("[loso] analytic-baseline dump: encoding " + encoding + " over " +
                std::to_string(split.train_samples.size()) + " train + " +
                std::to_string(split.test_samples.size()) + " test windows…");
    const auto [train_flat, train_cols] = encode_matrix(split.train_samples);
    const auto [test_flat, test_cols] = encode_matrix(split.test_samples);
    NN_LOG_INFO("[loso] analytic-baseline dump: " + encoding + " encoded");
    if (train_cols == 0 || test_cols == 0) return;

    cnpy::npy_save((dir / (stem + "_train_windows.npy")).string(),
        train_flat.data(),
        {split.train_samples.size(), train_cols},
        "w");
    cnpy::npy_save((dir / (stem + "_test_windows.npy")).string(),
        test_flat.data(),
        {split.test_samples.size(), test_cols},
        "w");

    std::ofstream meta(dir / (stem + "_test_windows_meta.csv"));
    if (meta.is_open())
    {
        meta << "speaker_id,recording_id,window_id,source_window_index\n";
        for (const auto& m : split.test_meta)
            meta << m.speaker_id << ',' << m.recording_id << ',' << m.window_id << ','
                 << m.source_window_index << '\n';
    }
}

// Writes every result artifact for the whole experiment: comparative CSV, publication
// table, JSON summary, and (when configured) the pgfplots/LaTeX exports.
void write_experiment_outputs(const Meeting01Config& config,
    std::size_t cfg_hash,
    const std::vector<ResultRow>& all_rows,
    const std::filesystem::path& out_dir)
{
    // Under nested LOSO each (dataset, fold) runs as its own process (one
    // `--dataset` + one `--cv-fold`); tag every output with dataset + fold so the
    // runs do not clobber each other. The Python aggregator globs
    // `<run_tag>_<dataset>_fold*_comparative_metrics.csv`.
    const std::string tag = fold_output_tag(config,
        config.evaluation.datasets.empty() ? std::string{} : config.evaluation.datasets.front());

    const std::filesystem::path csv_path = out_dir / (tag + "_comparative_metrics.csv");
    write_rows_csv(csv_path, all_rows);

    const std::filesystem::path table_path = out_dir / (tag + "_publication_table.csv");
    write_publication_table(table_path, all_rows);

    const std::filesystem::path summary_json = out_dir / (tag + "_summary.json");
    write_summary_json(summary_json, config, cfg_hash, all_rows);

    if (!config.dataset.latex_data_dir.empty())
    {
        const std::filesystem::path latex_dir =
            std::filesystem::path(config.dataset.latex_data_dir);
        write_latex_exports(latex_dir, tag, config, all_rows);
        write_pgfplots_summary_dat(latex_dir / (tag + "_summary.dat"), all_rows);
        write_pgfplots_sweep_dat(latex_dir / (tag + "_sweep.dat"), all_rows);
    }

    NN_LOG_INFO("[comparative] Results written to: " + csv_path.string() + ", " +
                table_path.string() + ", " + summary_json.string());
}

} // namespace

auto run_comparative_experiment(int argc, char* argv[]) -> int
{
    using namespace meeting01;

    try
    {
        const CliOptions cli = parse_cli(argc, argv);
        if (cli.help)
        {
            print_usage(argv[0]);
            return 0;
        }
        if (cli.no_tui) nn::progress::ProgressManager::instance().set_enabled(false);

        const Meeting01Config config = load_config(resolve_profile_path(cli), cli);
        config.validate();
        const std::size_t cfg_hash = config_hash(config);
        const std::string backend_name = active_backend_name();

        const OutputDirs dirs = resolve_output_dirs(config);
        const std::filesystem::path& out_dir = dirs.out_dir;
        const std::filesystem::path& models_dir = dirs.models_dir;
        const std::filesystem::path& chk_dir = dirs.chk_dir;

        std::vector<ResultRow> all_rows;

        // Total individual runs: every searched family (SNN and, since the 2026-09-22
        // scope change, every baseline too) is its own GA search of
        // population×(1+generations) evaluations per (dataset, repeat) — encoding is a
        // gene for all four families now, not an outer sweep dimension for any of them,
        // so it does not multiply any family's cost.
        const int n_datasets = static_cast<int>(config.evaluation.datasets.size());
        const auto& gc = config.evaluation.ga;
        auto ga_evals = [](int population, int generations)
        { return population * (1 + generations); };
        int evals_per_dataset_repeat = 0;
        if (!config.evaluation.snn_architectures.empty())
            evals_per_dataset_repeat += ga_evals(gc.snn.population_size, gc.snn.generations);
        for (const auto& family : config.evaluation.baselines)
        {
            if (family == "lstm-ae")
                evals_per_dataset_repeat += ga_evals(gc.lstm.population_size, gc.lstm.generations);
            else if (family == "gru-ae")
                evals_per_dataset_repeat += ga_evals(gc.gru.population_size, gc.gru.generations);
            else if (family == "transformer-ae")
                evals_per_dataset_repeat +=
                    ga_evals(gc.transformer.population_size, gc.transformer.generations);
        }
        const int total_outer_runs =
            n_datasets * config.experiment.repeats * evals_per_dataset_repeat;

        // Overall-progress banner: an optional pre-rendered line a wrapper script can inject
        // via MEETING01_OVERALL (this process, one dataset/fold slice of a larger grid, cannot
        // know the outer progress on its own). Logging it renders it as a persistent top line
        // above the per-run bars. No current script sets this — 01_meeting01_run_loso.sh reports
        // per-fold progress via its own "[loso] ... epoch N/M" stderr lines instead (see that
        // script's header) — so this is presently a harmless no-op, kept as the hook a future
        // wrapper can use without touching this file.
        if (const char* overall = std::getenv("MEETING01_OVERALL");
            overall != nullptr && overall[0] != '\0')
        {
            nn::progress::ProgressManager::instance().log(std::string(overall));
        }

        const uint32_t run_bar = nn::progress::ProgressManager::instance().create_bar(
            "Profile: " + config.experiment.run_tag, static_cast<float>(total_outer_runs));
        nn::progress::ProgressManager::instance().set_description(
            run_bar, "SNN vs LSTM comparative experiment");

        int completed_runs = 0;

        // Structured JSONL event sink for the live monitor. One file per (dataset, fold)
        // process, truncated on open. Disabled for the legacy pooled path or when no
        // results_dir is set — every emit() then no-ops.
        if (config.dataset.cv_fold >= 0 && !config.dataset.results_dir.empty() &&
            !config.evaluation.datasets.empty())
        {
            const std::string& ev_ds = config.evaluation.datasets.front();
            ExperimentEvents::instance().open((std::filesystem::path(config.dataset.results_dir) /
                                               (fold_output_tag(config, ev_ds) + "_events.jsonl"))
                    .string());
            ExperimentEvents::instance().set_common({{"v", 1},
                {"run_tag", config.experiment.run_tag},
                {"dataset", ev_ds},
                {"fold", config.dataset.cv_fold}});
            const char* git_env = std::getenv("MEETING01_GIT_COMMIT");
            ExperimentEvents::instance().emit("session_begin",
                {{"cv_num_folds", config.dataset.cv_num_folds},
                    {"all_datasets", config.evaluation.datasets},
                    {"seed", config.experiment.seed},
                    {"repeats", config.experiment.repeats},
                    {"total_outer_runs", total_outer_runs},
                    {"config_hash", cfg_hash},
                    {"git_commit",
                        (git_env != nullptr && git_env[0] != '\0') ? git_env : "unknown"},
                    {"backend", backend_name},
                    {"caps",
                        {{"train", config.dataset.loso_max_train_windows},
                            {"val", config.dataset.loso_max_val_windows},
                            {"test", config.dataset.loso_max_test_windows}}},
                    {"search_space",
                        {{"snn_architectures", config.evaluation.snn_architectures},
                            {"encodings", config.evaluation.encodings},
                            {"baselines", config.evaluation.baselines},
                            {"ga",
                                {{"snn",
                                     {{"population_size", config.evaluation.ga.snn.population_size},
                                         {"generations", config.evaluation.ga.snn.generations},
                                         {"voltage_threshold_range",
                                             {config.evaluation.ga.snn.voltage_threshold_min,
                                                 config.evaluation.ga.snn.voltage_threshold_max}},
                                         {"alpha_range",
                                             {config.evaluation.ga.snn.alpha_min,
                                                 config.evaluation.ga.snn.alpha_max}}}},
                                    {"lstm",
                                        {{"population_size",
                                             config.evaluation.ga.lstm.population_size},
                                            {"generations",
                                                config.evaluation.ga.lstm.generations}}},
                                    {"gru",
                                        {{"population_size",
                                             config.evaluation.ga.gru.population_size},
                                            {"generations", config.evaluation.ga.gru.generations}}},
                                    {"transformer",
                                        {{"population_size",
                                             config.evaluation.ga.transformer.population_size},
                                            {"generations",
                                                config.evaluation.ga.transformer
                                                    .generations}}}}}}}});
        }

        for (const auto& dataset_name : config.evaluation.datasets)
        {
            NN_LOG_INFO("[loso] " + dataset_name + " fold" +
                        std::to_string(config.dataset.cv_fold) + ": building split…");
            const auto t_bs0 = std::chrono::steady_clock::now();
            const DatasetSplit split = build_split(config, dataset_name, config.dataset.cv_fold);
            NN_LOG_INFO("[loso] split built: train=" + std::to_string(split.train_samples.size()) +
                        " val=" + std::to_string(split.val_samples.size()) +
                        " test=" + std::to_string(split.test_samples.size()) + "  (" +
                        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t_bs0)
                                .count()) +
                        " ms)");
            assert_split_disjoint_and_manifest(config, split, dataset_name);
            NN_LOG_INFO("[loso] leakage gate + split manifest OK");

            ExperimentEvents::instance().emit("fold_begin",
                {{"dataset", dataset_name},
                    {"window_counts",
                        {{"train", split.train_samples.size()},
                            {"val", split.val_samples.size()},
                            {"test", split.test_samples.size()}}},
                    {"speakers",
                        {{"train", split.train_speakers},
                            {"val", split.val_speaker},
                            {"test", split.test_speaker}}}});

            // Per-window reconstruction errors accumulate across every model / encoding /
            // seed of this fold, then flush once. Rows coming straight from a resume
            // checkpoint are not regenerated — clear results/meeting01/checkpoints/ before
            // a run that needs the per-window CSV (the article pipeline always does).
            std::vector<PerWindowError> pw_rows;

            // Diagnostic dumps only — not part of any fit. Kept per-encoding since
            // they're plain descriptive stats of the raw signal under each encoding, not
            // a model training pass.
            for (const auto& encoding : config.evaluation.encodings)
                dump_analytic_baseline_inputs(config, split, dataset_name, encoding, out_dir);

            // Architecture search: every family (SNN and, since the 2026-09-22 scope
            // change, every baseline too) runs its own NSGA-II search once per
            // (dataset, run_id), covering every encoding and architecture axis jointly
            // — encoding is a gene for all four families now, not an outer sweep
            // dimension for any of them, so there is no per-encoding loop left here.
            for (int run_id = 0; run_id < config.experiment.repeats; ++run_id)
            {
                const std::uint32_t run_seed =
                    config.experiment.seed_deterministic
                        ? config.experiment.seed
                        : config.experiment.seed + static_cast<std::uint32_t>(run_id);

                std::vector<FamilyWinnerSummary> family_winners;

                for (const auto& family : config.evaluation.baselines)
                {
                    if (family == "lstm-ae")
                    {
                        family_winners.push_back(run_lstm_ga_search(config,
                            split,
                            dataset_name,
                            run_id,
                            run_seed,
                            backend_name,
                            cfg_hash,
                            chk_dir,
                            models_dir,
                            run_bar,
                            completed_runs,
                            all_rows,
                            pw_rows));
                    }
                    else if (family == "gru-ae")
                    {
                        family_winners.push_back(run_gru_ga_search(config,
                            split,
                            dataset_name,
                            run_id,
                            run_seed,
                            backend_name,
                            cfg_hash,
                            chk_dir,
                            models_dir,
                            run_bar,
                            completed_runs,
                            all_rows,
                            pw_rows));
                    }
                    else if (family == "transformer-ae")
                    {
                        family_winners.push_back(run_transformer_ga_search(config,
                            split,
                            dataset_name,
                            run_id,
                            run_seed,
                            backend_name,
                            cfg_hash,
                            chk_dir,
                            models_dir,
                            run_bar,
                            completed_runs,
                            all_rows,
                            pw_rows));
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "run_comparative_experiment: unknown baseline '" + family +
                            "' — validate() should have rejected this. Valid: lstm-ae, "
                            "gru-ae, transformer-ae.");
                    }
                }

                if (!config.evaluation.snn_architectures.empty())
                {
                    family_winners.push_back(run_snn_ga_search(config,
                        split,
                        dataset_name,
                        run_id,
                        run_seed,
                        backend_name,
                        cfg_hash,
                        chk_dir,
                        models_dir,
                        run_bar,
                        completed_runs,
                        all_rows,
                        pw_rows));
                }

                // Compare the 4 winners: same ordering pick_winner already uses within a
                // single family's Pareto front (val_mse first, inference_cost the tie-
                // break). This is the "find the best autoencoder, period" step the
                // 2026-09-22 scope change is actually for — the per-family manifests
                // each finalize_*_selection call already wrote only answer "what won
                // within this family".
                if (!family_winners.empty() && !config.dataset.results_dir.empty())
                {
                    const auto overall = std::min_element(family_winners.begin(),
                        family_winners.end(),
                        [](const FamilyWinnerSummary& a, const FamilyWinnerSummary& b)
                        {
                            if (a.val_mse != b.val_mse) return a.val_mse < b.val_mse;
                            return a.inference_cost < b.inference_cost;
                        });

                    nlohmann::json man;
                    man["dataset"] = dataset_name;
                    man["cv_fold"] = config.dataset.cv_fold;
                    man["run_id"] = run_id + 1;
                    man["seed"] = run_seed;
                    man["selection_metric"] = "val_mse";
                    man["overall_winner"] = overall->family_token;
                    for (const auto& fw : family_winners)
                        man["families"].push_back({{"family", fw.family_token},
                            {"val_mse", fw.val_mse},
                            {"inference_cost", fw.inference_cost}});
                    const std::filesystem::path man_path =
                        std::filesystem::path(config.dataset.results_dir) /
                        (fold_output_tag(config, dataset_name) + "_run" +
                            std::to_string(run_id + 1) + "_overall_winner_manifest.json");
                    std::ofstream mf(man_path);
                    if (mf.is_open()) mf << man.dump(2);
                }
            }

            if (!pw_rows.empty() && !config.dataset.results_dir.empty())
            {
                const std::filesystem::path pw_path =
                    std::filesystem::path(config.dataset.results_dir) /
                    (fold_output_tag(config, dataset_name) + "_per_window_errors.csv");
                write_per_window_errors_csv(pw_path, pw_rows);
            }

            // Re-attach to the on-disk file by path before the terminal event:
            // an external process (e.g. a `git` working-tree op on a tracked
            // events file) may have swapped the inode mid-run, leaving our fd
            // pointed at an unlinked orphan.
            ExperimentEvents::instance().reopen_append();
            ExperimentEvents::instance().emit(
                "fold_end", {{"dataset", dataset_name}, {"pw_rows", pw_rows.size()}});
        }

        ExperimentEvents::instance().reopen_append();
        ExperimentEvents::instance().emit(
            "session_end", {{"status", "ok"}, {"n_rows", all_rows.size()}});
        ExperimentEvents::instance().close();

        nn::progress::ProgressManager::instance().complete_bar(run_bar);
        nn::progress::ProgressManager::instance().shutdown();

        if (config.experiment.repeats > 1 && config.experiment.check_determinism)
        {
            validate_repeat_determinism(config, all_rows);
        }

        write_experiment_outputs(config, cfg_hash, all_rows, out_dir);

        flushProgressAsync();
        return 0;
    }
    catch (const std::exception& ex)
    {
        NN_LOG_ERROR(std::string("[comparative] Fatal error: ") + ex.what());
        ExperimentEvents::instance().reopen_append();
        ExperimentEvents::instance().emit("session_error", {{"what", ex.what()}});
        ExperimentEvents::instance().close();
        return 1;
    }
}

} // namespace meeting01
