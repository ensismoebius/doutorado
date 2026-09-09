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
#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Output.hpp"
#include "../include/Meeting01PerWindow.hpp"
#include "../include/Meeting01Runner.hpp"
#include "../include/Meeting01Training.hpp"
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
#if defined(NN_BACKEND_OPENCL)
    return "opencl";
#elif defined(NN_BACKEND_DEVICE)
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

// Trains one fixed-architecture baseline family (LSTM-/GRU-/Transformer-AE) for this
// (dataset, encoding, run_id): fit on train, early-stop on val, then evaluate once on
// val and once on the held-out test speaker. Emits a "val" row always and a "test" row
// whenever the fold carries a test partition. Model type and train entry point are the
// only things that vary, so this is a template over the concrete AE.
template <typename Model>
void run_baseline(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    const BaselineFamily& fam,
    Model& model,
    TrainResult (*train_fn)(Model&,
        const Meeting01Config&,
        const std::vector<Tensor>&,
        const std::vector<Tensor>&,
        const std::string&,
        std::uint32_t,
        std::size_t,
        std::size_t,
        float&,
        float&),
    const std::string& encoding,
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
    const bool has_test = !split.test_samples.empty();

    auto append_pw = [&](const std::vector<Tensor>& samples,
                         const std::vector<WindowMetadata>& meta,
                         const std::string& split_name)
    {
        PerWindowError proto;
        proto.model = fam.token;
        proto.architecture = fam.arch;
        proto.run_id = run_id + 1;
        proto.seed = run_seed;
        proto.cv_fold = config.dataset.cv_fold;
        proto.split = split_name;
        auto pw = per_window_errors_ae(
            model, samples, meta, encoding, run_seed, config.model.lstm_frame_size, proto);
        pw_rows.insert(pw_rows.end(), pw.begin(), pw.end());
    };

    CheckpointKey val_key{config.experiment.run_tag,
        backend_name,
        dataset_name,
        fam.token,
        encoding,
        fam.arch,
        0.0f,
        0.0f,
        run_id + 1,
        "val",
        config.dataset.cv_fold};
    CheckpointKey test_key = val_key;
    test_key.split = "test";

    const auto val_chk = checkpoint_path(chk_dir, val_key);
    const auto test_chk = checkpoint_path(chk_dir, test_key);

    const std::string config_id =
        make_config_id(fam.token, encoding, "baseline", 0.0f, 0.0f, run_seed, run_id + 1);

    const bool val_cached = checkpoint_is_valid(val_chk, cfg_hash);
    const bool test_cached = !has_test || checkpoint_is_valid(test_chk, cfg_hash);
    if (val_cached && test_cached)
    {
        all_rows.push_back(checkpoint_load(val_chk));
        emit_config_end(all_rows.back(), config_id, "baseline");
        if (has_test)
        {
            all_rows.push_back(checkpoint_load(test_chk));
            emit_config_end(all_rows.back(), config_id, "baseline");
        }
        nn::progress::ProgressManager::instance().update_bar(
            run_bar, static_cast<float>(++completed_runs));
        return;
    }

    EventContext evctx;
    evctx.config_id = config_id;
    evctx.model = fam.token;
    evctx.encoding = encoding;
    evctx.role = "baseline";
    evctx.run_id = run_id + 1;
    evctx.seed = run_seed;
    evctx.max_epochs = config.training.epochs;
    evctx.lr = config.training.learning_rate;
    evctx.lr_biophysical = config.training.learning_rate_biophysical;
    evctx.early_stop_patience = config.training.early_stop_patience;
    ExperimentEvents::instance().set_pending_context(evctx);

    nn::progress::ProgressManager::instance().set_description(run_bar,
        dataset_name + " fold" + std::to_string(config.dataset.cv_fold) + " · " + fam.token +
            " · " + encoding + " · seed " + std::to_string(run_seed));

    float train_ms = 0.0f;
    float infer_ms = 0.0f;

    TrainResult train_result = train_fn(model,
        config,
        split.train_samples,
        split.val_samples,
        encoding,
        run_seed,
        static_cast<std::size_t>(run_id),
        static_cast<std::size_t>(config.experiment.repeats),
        train_ms,
        infer_ms);

    RunMetrics val_metrics = train_result.metrics;
    val_metrics.train_ms = train_ms;

    if (config.dataset.save_models)
    {
        const std::string base_name = sanitize_name(
            config.experiment.run_tag + "_" + fam.arch + "_" + dataset_name + "_" + encoding +
            "_fold" + std::to_string(config.dataset.cv_fold) + "_run" + std::to_string(run_id + 1));
        if (!save_state_dict_text(models_dir / (base_name + "_state_dict.txt"), model.state_dict()))
        {
            NN_LOG_WARN(
                "[comparative] failed to save " + fam.token + " state_dict for " + base_name);
        }
    }

    all_rows.push_back(make_baseline_row(config,
        backend_name,
        dataset_name,
        fam,
        encoding,
        run_id,
        run_seed,
        cfg_hash,
        "val",
        val_metrics));
    checkpoint_save(val_chk, all_rows.back(), train_result.history, cfg_hash);
    emit_config_end(all_rows.back(), config_id, "baseline");
    append_pw(split.val_samples, split.val_meta, "val");

    if (has_test)
    {
        const RunMetrics test_metrics = evaluate_ae(model,
            split.test_samples,
            std::vector<int>(split.test_samples.size(), 0),
            config.training.max_reconstruct_mean_deviation,
            val_metrics.macs,
            val_metrics.parameter_count,
            encoding,
            run_seed,
            0.0f,
            config.model.lstm_frame_size);
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
        checkpoint_save(test_chk, all_rows.back(), train_result.history, cfg_hash);
        emit_config_end(all_rows.back(), config_id, "baseline");
        append_pw(split.test_samples, split.test_meta, "test");
    }

    nn::progress::ProgressManager::instance().update_bar(
        run_bar, static_cast<float>(++completed_runs));
}

// Dispatches one baseline family token to the right concrete AE + train entry point.
void run_baseline_family(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    const std::string& family_token,
    const std::string& encoding,
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
    if (family_token == "lstm-ae")
    {
        nn::models::lstm::LSTMAutoencoder model(make_lstm_cfg(config));
        run_baseline<nn::models::lstm::LSTMAutoencoder>(config,
            split,
            dataset_name,
            BaselineFamily{"lstm-ae", "lstm"},
            model,
            &train_with_early_stopping_lstm,
            encoding,
            run_id,
            run_seed,
            backend_name,
            cfg_hash,
            chk_dir,
            models_dir,
            run_bar,
            completed_runs,
            all_rows,
            pw_rows);
    }
    else if (family_token == "gru-ae")
    {
        nn::models::gru::GRUAutoencoder model(make_gru_cfg(config));
        run_baseline<nn::models::gru::GRUAutoencoder>(config,
            split,
            dataset_name,
            BaselineFamily{"gru-ae", "gru"},
            model,
            &train_with_early_stopping_gru,
            encoding,
            run_id,
            run_seed,
            backend_name,
            cfg_hash,
            chk_dir,
            models_dir,
            run_bar,
            completed_runs,
            all_rows,
            pw_rows);
    }
    else if (family_token == "transformer-ae")
    {
        nn::models::transformer::TransformerAutoencoder model(make_transformer_cfg(config));
        run_baseline<nn::models::transformer::TransformerAutoencoder>(config,
            split,
            dataset_name,
            BaselineFamily{"transformer-ae", "transformer"},
            model,
            &train_with_early_stopping_transformer,
            encoding,
            run_id,
            run_seed,
            backend_name,
            cfg_hash,
            chk_dir,
            models_dir,
            run_bar,
            completed_runs,
            all_rows,
            pw_rows);
    }
    else
    {
        throw std::invalid_argument(
            "run_baseline_family: unknown baseline '" + family_token +
            "' — validate() should have rejected this. Valid: lstm-ae, gru-ae, transformer-ae.");
    }
}

/** Writes the per-run epoch-history and batch-convergence .dat files for one SNN combo,
 *  when LaTeX data export is configured. */
void write_snn_combo_dats(const Meeting01Config& config,
    const std::string& encoding,
    const std::string& architecture,
    float voltage_threshold,
    float alpha,
    int run_id,
    const TrainResult& train_result)
{
    if (config.dataset.latex_data_dir.empty())
    {
        return;
    }

    const std::filesystem::path latex_dir = std::filesystem::path(config.dataset.latex_data_dir);
    write_epoch_history_dat(
        latex_dir / (config.experiment.run_tag + "_snn_" + encoding + "_" + architecture + "_vth" +
                        std::to_string(voltage_threshold) + "_a" + std::to_string(alpha) + "_run" +
                        std::to_string(run_id + 1) + "_history.dat"),
        "snn-ae",
        encoding,
        architecture,
        voltage_threshold,
        alpha,
        run_id + 1,
        train_result.history);
    write_batch_convergence_dat(
        latex_dir / (config.experiment.run_tag + "_snn_" + encoding + "_" + architecture + "_vth" +
                        std::to_string(voltage_threshold) + "_a" + std::to_string(alpha) + "_run" +
                        std::to_string(run_id + 1) + "_convergence.dat"),
        "snn-ae",
        encoding,
        architecture,
        voltage_threshold,
        alpha,
        run_id + 1,
        train_result.history);
}

/** Writes the encoder/decoder parameter dumps for one SNN model, when model saving is
 *  configured. `role_tag` ("combo" for a sweep candidate, "final" for the retrained
 *  winner) plus the fold index keep the ~18 nested-LOSO processes from colliding on the
 *  same filename. */
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

// Trains (or loads from checkpoint) one SNN-AE sweep candidate for this (dataset,
// encoding, architecture, voltage_threshold, alpha, run_id) combo on `train`, scores it
// on the inner validation speaker, appends a split="val" ResultRow, and returns that
// validation MSE so the caller can select the fold's winner. The held-out test speaker
// is never touched here.
auto run_snn_combo(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    const std::string& encoding,
    const std::string& architecture,
    float voltage_threshold,
    float alpha,
    int run_id,
    std::uint32_t run_seed,
    const std::string& backend_name,
    std::size_t cfg_hash,
    const std::filesystem::path& chk_dir,
    const std::filesystem::path& models_dir,
    std::uint32_t run_bar,
    int& completed_runs,
    std::vector<ResultRow>& all_rows,
    std::vector<PerWindowError>& pw_rows) -> float
{
    const CheckpointKey snn_key{config.experiment.run_tag,
        backend_name,
        dataset_name,
        "snn-ae",
        encoding,
        architecture,
        voltage_threshold,
        alpha,
        run_id + 1,
        "val",
        config.dataset.cv_fold};
    const auto snn_chk = checkpoint_path(chk_dir, snn_key);

    const std::string config_id = make_config_id(
        "snn-ae", encoding, "snn_sweep", voltage_threshold, alpha, run_seed, run_id + 1);

    if (checkpoint_is_valid(snn_chk, cfg_hash))
    {
        all_rows.push_back(checkpoint_load(snn_chk));
        emit_config_end(all_rows.back(), config_id, "snn_sweep");
        nn::progress::ProgressManager::instance().update_bar(
            run_bar, static_cast<float>(++completed_runs));
        return all_rows.back().metrics.mse;
    }

    {
        EventContext evctx;
        evctx.config_id = config_id;
        evctx.model = "snn-ae";
        evctx.encoding = encoding;
        evctx.role = "snn_sweep";
        evctx.hyperparams = {
            {"v_th", voltage_threshold}, {"alpha", alpha}, {"architecture", architecture}};
        evctx.run_id = run_id + 1;
        evctx.seed = run_seed;
        evctx.max_epochs = config.training.epochs;
        evctx.lr = config.training.learning_rate;
        evctx.lr_biophysical = config.training.learning_rate_biophysical;
        evctx.early_stop_patience = config.training.early_stop_patience;
        ExperimentEvents::instance().set_pending_context(evctx);
    }

    {
        std::ostringstream d;
        d << dataset_name << " fold" << config.dataset.cv_fold << " · SNN-" << architecture << " · "
          << encoding << " · seed " << run_seed << " · v=" << std::fixed << std::setprecision(2)
          << voltage_threshold << " a=" << alpha;
        nn::progress::ProgressManager::instance().set_description(run_bar, d.str());
    }

    float train_ms = 0.0f;
    float infer_ms = 0.0f;

    AutoencoderConfig snn_config = make_snn_cfg( //
        config,                                  //
        alpha,                                   //
        voltage_threshold                        //
    );
    snn_config.initializer_seed = run_seed;
    snn_config.initializer_sampler_type =
        "comparative|" + dataset_name + "|" + encoding + "|" + architecture + "|" +
        std::to_string(extract_layer_sizes(config.model.encoder_layer_spec).empty()
                           ? 0
                           : extract_layer_sizes(config.model.encoder_layer_spec).front()) +
        "|" + std::to_string(voltage_threshold) + "|" + std::to_string(alpha);

    ProtocolSpikingAutoencoder snn_model(snn_config);

    TrainResult train_result = train_with_early_stopping_snn( //
        snn_model,                                            //
        config,                                               //
        split.train_samples,                                  //
        split.val_samples,                                    //
        split.val_labels,                                     //
        encoding,                                             //
        architecture,                                         //
        alpha,                                                //
        voltage_threshold,                                    //
        run_seed,                                             //
        static_cast<std::size_t>(run_id),                     //
        static_cast<std::size_t>(config.experiment.repeats),  //
        train_ms,                                             //
        infer_ms                                              //
    );

    RunMetrics metrics = train_result.metrics;
    metrics.train_ms = train_ms;

    write_snn_combo_dats(
        config, encoding, architecture, voltage_threshold, alpha, run_id, train_result);
    save_snn_combo_models(config,
        dataset_name,
        encoding,
        architecture,
        voltage_threshold,
        alpha,
        run_id,
        models_dir,
        snn_model);

    ResultRow snn_row{backend_name,
        config.experiment.run_tag,
        dataset_name,
        "snn-ae",
        encoding,
        architecture,
        static_cast<int>(config.model.encoder_layer_spec.size()),
        voltage_threshold,
        alpha,
        run_id + 1,
        run_seed,
        cfg_hash,
        metrics};
    snn_row.split = "val";
    snn_row.cv_fold = config.dataset.cv_fold;
    all_rows.push_back(snn_row);
    checkpoint_save(snn_chk, all_rows.back(), train_result.history, cfg_hash);
    emit_config_end(all_rows.back(), config_id, "snn_sweep");

    PerWindowError proto;
    proto.model = "snn-ae";
    proto.architecture = architecture;
    proto.v_th = voltage_threshold;
    proto.alpha = alpha;
    proto.run_id = run_id + 1;
    proto.seed = run_seed;
    proto.cv_fold = config.dataset.cv_fold;
    proto.split = "val";
    auto pw = per_window_errors_snn(snn_model,
        split.val_samples,
        split.val_meta,
        encoding,
        architecture,
        alpha,
        voltage_threshold,
        run_seed,
        proto);
    pw_rows.insert(pw_rows.end(), pw.begin(), pw.end());

    nn::progress::ProgressManager::instance().update_bar(
        run_bar, static_cast<float>(++completed_runs));
    return metrics.mse;
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

// Runs the full SNN architecture × voltage_threshold × alpha sweep for one (dataset,
// encoding, run_id) combo.
void run_snn_sweep(const Meeting01Config& config,
    const DatasetSplit& split,
    const std::string& dataset_name,
    const std::string& encoding,
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
    struct Candidate
    {
        std::string architecture;
        float v_th;
        float alpha;
        float val_mse;
    };
    std::vector<Candidate> candidates;

    for (const auto& architecture : config.evaluation.snn_architectures)
    {
        for (float voltage_threshold : config.evaluation.v_th_values)
        {
            for (float alpha : config.evaluation.alpha_values)
            {
                const float val_mse = run_snn_combo(config,
                    split,
                    dataset_name,
                    encoding,
                    architecture,
                    voltage_threshold,
                    alpha,
                    run_id,
                    run_seed,
                    backend_name,
                    cfg_hash,
                    chk_dir,
                    models_dir,
                    run_bar,
                    completed_runs,
                    all_rows,
                    pw_rows);
                candidates.push_back({architecture, voltage_threshold, alpha, val_mse});
            }
        }
    }

    // Nested-LOSO final fit: only when the fold carries a held-out test speaker.
    if (split.test_samples.empty() || candidates.empty()) return;

    const auto best = *std::min_element(candidates.begin(),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.val_mse < b.val_mse; });

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

    AutoencoderConfig snn_config = make_snn_cfg(config, best.alpha, best.v_th);
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

    RunMetrics test_metrics = evaluate_snn(snn_model,
        split.test_samples,
        std::vector<int>(split.test_samples.size(), 0),
        config.training.max_reconstruct_mean_deviation,
        estimate_snn_macs(static_cast<std::size_t>(config.dataset.window_size),
            extract_layer_sizes(config.model.encoder_layer_spec).empty()
                ? 0
                : extract_layer_sizes(config.model.encoder_layer_spec).front(),
            static_cast<int>(extract_layer_sizes(config.model.encoder_layer_spec).size())),
        parameter_count(snn_model.params()),
        encoding,
        best.architecture,
        best.alpha,
        best.v_th,
        run_seed,
        infer_ms);
    test_metrics.train_ms = train_ms;

    ResultRow test_row{backend_name,
        config.experiment.run_tag,
        dataset_name,
        "snn-ae",
        encoding,
        best.architecture,
        static_cast<int>(config.model.encoder_layer_spec.size()),
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
        man["test_speaker"] = split.test_speaker;
        man["selected"] = {{"architecture", best.architecture},
            {"v_th", best.v_th},
            {"alpha", best.alpha},
            {"val_mse", best.val_mse}};
        for (const auto& c : candidates)
            man["candidates"].push_back({{"architecture", c.architecture},
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

// Hard leakage gate + split manifest. Aborts the run (named exception, no fallback) if
// any speaker or any source recording appears in more than one of train/val/test.
void assert_split_disjoint_and_manifest(
    const Meeting01Config& config, const DatasetSplit& split, const std::string& dataset_name)
{
    if (config.dataset.cv_fold < 0) return; // legacy pooled path — not a LOSO fold

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
    const std::uint32_t seed = config.experiment.seed;

    auto encode_matrix =
        [&](const std::vector<Tensor>& samples) -> std::pair<std::vector<float>, std::size_t>
    {
        std::vector<float> flat;
        std::size_t cols = 0;
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const Tensor framed = to_lstm_frames(
                encode_sample(samples[i], encoding, seed + static_cast<std::uint32_t>(i)), frame);
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

        // Total individual runs: datasets × encodings × repeats × (baselines + SNN sweep).
        const int snn_per_combo = static_cast<int>(config.evaluation.snn_architectures.size()) *
                                  static_cast<int>(config.evaluation.v_th_values.size()) *
                                  static_cast<int>(config.evaluation.alpha_values.size());
        const int total_outer_runs =
            static_cast<int>(config.evaluation.datasets.size()) *
            static_cast<int>(config.evaluation.encodings.size()) * config.experiment.repeats *
            (static_cast<int>(config.evaluation.baselines.size()) + snn_per_combo);

        // Overall-progress banner across the whole 4-profile run. Each profile is a separate
        // process, so this process cannot know the outer progress on its own — the wrapper
        // (01_meeting01_run_article_profiles.sh) computes it the same way run_thesis_profiles.sh
        // does (work-weighted, EMA-smoothed seconds-per-unit-work — see scripts/lib/run_eta.sh) and
        // passes the ready-made line in via MEETING01_OVERALL. Logging it renders it as a
        // persistent top line above the per-profile bars; empty/unset when run standalone, so
        // unchanged.
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
                            {"v_th_values", config.evaluation.v_th_values},
                            {"alpha_values", config.evaluation.alpha_values},
                            {"encodings", config.evaluation.encodings},
                            {"baselines", config.evaluation.baselines}}}});
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

            for (const auto& encoding : config.evaluation.encodings)
            {
                dump_analytic_baseline_inputs(config, split, dataset_name, encoding, out_dir);

                for (int run_id = 0; run_id < config.experiment.repeats; ++run_id)
                {
                    const std::uint32_t run_seed =
                        config.experiment.seed_deterministic
                            ? config.experiment.seed
                            : config.experiment.seed + static_cast<std::uint32_t>(run_id);

                    for (const auto& family : config.evaluation.baselines)
                    {
                        run_baseline_family(config,
                            split,
                            dataset_name,
                            family,
                            encoding,
                            run_id,
                            run_seed,
                            backend_name,
                            cfg_hash,
                            chk_dir,
                            models_dir,
                            run_bar,
                            completed_runs,
                            all_rows,
                            pw_rows);
                    }

                    run_snn_sweep(config,
                        split,
                        dataset_name,
                        encoding,
                        run_id,
                        run_seed,
                        backend_name,
                        cfg_hash,
                        chk_dir,
                        models_dir,
                        run_bar,
                        completed_runs,
                        all_rows,
                        pw_rows);
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
