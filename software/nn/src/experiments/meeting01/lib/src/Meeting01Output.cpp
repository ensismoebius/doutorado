#include "../include/Meeting01Output.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

#include "../include/Meeting01PerWindow.hpp"
#include "cnpy.h"
#include "io/ReportIO.hpp"
#include "nlohmann/json.hpp"
#include "statistics/inference_tests.hpp"

namespace meeting01
{

void write_rows_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    // Through <path>.tmp + rename, like checkpoint_save: this file is the RESUME=1 "fold
    // finished" marker, and a process killed while writing it straight to `path` left a
    // truncated but non-empty file that the next RESUME skipped as a finished fold.
    const std::filesystem::path tmp(path.string() + ".tmp");
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out.is_open())
        {
            throw std::runtime_error("write_rows_csv: cannot open " + tmp.string());
        }
        out << "backend,profile,dataset,model,encoding,architecture,layers,v_th,alpha,run,seed,"
               "config_hash,split,cv_fold,";
        out << "mse,mae,r2,precision,recall,f1,spike_rate,arith_energy_nj,train_ms,infer_ms,"
               "param_count,macs,acs,energy_nj,energy_ops_nj,energy_mem_nj,mem_accesses,"
               "energy_nj_sram8k,energy_nj_sram32k,energy_nj_sram1m,energy_nj_as_executed,"
               "energy_nj_encoder,hidden_silent_share,hidden_always_share,lif_v_th_mean,lif_v_th_"
               "min,lif_beta_mean,"
               "train_mse\n";

        out << std::fixed << std::setprecision(6);
        for (const auto& row : rows)
        {
            out << row.backend << ',' << row.profile << ',' << row.dataset << ',' << row.model
                << ',' << row.encoding << ',' << row.architecture << ',' << row.layers << ','
                << row.v_th << ',' << row.alpha << ',' << row.run_id << ',' << row.seed << ','
                << row.config_hash << ',' << row.split << ',' << row.cv_fold << ','
                << row.metrics.mse << ',' << row.metrics.mae << ',' << row.metrics.r2 << ','
                << row.metrics.precision << ',' << row.metrics.recall << ',' << row.metrics.f1
                << ',' << row.metrics.spike_rate << ',' << row.metrics.arith_energy_nj << ','
                << row.metrics.train_ms << ',' << row.metrics.infer_ms << ','
                << row.metrics.parameter_count << ',' << row.metrics.macs << ',' << row.metrics.acs
                << ',' << row.metrics.energy_nj << ',' << row.metrics.energy_ops_nj << ','
                << row.metrics.energy_mem_nj << ',' << row.metrics.mem_accesses << ','
                << row.metrics.energy_nj_sram8k << ',' << row.metrics.energy_nj_sram32k << ','
                << row.metrics.energy_nj_sram1m << ',' << row.metrics.energy_nj_as_executed << ','
                << row.metrics.energy_nj_encoder << ',' << row.metrics.hidden_silent_share << ','
                << row.metrics.hidden_always_share << ',' << row.metrics.lif_v_th_mean << ','
                << row.metrics.lif_v_th_min << ',' << row.metrics.lif_beta_mean << ','
                << row.metrics.train_mse << '\n';
        }
        out.close();
        if (!out)
        {
            throw std::runtime_error("write_rows_csv: writing " + tmp.string() +
                                     " failed (disk full?) -- " + path.string() + " not written");
        }
    }
    std::filesystem::rename(tmp, path);
}

void write_summary_json(const std::filesystem::path& path,
    const Meeting01Config& cfg,
    std::size_t cfg_hash,
    const std::vector<ResultRow>& rows)
{
    nlohmann::json j;
    j["seed"] = cfg.experiment.seed;
    j["config_hash"] = cfg_hash;
    j["experiment"]["seed"] = cfg.experiment.seed;
    j["experiment"]["repeats"] = cfg.experiment.repeats;
    j["training"]["epochs"] = cfg.training.epochs;
    j["training"]["samples_per_batch"] = cfg.training.samples_per_batch;
    j["training"]["batches_per_epoch"] = cfg.training.batches_per_epoch;
    j["dataset"]["window_size"] = cfg.dataset.window_size;
    j["limitation_notes"] = {
        "Surrogate gradient in spiking model is an approximation.",
        "Latency encoding can reduce information throughput.",
        // NOLINTNEXTLINE(bugprone-suspicious-missing-comma): one note split over two lines
        "arith_energy_nj is an arithmetic-only estimate (4.6 pJ/MAC, 0.9 pJ/AC, Horowitz "
        "ISSCC 2014, 45 nm fp32) from counted operations; memory traffic is not included.",
        "spike_rate is the spiking fraction of the hidden LIF (neuron, step) pairs; NaN for "
        "models without spiking neurons.",
        "Evaluation depends on available FSDD/PhysioNet files under dataset_root.",
    };

    std::vector<float> snn_mse;
    std::vector<float> lstm_mse;
    for (const auto& row : rows)
    {
        if (row.model == "snn-ae") snn_mse.push_back(row.metrics.mse);
        if (row.model == "lstm-ae") lstm_mse.push_back(row.metrics.mse);
    }

    j["statistics"]["mse"]["t_test_p"] = statistics::t_test_pvalue_approx(snn_mse, lstm_mse);
    j["statistics"]["mse"]["wilcoxon_p"] =
        statistics::wilcoxon_signed_rank_pvalue_approx(snn_mse, lstm_mse);
    j["statistics"]["mse"]["cohens_d"] = statistics::cohens_d(snn_mse, lstm_mse);

    std::string error;
    if (!nn::io::write_json_file(path, j, 2, &error))
    {
        throw std::runtime_error("Failed to write summary JSON: " + error);
    }
}

void write_publication_table(const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    std::ofstream out(path);
    out << "Model,Codificacao,Camadas,MSE,MAE,R2,F1,SpikeRate,EnergiaAritNj,TempoTreinoMs\n";

    std::map<std::string, std::vector<const ResultRow*>> groups;
    for (const auto& row : rows)
    {
        const std::string key = row.model + "|" + row.encoding + "|" + row.architecture + "|" +
                                std::to_string(row.layers);
        groups[key].push_back(&row);
    }

    for (const auto& [key, vec] : groups)
    {
        float mse = 0.0f;
        float mae = 0.0f;
        float r2 = 0.0f;
        float f1 = 0.0f;
        float spike_rate = 0.0f;
        float arith_energy_nj = 0.0f;
        float train_ms = 0.0f;
        for (const auto* row : vec)
        {
            mse += row->metrics.mse;
            mae += row->metrics.mae;
            r2 += row->metrics.r2;
            f1 += row->metrics.f1;
            spike_rate += row->metrics.spike_rate;
            arith_energy_nj += row->metrics.arith_energy_nj;
            train_ms += row->metrics.train_ms;
        }
        const float n = static_cast<float>(vec.size());

        const auto* first = vec.front();
        out << first->model << ',' << first->encoding << ',' << first->layers << ',' << (mse / n)
            << ',' << (mae / n) << ',' << (r2 / n) << ',' << (f1 / n) << ',' << (spike_rate / n)
            << ',' << (arith_energy_nj / n) << ',' << (train_ms / n) << '\n';
    }
}

auto mean_window_mse(const std::vector<PerWindowError>& rows) -> float
{
    if (rows.empty())
        throw std::invalid_argument(
            "mean_window_mse: no windows. A final fit always has fit windows; check the carve.");
    double sum = 0.0;
    for (const auto& row : rows) sum += row.mse;
    return static_cast<float>(sum / static_cast<double>(rows.size()));
}

void write_per_window_errors_csv(
    const std::filesystem::path& path, const std::vector<PerWindowError>& rows)
{
    // Truncate, never append: a fold re-run into the same results directory used to add
    // its rows after the previous run's, and the readers then averaged the two runs (old
    // binary and new) as if they were one.
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open())
    {
        throw std::runtime_error("write_per_window_errors_csv: cannot open " + path.string());
    }
    out << "model,encoding,architecture,v_th,alpha,run_id,seed,cv_fold,split,"
           "speaker_id,recording_id,window_id,source_window_index,mse,mae\n";
    out << std::fixed << std::setprecision(8);
    for (const auto& r : rows)
    {
        out << r.model << ',' << r.encoding << ',' << r.architecture << ',' << r.v_th << ','
            << r.alpha << ',' << r.run_id << ',' << r.seed << ',' << r.cv_fold << ',' << r.split
            << ',' << r.speaker_id << ',' << r.recording_id << ',' << r.window_id << ','
            << r.source_window_index << ',' << r.mse << ',' << r.mae << '\n';
    }
}

void write_reference_inputs(
    const std::filesystem::path& dir, const std::string& stem, const DatasetSplit& split)
{
    auto write_part = [&](const std::string& part,
                          const std::vector<Tensor>& samples,
                          const std::vector<WindowMetadata>& meta)
    {
        if (samples.empty())
        {
            throw std::runtime_error("write_reference_inputs: " + stem + " has an empty " + part +
                                     " split; the reference baselines need windows in every "
                                     "part. Check the fold's speaker assignment and the "
                                     "loso_max_*_windows caps in the profile.");
        }
        if (meta.size() != samples.size())
        {
            throw std::runtime_error("write_reference_inputs: " + stem + " " + part + " has " +
                                     std::to_string(samples.size()) + " windows but " +
                                     std::to_string(meta.size()) +
                                     " metadata rows; build_split must keep them parallel.");
        }

        const nn::Index width = samples.front().size();
        std::vector<float> flat;
        flat.reserve(samples.size() * static_cast<std::size_t>(width));
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            if (samples[i].size() != width)
            {
                std::string msg = "write_reference_inputs: ";
                msg += stem;
                msg += ' ';
                msg += part;
                msg += " window " + std::to_string(i);
                msg += " has " + std::to_string(samples[i].size());
                msg += " samples, expected " + std::to_string(width);
                msg += "; every window of a dataset must share window_size.";
                throw std::runtime_error(msg);
            }
            if (meta[i].valid_length < 0 || meta[i].valid_length > width)
            {
                std::string msg = "write_reference_inputs: ";
                msg += stem;
                msg += ' ';
                msg += part;
                msg += " window " + std::to_string(i);
                msg += " has valid_length " + std::to_string(meta[i].valid_length);
                msg += " outside [0, " + std::to_string(width);
                throw std::runtime_error(msg + "]; the loader's WindowMetadata is corrupt.");
            }
            for (nn::Index k = 0; k < width; ++k) flat.push_back(samples[i].at(k));
        }

        const std::string base = stem + "_target_" + part + "_windows";
        cnpy::npy_save((dir / (base + ".npy")).string(),
            flat.data(),
            {samples.size(), static_cast<std::size_t>(width)},
            "w");

        const std::filesystem::path meta_path = dir / (base + "_meta.csv");
        std::ofstream out(meta_path);
        if (!out.is_open())
        {
            throw std::runtime_error("write_reference_inputs: cannot open " + meta_path.string() +
                                     " for writing; check that the results directory exists "
                                     "and is writable.");
        }
        out << "speaker_id,recording_id,window_id,source_window_index,valid_length\n";
        for (const auto& m : meta)
        {
            out << m.speaker_id << ',' << m.recording_id << ',' << m.window_id << ','
                << m.source_window_index << ',' << m.valid_length << '\n';
        }
    };

    write_part("train", split.train_samples, split.train_meta);
    write_part("val", split.val_samples, split.val_meta);
    write_part("test", split.test_samples, split.test_meta);
}

void validate_repeat_determinism(const Meeting01Config& cfg, const std::vector<ResultRow>& rows)
{
    if (cfg.experiment.repeats <= 1) return;

    std::map<std::string, std::vector<const ResultRow*>> groups;
    for (const auto& row : rows)
    {
        const std::string key = row.dataset + "|" + row.model + "|" + row.encoding + "|" +
                                row.architecture + "|" + std::to_string(row.layers) + "|" +
                                std::to_string(row.v_th) + "|" + std::to_string(row.alpha);
        groups[key].push_back(&row);
    }

    auto almost_eq = [](float a, float b) { return std::fabs(a - b) <= 1e-4f; };
    auto rel_eq = [](float a, float b)
    {
        if (a == 0.0f && b == 0.0f) return true;
        const float diff = std::fabs(a - b);
        const float scale = std::max(std::fabs(a), std::fabs(b));
        return diff <= 1e-4f * scale;
    };

    for (const auto& [key, group] : groups)
    {
        if (static_cast<int>(group.size()) != cfg.experiment.repeats)
        {
            throw std::runtime_error("Determinism check failed (missing repeats) for key: " + key);
        }

        const RunMetrics& ref = group.front()->metrics;
        for (std::size_t i = 1; i < group.size(); ++i)
        {
            const RunMetrics& cur = group[i]->metrics;

            std::ostringstream diff;
            bool has_diff = false;

            if (!almost_eq(ref.mse, cur.mse) && !rel_eq(ref.mse, cur.mse))
            {
                diff << " mse(" << ref.mse << " vs " << cur.mse << ")";
                has_diff = true;
            }
            if (!almost_eq(ref.mae, cur.mae) && !rel_eq(ref.mae, cur.mae))
            {
                diff << " mae(" << ref.mae << " vs " << cur.mae << ")";
                has_diff = true;
            }
            if (!almost_eq(ref.r2, cur.r2) && !rel_eq(ref.r2, cur.r2))
            {
                diff << " r2(" << ref.r2 << " vs " << cur.r2 << ")";
                has_diff = true;
            }
            if (!almost_eq(ref.f1, cur.f1) && !rel_eq(ref.f1, cur.f1))
            {
                diff << " f1(" << ref.f1 << " vs " << cur.f1 << ")";
                has_diff = true;
            }
            // NaN on both sides (a model with no spiking neuron) is the same value here.
            const bool both_nan = std::isnan(ref.spike_rate) && std::isnan(cur.spike_rate);
            if (!both_nan && !almost_eq(ref.spike_rate, cur.spike_rate) &&
                !rel_eq(ref.spike_rate, cur.spike_rate))
            {
                diff << " spike_rate(" << ref.spike_rate << " vs " << cur.spike_rate << ")";
                has_diff = true;
            }
            if (!almost_eq(ref.arith_energy_nj, cur.arith_energy_nj) &&
                !rel_eq(ref.arith_energy_nj, cur.arith_energy_nj))
            {
                diff << " arith_energy_nj(" << ref.arith_energy_nj << " vs " << cur.arith_energy_nj
                     << ")";
                has_diff = true;
            }

            if (has_diff)
            {
                throw std::runtime_error(
                    "Determinism check failed (metrics differ) for key: " + key + diff.str());
            }
        }
    }
}

} // namespace meeting01
