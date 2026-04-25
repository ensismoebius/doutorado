#include "../include/ComparativeOutput.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "nn/io/ReportIO.hpp"
#include "nn/statistics/inference_tests.hpp"

namespace comparative_autoencoder_experiment
{

void write_rows_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    std::ofstream out(path);
    out << "dataset,model,encoding,architecture,layers,v_th,alpha,run,seed,config_hash,";
    out << "mse,mae,r2,precision,recall,f1,spike_rate,energy,train_ms,infer_ms,param_count,macs\n";

    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << row.dataset << ',' << row.model << ',' << row.encoding << ',' << row.architecture
            << ',' << row.layers << ',' << row.v_th << ',' << row.alpha << ',' << row.run_id << ','
            << row.seed << ',' << row.config_hash << ',' << row.metrics.mse << ','
            << row.metrics.mae << ',' << row.metrics.r2 << ',' << row.metrics.precision << ','
            << row.metrics.recall << ',' << row.metrics.f1 << ',' << row.metrics.spike_rate << ','
            << row.metrics.energy << ',' << row.metrics.train_ms << ',' << row.metrics.infer_ms
            << ',' << row.metrics.parameter_count << ',' << row.metrics.macs << '\n';
    }
}

void write_summary_json(const std::filesystem::path& path,
    const ComparativeConfig& cfg,
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
        "Energy metric is a proxy: spikes + 10*MACs.",
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
    out << "Model,Codificacao,Camadas,MSE,MAE,R2,F1,SpikeRate,Energia,TempoTreinoMs\n";

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
        float energy = 0.0f;
        float train_ms = 0.0f;
        for (const auto* row : vec)
        {
            mse += row->metrics.mse;
            mae += row->metrics.mae;
            r2 += row->metrics.r2;
            f1 += row->metrics.f1;
            spike_rate += row->metrics.spike_rate;
            energy += row->metrics.energy;
            train_ms += row->metrics.train_ms;
        }
        const float n = static_cast<float>(vec.size());

        const auto* first = vec.front();
        out << first->model << ',' << first->encoding << ',' << first->layers << ',' << (mse / n)
            << ',' << (mae / n) << ',' << (r2 / n) << ',' << (f1 / n) << ',' << (spike_rate / n)
            << ',' << (energy / n) << ',' << (train_ms / n) << '\n';
    }
}

void validate_repeat_determinism(const ComparativeConfig& cfg, const std::vector<ResultRow>& rows)
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
    auto rel_eq = [](float a, float b) {
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
            if (!almost_eq(ref.spike_rate, cur.spike_rate) && !rel_eq(ref.spike_rate, cur.spike_rate))
            {
                diff << " spike_rate(" << ref.spike_rate << " vs " << cur.spike_rate << ")";
                has_diff = true;
            }
            if (!almost_eq(ref.energy, cur.energy) && !rel_eq(ref.energy, cur.energy))
            {
                diff << " energy(" << ref.energy << " vs " << cur.energy << ")";
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

} // namespace comparative_autoencoder_experiment
