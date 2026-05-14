#include "../include/ComparativeOutput.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "io/ReportIO.hpp"
#include "statistics/inference_tests.hpp"

namespace comparative_autoencoder_experiment
{

namespace
{

template <typename T>
auto join_values(const std::vector<T>& values, const char* sep = ";") -> std::string
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0) oss << sep;
        oss << values[i];
    }
    return oss.str();
}

auto model_label(const ResultRow& row) -> std::string
{
    if (row.model == "lstm-ae") return "LSTM-AE";
    if (row.architecture == "dense") return "SNN-dense";
    if (row.architecture == "conv1d") return "SNN-conv1d";
    if (row.architecture == "recurrent") return "SNN-recurrent";
    return row.model;
}

struct Aggregated
{
    double mse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double spike_rate = 0.0;
    double energy = 0.0;
    double train_ms = 0.0;
    double infer_ms = 0.0;
    double param_count = 0.0;
    double macs = 0.0;
    int count = 0;

    void add(const ResultRow& row)
    {
        mse += row.metrics.mse;
        mae += row.metrics.mae;
        r2 += row.metrics.r2;
        precision += row.metrics.precision;
        recall += row.metrics.recall;
        f1 += row.metrics.f1;
        spike_rate += row.metrics.spike_rate;
        energy += row.metrics.energy;
        train_ms += row.metrics.train_ms;
        infer_ms += row.metrics.infer_ms;
        param_count += static_cast<double>(row.metrics.parameter_count);
        macs += static_cast<double>(row.metrics.macs);
        ++count;
    }
};

template <typename Getter>
auto avg(const Aggregated& a, Getter getter) -> double
{
    if (a.count == 0) return 0.0;
    return getter(a) / static_cast<double>(a.count);
}

} // namespace

void write_rows_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    std::ofstream out(path);
    out << "backend,profile,dataset,model,encoding,architecture,layers,v_th,alpha,run,seed,config_"
           "hash,";
    out << "mse,mae,r2,precision,recall,f1,spike_rate,energy,train_ms,infer_ms,param_count,macs\n";

    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << row.backend << ',' << row.profile << ',' << row.dataset << ',' << row.model << ','
            << row.encoding << ',' << row.architecture << ',' << row.layers << ',' << row.v_th
            << ',' << row.alpha << ',' << row.run_id << ',' << row.seed << ',' << row.config_hash
            << ',' << row.metrics.mse << ',' << row.metrics.mae << ',' << row.metrics.r2 << ','
            << row.metrics.precision << ',' << row.metrics.recall << ',' << row.metrics.f1 << ','
            << row.metrics.spike_rate << ',' << row.metrics.energy << ',' << row.metrics.train_ms
            << ',' << row.metrics.infer_ms << ',' << row.metrics.parameter_count << ','
            << row.metrics.macs << '\n';
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

void write_latex_exports(const std::filesystem::path& dir,
    const std::string& run_tag,
    const ComparativeConfig& cfg,
    const std::vector<ResultRow>& rows)
{
    namespace fs = std::filesystem;
    fs::create_directories(dir);

    const fs::path summary_path = dir / (run_tag + "_summary_by_model.csv");
    const fs::path recon_path = dir / (run_tag + "_recon_by_encoding.csv");
    const fs::path eff_path = dir / (run_tag + "_efficiency_by_encoding.csv");
    const fs::path mse_plot_path = dir / (run_tag + "_mse_plot.csv");
    const fs::path sweep_path = dir / (run_tag + "_sweep_alpha.csv");
    const fs::path backend_path = dir / (run_tag + "_backend_timing.csv");
    const fs::path profile_manifest_path = dir / (run_tag + "_profile_manifest.csv");

    {
        std::map<std::string, Aggregated> by_model;
        for (const auto& row : rows)
        {
            by_model[model_label(row)].add(row);
        }

        std::ofstream out(summary_path);
        out << "model,mse,mae,r2,spike_rate,energy,infer_ms,train_ms,param_count,macs\n";
        out << std::fixed << std::setprecision(6);
        for (const auto& [model, agg] : by_model)
        {
            out << model << ',' << avg(agg, [](const Aggregated& a) { return a.mse; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.mae; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.r2; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.spike_rate; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.energy; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.infer_ms; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.train_ms; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.param_count; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.macs; }) << '\n';
        }
    }

    {
        std::map<std::string, Aggregated> by_model_encoding;
        for (const auto& row : rows)
        {
            const std::string key = model_label(row) + "|" + row.encoding;
            by_model_encoding[key].add(row);
        }

        std::ofstream recon(recon_path);
        std::ofstream eff(eff_path);
        recon << "model,encoding,mse,mae,r2,f1\n";
        eff << "model,encoding,spike_rate,energy,param_count,macs\n";

        recon << std::fixed << std::setprecision(6);
        eff << std::fixed << std::setprecision(6);

        for (const auto& [key, agg] : by_model_encoding)
        {
            const std::size_t pos = key.find('|');
            const std::string model = key.substr(0, pos);
            const std::string encoding = key.substr(pos + 1);
            recon << model << ',' << encoding << ','
                  << avg(agg, [](const Aggregated& a) { return a.mse; }) << ','
                  << avg(agg, [](const Aggregated& a) { return a.mae; }) << ','
                  << avg(agg, [](const Aggregated& a) { return a.r2; }) << ','
                  << avg(agg, [](const Aggregated& a) { return a.f1; }) << '\n';

            eff << model << ',' << encoding << ','
                << avg(agg, [](const Aggregated& a) { return a.spike_rate; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.energy; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.param_count; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.macs; }) << '\n';
        }
    }

    {
        std::map<std::string, Aggregated> by_encoding;
        for (const auto& row : rows)
        {
            const std::string model = model_label(row);
            by_encoding[row.encoding + "|" + model].add(row);
        }

        std::ofstream out(mse_plot_path);
        out << "encoding,lstm_ae,snn_dense,snn_conv1d,snn_recurrent\n";
        out << std::fixed << std::setprecision(6);
        const std::vector<std::string> encodings = {"direct", "poisson", "latency"};
        for (const auto& encoding : encodings)
        {
            auto get_mse = [&](const std::string& model_key)
            {
                const auto it = by_encoding.find(encoding + "|" + model_key);
                if (it == by_encoding.end()) return 0.0;
                return avg(it->second, [](const Aggregated& a) { return a.mse; });
            };
            out << encoding << ',' << get_mse("LSTM-AE") << ',' << get_mse("SNN-dense") << ','
                << get_mse("SNN-conv1d") << ',' << get_mse("SNN-recurrent") << '\n';
        }
    }

    {
        std::map<float, std::map<std::string, Aggregated>> by_alpha_arch;
        for (const auto& row : rows)
        {
            if (row.model != "snn-ae") continue;
            if (row.encoding != "direct") continue;
            if (std::fabs(row.v_th - 1.0f) > 1e-5f) continue;
            by_alpha_arch[row.alpha][row.architecture].add(row);
        }

        std::ofstream out(sweep_path);
        out << "alpha,dense,conv1d,recurrent\n";
        out << std::fixed << std::setprecision(6);
        for (const auto& [alpha, arch_map] : by_alpha_arch)
        {
            auto get_mse = [&](const std::string& arch)
            {
                const auto it = arch_map.find(arch);
                if (it == arch_map.end()) return 0.0;
                return avg(it->second, [](const Aggregated& a) { return a.mse; });
            };
            out << alpha << ',' << get_mse("dense") << ',' << get_mse("conv1d") << ','
                << get_mse("recurrent") << '\n';
        }
    }

    {
        std::map<std::string, Aggregated> by_backend_model;
        for (const auto& row : rows)
        {
            const std::string key = row.backend + "|" + model_label(row);
            by_backend_model[key].add(row);
        }

        std::ofstream out(backend_path);
        out << "backend,model,train_ms,infer_ms\n";
        out << std::fixed << std::setprecision(6);
        for (const auto& [key, agg] : by_backend_model)
        {
            const std::size_t pos = key.find('|');
            const std::string backend = key.substr(0, pos);
            const std::string model = key.substr(pos + 1);
            out << backend << ',' << model << ','
                << avg(agg, [](const Aggregated& a) { return a.train_ms; }) << ','
                << avg(agg, [](const Aggregated& a) { return a.infer_ms; }) << '\n';
        }
    }

    {
        std::ofstream out(profile_manifest_path);
        out << "run_tag,seed,repeats,datasets,encodings,snn_architectures,v_th_values,alpha_values,"
               "window_size,train_samples,val_samples,backend\n";
        out << run_tag << ',' << cfg.experiment.seed << ',' << cfg.experiment.repeats << ',' << '"'
            << join_values(cfg.evaluation.datasets) << '"' << ',' << '"'
            << join_values(cfg.evaluation.encodings) << '"' << ',' << '"'
            << join_values(cfg.evaluation.snn_architectures) << '"' << ',' << '"'
            << join_values(cfg.evaluation.v_th_values) << '"' << ',' << '"'
            << join_values(cfg.evaluation.alpha_values) << '"' << ',' << cfg.dataset.window_size
            << ',' << cfg.dataset.max_loaded_train_samples << ','
            << cfg.dataset.max_validation_samples << ','
            << (rows.empty() ? "unknown" : rows.front().backend) << '\n';
    }
}

void write_pgfplots_summary_dat(
    const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    std::map<std::string, Aggregated> by_config;
    for (const auto& row : rows)
    {
        const std::string key = row.model + "|" + row.encoding + "|" + row.architecture + "|" +
                                std::to_string(row.v_th) + "|" + std::to_string(row.alpha);
        by_config[key].add(row);
    }

    std::ofstream out(path);
    out << "model encoding architecture v_th alpha mse mae r2 f1 spike_rate energy infer_ms "
           "train_ms param_count macs\n";
    out << std::fixed << std::setprecision(4);

    for (const auto& [key, agg] : by_config)
    {
        const std::size_t pos1 = key.find('|');
        const std::size_t pos2 = key.find('|', pos1 + 1);
        const std::size_t pos3 = key.find('|', pos2 + 1);
        const std::size_t pos4 = key.find('|', pos3 + 1);

        const std::string model = key.substr(0, pos1);
        const std::string encoding = key.substr(pos1 + 1, pos2 - pos1 - 1);
        const std::string architecture = key.substr(pos2 + 1, pos3 - pos2 - 1);
        const std::string v_th_str = key.substr(pos3 + 1, pos4 - pos3 - 1);
        const std::string alpha_str = key.substr(pos4 + 1);

        out << model << ' ' << encoding << ' ' << architecture << ' ' << v_th_str << ' '
            << alpha_str << ' ' << avg(agg, [](const Aggregated& a) { return a.mse; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.mae; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.r2; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.f1; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.spike_rate; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.energy; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.infer_ms; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.train_ms; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.param_count; }) << ' '
            << avg(agg, [](const Aggregated& a) { return a.macs; }) << '\n';
    }
}

void write_pgfplots_sweep_dat(const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    std::map<float, std::map<std::string, Aggregated>> by_alpha_arch;
    for (const auto& row : rows)
    {
        if (row.model != "snn-ae") continue;
        if (row.encoding != "direct") continue;
        if (std::fabs(row.v_th - 1.0f) > 1e-5f) continue;
        by_alpha_arch[row.alpha][row.architecture].add(row);
    }

    std::ofstream out(path);
    out << "alpha mse_dense mse_conv1d mse_recurrent energy_dense energy_conv1d energy_recurrent\n";
    out << std::fixed << std::setprecision(4);

    for (const auto& [alpha, arch_map] : by_alpha_arch)
    {
        auto get_mse = [&](const std::string& arch)
        {
            const auto it = arch_map.find(arch);
            if (it == arch_map.end()) return 0.0;
            return avg(it->second, [](const Aggregated& a) { return a.mse; });
        };
        auto get_energy = [&](const std::string& arch)
        {
            const auto it = arch_map.find(arch);
            if (it == arch_map.end()) return 0.0;
            return avg(it->second, [](const Aggregated& a) { return a.energy; });
        };

        out << alpha << ' ' << get_mse("dense") << ' ' << get_mse("conv1d") << ' '
            << get_mse("recurrent") << ' ' << get_energy("dense") << ' ' << get_energy("conv1d")
            << ' ' << get_energy("recurrent") << '\n';
    }
}

void write_epoch_history_dat(const std::filesystem::path& path,
    const std::string& model,
    const std::string& encoding,
    const std::string& architecture,
    float v_th,
    float alpha,
    int run_id,
    const EpochHistory& history)
{
    std::ofstream out(path);
    out << "# model=" << model << " encoding=" << encoding << " architecture=" << architecture
        << " v_th=" << v_th << " alpha=" << alpha << " run=" << run_id << '\n';
    out << "epoch train_loss val_loss\n";
    out << std::fixed << std::setprecision(6);

    for (std::size_t i = 0; i < history.epoch_nums.size(); ++i)
    {
        out << static_cast<int>(history.epoch_nums[i]) << ' ' << history.train_losses[i] << ' '
            << history.val_losses[i] << '\n';
    }
}

void write_batch_convergence_dat(const std::filesystem::path& path,
    const std::string& model,
    const std::string& encoding,
    const std::string& architecture,
    float v_th,
    float alpha,
    int run_id,
    const EpochHistory& history)
{
    std::ofstream out(path);
    out << "# Batch convergence for: model=" << model << " encoding=" << encoding
        << " architecture=" << architecture << " v_th=" << v_th << " alpha=" << alpha
        << " run=" << run_id << '\n';
    out << "batch_num epoch batch_loss\n";
    out << std::fixed << std::setprecision(6);

    for (std::size_t i = 0; i < history.batch_losses.size(); ++i)
    {
        out << i << ' ' << static_cast<int>(history.batch_epochs[i]) << ' '
            << history.batch_losses[i] << '\n';
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
            if (!almost_eq(ref.spike_rate, cur.spike_rate) &&
                !rel_eq(ref.spike_rate, cur.spike_rate))
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
