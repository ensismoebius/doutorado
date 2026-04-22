/**
 * @file src/experiments/04/experiment04.cpp
 * @brief Standalone Experiment04 entrypoint, CLI normalization, and comparative runner.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "nlohmann/json.hpp"
#include "nn/io/ReportIO.hpp"
#include "nn/layers/losses/MSELoss.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/logging/StreamRedirector.hpp"
#include "nn/models/lstm/LSTMAutoencoder.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/statistics/inference_tests.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/utility/SignalPreprocessing.hpp"

namespace lstm_autoencoder_experiment
{
auto should_run_from_cli(int argc, char* argv[]) -> bool;
auto should_run_comparative_from_cli(int argc, char* argv[]) -> bool;
auto run_comparative_experiment(int argc, char* argv[]) -> int;
} // namespace lstm_autoencoder_experiment

class LstmAutoencoderExperiment
{
   public:
    auto run(int argc, char* argv[]) -> int;
};

using nn::logging::Level;
using nn::logging::Logger;
using nn::logging::StreamRedirector;

#ifndef NN_EXPERIMENT04_NO_MAIN
namespace
{
auto parse_log_level_from_env() -> Level
{
    const char* value = std::getenv("NN_EXPERIMENT04_LOG_LEVEL");
    if (value == nullptr) return Level::Info;

    const std::string_view level{value};
    if (level == "error") return Level::Error;
    if (level == "warn" || level == "warning") return Level::Warn;
    if (level == "debug") return Level::Debug;
    return Level::Info;
}

auto has_help_flag(int argc, char* argv[]) -> bool
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i] ? argv[i] : "";
        if (arg == "-h" || arg == "--help" || arg == "--help-all")
        {
            return true;
        }
    }

    return false;
}
} // namespace
#endif

namespace lstm_autoencoder_experiment
{

auto has_experiment04_marker(const std::string& arg) -> bool
{
    return arg == "--experiment04" || arg == "--lstm-autoencoder" || arg == "--experiment=04" ||
           arg == "--experiment=experiment04" || arg == "--experiment=lstm" ||
           arg == "--experiment=lstm-autoencoder";
}

void normalize_experiment04_aliases(int argc, char* argv[], std::vector<std::string>& args)
{
    args.clear();
    args.reserve(static_cast<std::size_t>(argc));

    if (argc > 0)
    {
        args.emplace_back(argv[0] ? argv[0] : "experiment04");
    }
    else
    {
        args.emplace_back("experiment04");
    }

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";

        if (has_experiment04_marker(arg))
        {
            args.emplace_back("--comparative");
            continue;
        }

        if (arg == "--lstm-profile" || arg == "--config")
        {
            args.emplace_back("--comparative-config");
            if (i + 1 < argc)
            {
                args.emplace_back(argv[++i] ? argv[i] : "");
            }
            continue;
        }

        if (arg.rfind("--lstm-profile=", 0) == 0)
        {
            const std::string value = arg.substr(std::string("--lstm-profile=").size());
            args.emplace_back("--comparative-config=" + value);
            continue;
        }

        args.push_back(arg);
    }
}

void to_argv(std::vector<std::string>& args, std::vector<char*>& argv_out)
{
    argv_out.clear();
    argv_out.reserve(args.size());
    for (std::string& arg : args)
    {
        argv_out.push_back(arg.data());
    }
}

auto should_run_from_cli(int argc, char* argv[]) -> bool
{
    if (should_run_comparative_from_cli(argc, argv))
    {
        return true;
    }

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (has_experiment04_marker(arg) || arg == "--lstm-profile" ||
            arg.rfind("--lstm-profile=", 0) == 0)
        {
            return true;
        }
    }

    return false;
}

} // namespace lstm_autoencoder_experiment

auto LstmAutoencoderExperiment::run(int argc, char* argv[]) -> int
{
    std::vector<char*> normalized_argv;
    std::vector<std::string> normalized_args;

    lstm_autoencoder_experiment::normalize_experiment04_aliases(argc, argv, normalized_args);
    lstm_autoencoder_experiment::to_argv(normalized_args, normalized_argv);

    return lstm_autoencoder_experiment::run_comparative_experiment(
        static_cast<int>(normalized_argv.size()), normalized_argv.data() //
    );
}

namespace comparative_autoencoder_experiment
{
using Tensor = nn::Tensor;

struct ComparativeConfig
{
    std::string dataset_root = ".";
    std::string results_dir = "results";
    std::string run_tag = "snn_lstm_compare";

    std::uint32_t seed = 1337u;
    int repeats = 3;

    int window_size = 128;
    int batch_size = 8;
    int max_train_samples = 512;
    int max_val_samples = 128;

    int epochs = 100;
    int early_stop_patience = 20;
    float learning_rate = 1e-3f;
    float anomaly_tau = 0.25f;

    int hidden_size = 64;
    int latent_size = 16;

    std::vector<std::string> datasets = {"fsdd", "physionet"};
    std::vector<std::string> encodings = {"direct", "poisson", "latency"};
    std::vector<std::string> snn_architectures = {"dense", "conv1d", "recurrent"};
    std::vector<int> layers = {1, 2, 3};
    std::vector<float> v_th_values = {0.5f, 1.0f, 1.5f};
    std::vector<float> alpha_values = {0.8f, 0.9f, 0.99f};
};

struct RunMetrics
{
    float mse = 0.0f;
    float mae = 0.0f;
    float r2 = 0.0f;

    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;

    float spike_rate = 0.0f;
    float energy = 0.0f;

    float train_ms = 0.0f;
    float infer_ms = 0.0f;

    std::size_t parameter_count = 0;
    std::size_t macs = 0;
};

struct ResultRow
{
    std::string dataset;
    std::string model;
    std::string encoding;
    std::string architecture;
    int layers = 1;
    float v_th = 1.0f;
    float alpha = 0.9f;
    int run_id = 0;

    std::uint32_t seed = 0u;
    std::size_t config_hash = 0u;

    RunMetrics metrics;
};

struct DatasetSplit
{
    std::vector<Tensor> train_samples;
    std::vector<Tensor> val_samples;
    std::vector<int> val_labels;
};

struct CliOptions
{
    std::string comparative_config;
    bool help = false;
};

constexpr const char* kDefaultComparativeProfileStem = "lstm-compare";

static_assert(sizeof(float) == 4, "Experiment requires 32-bit float.");

auto has_compare_marker(const std::string& arg) -> bool
{
    return arg == "--comparative" || arg == "--experiment=snn-lstm-compare" ||
           arg == "--experiment=compare" || arg == "--experiment=comparative";
}

auto source_profile_dir() -> std::filesystem::path
{
    return std::filesystem::path(__FILE__).parent_path() / "profiles";
}

auto source_results_dir() -> std::filesystem::path
{
    return std::filesystem::path(__FILE__).parent_path() / "results";
}

void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " --experiment=snn-lstm-compare [options]\n"
              << "Options:\n"
              << "  --comparative                     Run SNN-vs-LSTM comparative experiment\n"
              << "  --comparative-config <name|path>  Comparative profile stem or JSON path\n"
              << "  --help                            Print this message\n";
}

auto parse_cli(int argc, char* argv[]) -> CliOptions
{
    CliOptions opts;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            opts.help = true;
        }
        else if (has_compare_marker(arg))
        {
            continue;
        }
        else if (arg == "--comparative-config")
        {
            opts.comparative_config = next();
        }
        else if (arg.rfind("--comparative-config=", 0) == 0)
        {
            opts.comparative_config = arg.substr(std::string("--comparative-config=").size());
        }
    }

    return opts;
}

auto resolve_profile_path(const CliOptions& opts) -> std::filesystem::path
{
    namespace fs = std::filesystem;

    const fs::path source_dir = source_profile_dir();
    const fs::path runtime_dir = fs::path("profiles");

    const std::string profile_name = opts.comparative_config.empty()
                                         ? std::string(kDefaultComparativeProfileStem)
                                         : opts.comparative_config;
    const fs::path raw = fs::path(profile_name);

    if ((raw.has_parent_path() || raw.extension() == ".json") && fs::exists(raw))
    {
        return raw;
    }

    const std::string stem = raw.stem().string().empty()
                                 ? std::string(kDefaultComparativeProfileStem)
                                 : raw.stem().string();
    const fs::path source_candidate = source_dir / (stem + ".json");
    if (fs::exists(source_candidate)) return source_candidate;

    const fs::path runtime_candidate = runtime_dir / (stem + ".json");
    if (fs::exists(runtime_candidate)) return runtime_candidate;

    throw std::runtime_error("Cannot resolve comparative profile: " + profile_name);
}

auto load_config(const std::filesystem::path& path) -> ComparativeConfig
{
    ComparativeConfig cfg;

    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open comparative config: " + path.string());
    }

    nlohmann::json j;
    f >> j;

    auto get = [&](const std::string& key, auto& field)
    {
        if (j.contains(key)) field = j[key].get<std::decay_t<decltype(field)>>();
    };

    get("dataset_root", cfg.dataset_root);
    get("results_dir", cfg.results_dir);
    get("run_tag", cfg.run_tag);
    get("seed", cfg.seed);
    get("repeats", cfg.repeats);
    get("window_size", cfg.window_size);
    get("batch_size", cfg.batch_size);
    get("max_train_samples", cfg.max_train_samples);
    get("max_val_samples", cfg.max_val_samples);
    get("epochs", cfg.epochs);
    get("early_stop_patience", cfg.early_stop_patience);
    get("learning_rate", cfg.learning_rate);
    get("anomaly_tau", cfg.anomaly_tau);
    get("hidden_size", cfg.hidden_size);
    get("latent_size", cfg.latent_size);
    get("datasets", cfg.datasets);
    get("encodings", cfg.encodings);
    get("snn_architectures", cfg.snn_architectures);
    get("layers", cfg.layers);
    get("v_th_values", cfg.v_th_values);
    get("alpha_values", cfg.alpha_values);

    return cfg;
}

auto config_hash(const ComparativeConfig& cfg) -> std::size_t
{
    nlohmann::json j;
    j["dataset_root"] = cfg.dataset_root;
    j["results_dir"] = cfg.results_dir;
    j["run_tag"] = cfg.run_tag;
    j["seed"] = cfg.seed;
    j["repeats"] = cfg.repeats;
    j["window_size"] = cfg.window_size;
    j["batch_size"] = cfg.batch_size;
    j["max_train_samples"] = cfg.max_train_samples;
    j["max_val_samples"] = cfg.max_val_samples;
    j["epochs"] = cfg.epochs;
    j["early_stop_patience"] = cfg.early_stop_patience;
    j["learning_rate"] = cfg.learning_rate;
    j["anomaly_tau"] = cfg.anomaly_tau;
    j["hidden_size"] = cfg.hidden_size;
    j["latent_size"] = cfg.latent_size;
    j["datasets"] = cfg.datasets;
    j["encodings"] = cfg.encodings;
    j["snn_architectures"] = cfg.snn_architectures;
    j["layers"] = cfg.layers;
    j["v_th_values"] = cfg.v_th_values;
    j["alpha_values"] = cfg.alpha_values;
    return std::hash<std::string>{}(j.dump());
}

auto to_window_tensor(const Tensor& signal, int window_size) -> std::vector<Tensor>
{
    std::vector<Tensor> windows;
    if (window_size <= 0 || signal.size() == 0) return windows;

    std::size_t offset = 0;
    const std::size_t signal_len = static_cast<std::size_t>(signal.rows());
    while (offset < signal_len)
    {
        const std::size_t remaining = signal_len - offset;
        const std::size_t take =
            std::min<std::size_t>(remaining, static_cast<std::size_t>(window_size));

        Tensor sample(window_size, 1);
        sample.set_zero();
        for (int t = 0; t < window_size; ++t)
        {
            if (static_cast<std::size_t>(t) < take)
            {
                sample.at(static_cast<nn::Index>(t), 0) =
                    signal.at(static_cast<nn::Index>(offset + static_cast<std::size_t>(t)), 0);
            }
        }

        nn::utility::zscore_inplace(sample);
        windows.push_back(std::move(sample));
        offset += static_cast<std::size_t>(window_size);
    }

    return windows;
}

auto collect_signal_files(const ComparativeConfig& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(cfg.dataset_root);
    if (!fs::exists(root))
    {
        throw std::runtime_error("Dataset root does not exist: " + root.string());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file()) continue;
        const std::string path_str = entry.path().string();
        const std::string ext = entry.path().extension().string();

        if (dataset == "fsdd")
        {
            if ((ext == ".csv" || ext == ".txt") && path_str.find("FSDD") != std::string::npos)
            {
                files.push_back(entry.path());
            }
        }
        else if (dataset == "physionet")
        {
            if ((ext == ".csv" || ext == ".txt") &&
                (path_str.find("physionet") != std::string::npos ||
                    path_str.find("PhysioNet") != std::string::npos))
            {
                files.push_back(entry.path());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

auto build_split(const ComparativeConfig& cfg, const std::string& dataset) -> DatasetSplit
{
    DatasetSplit split;
    const auto files = collect_signal_files(cfg, dataset);
    if (files.empty())
    {
        throw std::runtime_error("No files found for dataset token: " + dataset);
    }

    std::vector<Tensor> all_samples;
    for (const auto& file : files)
    {
        const Tensor signal = nn::utility::read_csv_signal(file);
        const auto windows = to_window_tensor(signal, cfg.window_size);
        all_samples.insert(all_samples.end(), windows.begin(), windows.end());
    }

    if (all_samples.empty())
    {
        throw std::runtime_error("No windows created for dataset token: " + dataset);
    }

    const std::size_t max_total =
        static_cast<std::size_t>(cfg.max_train_samples + cfg.max_val_samples);
    if (all_samples.size() > max_total)
    {
        all_samples.resize(max_total);
    }

    const std::size_t val_count =
        std::min<std::size_t>(cfg.max_val_samples, all_samples.size() / 5);
    const std::size_t train_count = all_samples.size() - val_count;

    split.train_samples.assign(
        all_samples.begin(), all_samples.begin() + static_cast<long>(train_count));
    split.val_samples.assign(
        all_samples.begin() + static_cast<long>(train_count), all_samples.end());
    split.val_labels.assign(split.val_samples.size(), 0);

    for (std::size_t i = 0; i < split.val_samples.size(); ++i)
    {
        if (i % 10 != 0) continue;
        split.val_labels[i] = 1;
        Tensor& sample = split.val_samples[i];
        for (nn::Index t = 0; t < sample.rows(); ++t)
        {
            sample.at(t, 0) += 1.5f;
        }
    }

    return split;
}

auto encode_sample(const Tensor& sample, const std::string& encoding, std::uint32_t seed) -> Tensor
{
    Tensor encoded(sample.rows(), sample.cols());
    encoded.set_zero();

    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (nn::Index i = 0; i < sample.size(); ++i)
    {
        min_v = std::min(min_v, sample.at(i));
        max_v = std::max(max_v, sample.at(i));
    }
    const float range = std::max(max_v - min_v, 1e-6f);

    if (encoding == "direct")
    {
        return sample;
    }

    if (encoding == "poisson")
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        const float max_only = std::max(max_v, 1e-6f);
        for (nn::Index t = 0; t < sample.rows(); ++t)
        {
            for (nn::Index d = 0; d < sample.cols(); ++d)
            {
                const float p = std::clamp(sample.at(t, d) / max_only, 0.0f, 1.0f);
                encoded.at(t, d) = (dist(rng) < p) ? 1.0f : 0.0f;
            }
        }
        return encoded;
    }

    if (encoding == "latency")
    {
        const nn::Index T = sample.rows();
        for (nn::Index t = 0; t < sample.rows(); ++t)
        {
            for (nn::Index d = 0; d < sample.cols(); ++d)
            {
                const float scaled = (sample.at(t, d) - min_v) / range;
                const nn::Index t_spike = static_cast<nn::Index>(std::llround(
                    (1.0f - scaled) * static_cast<float>(std::max<nn::Index>(1, T - 1))));
                encoded.at(t, d) = (t >= t_spike) ? 1.0f : 0.0f;
            }
        }
        return encoded;
    }

    throw std::invalid_argument("Unsupported encoding token: " + encoding);
}

auto flatten_time_series(const Tensor& sample) -> Tensor
{
    Tensor flat(1, sample.rows() * sample.cols());
    nn::Index k = 0;
    for (nn::Index t = 0; t < sample.rows(); ++t)
    {
        for (nn::Index d = 0; d < sample.cols(); ++d)
        {
            flat.at(0, k++) = sample.at(t, d);
        }
    }
    return flat;
}

auto unflatten_time_series(const Tensor& flat, nn::Index rows, nn::Index cols) -> Tensor
{
    Tensor sample(rows, cols);
    nn::Index k = 0;
    for (nn::Index t = 0; t < rows; ++t)
    {
        for (nn::Index d = 0; d < cols; ++d)
        {
            sample.at(t, d) = flat.at(0, k++);
        }
    }
    return sample;
}

auto conv1d_temporal_smooth(const Tensor& sample) -> Tensor
{
    Tensor out(sample.rows(), sample.cols());
    out.set_zero();

    for (nn::Index t = 0; t < sample.rows(); ++t)
    {
        for (nn::Index d = 0; d < sample.cols(); ++d)
        {
            const nn::Index t_prev = std::max<nn::Index>(0, t - 1);
            const nn::Index t_next = std::min<nn::Index>(sample.rows() - 1, t + 1);
            const float v = 0.25f * sample.at(t_prev, d) + 0.5f * sample.at(t, d) +
                            0.25f * sample.at(t_next, d);
            out.at(t, d) = v;
        }
    }

    return out;
}

auto recurrent_lif_encode(const Tensor& sample, float alpha, float v_th) -> Tensor
{
    Tensor spikes(sample.rows(), sample.cols());
    spikes.set_zero();

    Tensor v_prev(1, sample.cols());
    v_prev.set_zero();
    Tensor s_prev(1, sample.cols());
    s_prev.set_zero();

    const float stable_alpha = std::clamp(alpha, 0.0f, 0.9999f);
    const float stable_vth = std::max(v_th, 1e-4f);

    for (nn::Index t = 0; t < sample.rows(); ++t)
    {
        Tensor x_t = sample.row(t);
        Tensor v_t = (v_prev * stable_alpha) + x_t - (s_prev * stable_vth);
        Tensor s_t(1, sample.cols());
        for (nn::Index d = 0; d < sample.cols(); ++d)
        {
            s_t.at(0, d) = v_t.at(0, d) >= stable_vth ? 1.0f : 0.0f;
            spikes.at(t, d) = s_t.at(0, d);
        }
        v_prev = v_t;
        s_prev = s_t;
    }

    return spikes;
}

auto apply_snn_architecture_transform(
    const Tensor& encoded, const std::string& architecture, float alpha, float v_th) -> Tensor
{
    if (architecture == "conv1d") return conv1d_temporal_smooth(encoded);
    if (architecture == "recurrent") return recurrent_lif_encode(encoded, alpha, v_th);
    return encoded;
}

auto mse_between(const Tensor& a, const Tensor& b) -> float
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        throw std::invalid_argument("mse_between shape mismatch");
    }
    float sum = 0.0f;
    const nn::Index n = a.size();
    for (nn::Index i = 0; i < n; ++i)
    {
        const float d = a.at(i) - b.at(i);
        sum += d * d;
    }
    return (n > 0) ? (sum / static_cast<float>(n)) : 0.0f;
}

auto mae_between(const Tensor& a, const Tensor& b) -> float
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        throw std::invalid_argument("mae_between shape mismatch");
    }
    float sum = 0.0f;
    const nn::Index n = a.size();
    for (nn::Index i = 0; i < n; ++i)
    {
        sum += std::fabs(a.at(i) - b.at(i));
    }
    return (n > 0) ? (sum / static_cast<float>(n)) : 0.0f;
}

void compute_precision_recall_f1(const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    float& precision,
    float& recall,
    float& f1)
{
    int tp = 0;
    int fp = 0;
    int fn = 0;
    for (std::size_t i = 0; i < y_true.size() && i < y_pred.size(); ++i)
    {
        if (y_true[i] == 1 && y_pred[i] == 1) ++tp;
        if (y_true[i] == 0 && y_pred[i] == 1) ++fp;
        if (y_true[i] == 1 && y_pred[i] == 0) ++fn;
    }

    precision = (tp + fp) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    recall = (tp + fn) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    f1 = (precision + recall) > 0.0f ? (2.0f * precision * recall) / (precision + recall) : 0.0f;
}

auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t
{
    const std::size_t T = static_cast<std::size_t>(cfg.seq_len);
    const std::size_t I = static_cast<std::size_t>(cfg.input_size);
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t L = static_cast<std::size_t>(cfg.num_layers);

    const std::size_t per_gate = H * (I + H);
    const std::size_t per_step = 4 * per_gate;
    const std::size_t per_stack = per_step * L;
    const std::size_t proj = H * static_cast<std::size_t>(cfg.latent_size) +
                             static_cast<std::size_t>(cfg.latent_size) * H + H * I;
    return T * per_stack + proj;
}

auto estimate_snn_macs(std::size_t input_features, int hidden_size, int layers) -> std::size_t
{
    const std::size_t H = static_cast<std::size_t>(hidden_size);
    const std::size_t L = static_cast<std::size_t>(std::max(1, layers));
    const std::size_t in_proj = input_features * H;
    const std::size_t hidden_proj = (L > 1) ? (L - 1) * H * H : 0;
    const std::size_t out_proj = H * input_features;
    return in_proj + hidden_proj + out_proj;
}

auto parameter_count(std::span<nn::Tensor*> params) -> std::size_t
{
    std::size_t count = 0;
    for (nn::Tensor* p : params)
    {
        if (!p) continue;
        count += static_cast<std::size_t>(p->size());
    }
    return count;
}

auto evaluate_lstm(nn::models::lstm::LSTMAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float anomaly_tau,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    std::uint32_t seed,
    float infer_ms) -> RunMetrics
{
    RunMetrics m;
    m.macs = macs;
    m.parameter_count = param_count;
    m.infer_ms = infer_ms;

    std::vector<int> pred_labels;
    pred_labels.reserve(val_samples.size());

    float mse_acc = 0.0f;
    float mae_acc = 0.0f;
    float y_mean_acc = 0.0f;
    std::size_t n_values = 0;

    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        const Tensor recon = model.forward(encoded, false);

        mse_acc += mse_between(encoded, recon);
        mae_acc += mae_between(encoded, recon);

        float sample_residual_mean = 0.0f;
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            sample_residual_mean += std::fabs(encoded.at(k) - recon.at(k));
            y_mean_acc += encoded.at(k);
            ++n_values;
        }
        sample_residual_mean /= static_cast<float>(std::max<nn::Index>(1, encoded.size()));
        pred_labels.push_back(sample_residual_mean > anomaly_tau ? 1 : 0);
    }

    m.mse = val_samples.empty() ? 0.0f : mse_acc / static_cast<float>(val_samples.size());
    m.mae = val_samples.empty() ? 0.0f : mae_acc / static_cast<float>(val_samples.size());

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    const float y_mean = (n_values > 0) ? y_mean_acc / static_cast<float>(n_values) : 0.0f;
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        const Tensor recon = model.forward(encoded, false);
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            const float y = encoded.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    compute_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

    m.spike_rate = 0.0f;
    m.energy = 10.0f * static_cast<float>(m.macs);

    return m;
}

auto evaluate_snn(ProtocolSpikingAutoencoder& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float anomaly_tau,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed,
    float infer_ms) -> RunMetrics
{
    RunMetrics m;
    m.macs = macs;
    m.parameter_count = param_count;
    m.infer_ms = infer_ms;

    std::vector<int> pred_labels;
    pred_labels.reserve(val_samples.size());

    float mse_acc = 0.0f;
    float mae_acc = 0.0f;
    float y_mean_acc = 0.0f;
    std::size_t n_values = 0;
    float spike_sum = 0.0f;

    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);

        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        const Tensor recon_flat = model.forward(flat, false);
        const Tensor recon = unflatten_time_series(recon_flat, encoded.rows(), encoded.cols());

        mse_acc += mse_between(encoded, recon);
        mae_acc += mae_between(encoded, recon);

        float sample_residual_mean = 0.0f;
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            sample_residual_mean += std::fabs(encoded.at(k) - recon.at(k));
            y_mean_acc += encoded.at(k);
            spike_sum += recon.at(k) > 0.0f ? 1.0f : 0.0f;
            ++n_values;
        }
        sample_residual_mean /= static_cast<float>(std::max<nn::Index>(1, encoded.size()));
        pred_labels.push_back(sample_residual_mean > anomaly_tau ? 1 : 0);
    }

    m.mse = val_samples.empty() ? 0.0f : mse_acc / static_cast<float>(val_samples.size());
    m.mae = val_samples.empty() ? 0.0f : mae_acc / static_cast<float>(val_samples.size());

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    const float y_mean = (n_values > 0) ? y_mean_acc / static_cast<float>(n_values) : 0.0f;
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        const Tensor recon =
            unflatten_time_series(model.forward(flat, false), encoded.rows(), encoded.cols());
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            const float y = encoded.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    compute_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

    m.spike_rate = (n_values > 0) ? spike_sum / static_cast<float>(n_values) : 0.0f;
    m.energy = m.spike_rate * static_cast<float>(n_values) + 10.0f * static_cast<float>(m.macs);

    return m;
}

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
    j["seed"] = cfg.seed;
    j["config_hash"] = cfg_hash;
    j["epochs"] = cfg.epochs;
    j["batch_size"] = cfg.batch_size;
    j["window_size"] = cfg.window_size;
    j["repeats"] = cfg.repeats;
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

auto make_lstm_cfg(const ComparativeConfig& cfg) -> nn::models::lstm::LSTMAutoencoderConfig
{
    nn::models::lstm::LSTMAutoencoderConfig arch;
    arch.input_size = 1;
    arch.seq_len = cfg.window_size;
    arch.hidden_size = cfg.hidden_size;
    arch.latent_size = cfg.latent_size;
    arch.num_layers = 1;
    return arch;
}

auto make_snn_cfg(const ComparativeConfig& cfg, int layers, float alpha, float v_th)
    -> AutoencoderConfig
{
    AutoencoderConfig model_cfg;
    model_cfg.loss_type = "mse";
    model_cfg.input_features = cfg.window_size;
    model_cfg.hidden_size = cfg.hidden_size;
    model_cfg.latent_size = cfg.latent_size;
    model_cfg.depth = std::max(1, layers);
    model_cfg.time_step = 1.0f;
    model_cfg.resistance = 1.0f / std::max(v_th, 1e-3f);
    model_cfg.capacitance = std::max(1e-3f, -1.0f / std::log(std::max(alpha, 1e-3f)));
    model_cfg.encoder_layer_spec = {"linear:hidden:leaky", "linear:latent:identity"};
    model_cfg.decoder_layer_spec = {"linear:hidden:leaky", "linear:output:identity"};
    return model_cfg;
}

void train_lstm_once(nn::models::lstm::LSTMAutoencoder& model,
    Adam& optimizer,
    const std::vector<Tensor>& train_samples,
    const std::string& encoding,
    std::uint32_t seed)
{
    MSELossImpl<nn::EigenTensorBackend> mse_loss;
    for (std::size_t i = 0; i < train_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(train_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        optimizer.zero_grad(model.params());

        const Tensor recon = model.forward(encoded, true);
        mse_loss.set_target(encoded);
        const Tensor loss = mse_loss.forward(recon, true);
        (void) loss;
        const Tensor grad = mse_loss.backward(recon);
        model.backward(grad);
        optimizer.step(model.params());
    }
}

void train_snn_once(ProtocolSpikingAutoencoder& model,
    Adam& optimizer,
    const std::vector<Tensor>& train_samples,
    const std::string& encoding,
    const std::string& architecture,
    float alpha,
    float v_th,
    std::uint32_t seed)
{
    MSELossImpl<nn::EigenTensorBackend> mse_loss;
    for (std::size_t i = 0; i < train_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(train_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);

        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        optimizer.zero_grad(model.params());

        const Tensor recon_flat = model.forward(flat, true);
        mse_loss.set_target(flat);
        const Tensor loss = mse_loss.forward(recon_flat, true);
        (void) loss;
        const Tensor grad = mse_loss.backward(recon_flat);
        model.backward(grad);
        optimizer.step(model.params());
    }
}

auto train_with_early_stopping_lstm(nn::models::lstm::LSTMAutoencoder& model,
    Adam& optimizer,
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    float& train_ms,
    float& infer_ms) -> RunMetrics
{
    auto best = std::numeric_limits<float>::infinity();
    int bad_epochs = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        train_lstm_once(
            model, optimizer, train_samples, encoding, seed + static_cast<std::uint32_t>(epoch));

        float val_mse = 0.0f;
        MSELossImpl<nn::EigenTensorBackend> mse_loss;
        for (std::size_t i = 0; i < val_samples.size(); ++i)
        {
            const Tensor encoded =
                encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
            model.reset_state();
            const Tensor recon = model.forward(encoded, false);
            mse_loss.set_target(encoded);
            val_mse += mse_loss.forward(recon, false).at(0, 0);
        }
        if (!val_samples.empty()) val_mse /= static_cast<float>(val_samples.size());

        if (val_mse + 1e-8f < best)
        {
            best = val_mse;
            bad_epochs = 0;
        }
        else
        {
            ++bad_epochs;
            if (bad_epochs >= cfg.early_stop_patience) break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        model.reset_state();
        (void) model.forward(encoded, false);
    }
    const auto infer_end = std::chrono::steady_clock::now();
    infer_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

    return evaluate_lstm(model,
        val_samples,
        std::vector<int>(val_samples.size(), 0),
        cfg.anomaly_tau,
        estimate_lstm_macs(make_lstm_cfg(cfg)),
        parameter_count(model.params()),
        encoding,
        seed,
        infer_ms);
}

auto train_with_early_stopping_snn(ProtocolSpikingAutoencoder& model,
    Adam& optimizer,
    const ComparativeConfig& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    const std::string& encoding,
    const std::string& architecture,
    int layers,
    float alpha,
    float v_th,
    std::uint32_t seed,
    float& train_ms,
    float& infer_ms) -> RunMetrics
{
    auto best = std::numeric_limits<float>::infinity();
    int bad_epochs = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        train_snn_once(model,
            optimizer,
            train_samples,
            encoding,
            architecture,
            alpha,
            v_th,
            seed + static_cast<std::uint32_t>(epoch));

        float val_mse = 0.0f;
        MSELossImpl<nn::EigenTensorBackend> mse_loss;
        for (std::size_t i = 0; i < val_samples.size(); ++i)
        {
            Tensor encoded =
                encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
            encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
            const Tensor flat = flatten_time_series(encoded);
            model.reset_state();
            const Tensor recon = model.forward(flat, false);
            mse_loss.set_target(flat);
            val_mse += mse_loss.forward(recon, false).at(0, 0);
        }
        if (!val_samples.empty()) val_mse /= static_cast<float>(val_samples.size());

        if (val_mse + 1e-8f < best)
        {
            best = val_mse;
            bad_epochs = 0;
        }
        else
        {
            ++bad_epochs;
            if (bad_epochs >= cfg.early_stop_patience) break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        Tensor encoded =
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i));
        encoded = apply_snn_architecture_transform(encoded, architecture, alpha, v_th);
        const Tensor flat = flatten_time_series(encoded);
        model.reset_state();
        (void) model.forward(flat, false);
    }
    const auto infer_end = std::chrono::steady_clock::now();
    infer_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

    return evaluate_snn(model,
        val_samples,
        val_labels,
        cfg.anomaly_tau,
        estimate_snn_macs(static_cast<std::size_t>(cfg.window_size), cfg.hidden_size, layers),
        parameter_count(model.params()),
        encoding,
        architecture,
        alpha,
        v_th,
        seed,
        infer_ms);
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
    if (cfg.repeats <= 1) return;

    std::map<std::string, std::vector<const ResultRow*>> groups;
    for (const auto& row : rows)
    {
        const std::string key = row.dataset + "|" + row.model + "|" + row.encoding + "|" +
                                row.architecture + "|" + std::to_string(row.layers) + "|" +
                                std::to_string(row.v_th) + "|" + std::to_string(row.alpha);
        groups[key].push_back(&row);
    }

    auto almost_eq = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };

    for (const auto& [key, group] : groups)
    {
        if (static_cast<int>(group.size()) != cfg.repeats)
        {
            throw std::runtime_error("Determinism check failed (missing repeats) for key: " + key);
        }

        const RunMetrics& ref = group.front()->metrics;
        for (std::size_t i = 1; i < group.size(); ++i)
        {
            const RunMetrics& cur = group[i]->metrics;
            if (!almost_eq(ref.mse, cur.mse) || !almost_eq(ref.mae, cur.mae) ||
                !almost_eq(ref.r2, cur.r2) || !almost_eq(ref.f1, cur.f1) ||
                !almost_eq(ref.spike_rate, cur.spike_rate) || !almost_eq(ref.energy, cur.energy))
            {
                throw std::runtime_error(
                    "Determinism check failed (metrics differ) for key: " + key);
            }
        }
    }
}

auto should_run_comparative_cli(int argc, char* argv[]) -> bool
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (has_compare_marker(arg) || arg == "--comparative-config" ||
            arg.rfind("--comparative-config=", 0) == 0)
        {
            return true;
        }
    }
    return false;
}

} // namespace comparative_autoencoder_experiment

namespace lstm_autoencoder_experiment
{

auto should_run_comparative_from_cli(int argc, char* argv[]) -> bool
{
    return comparative_autoencoder_experiment::should_run_comparative_cli(argc, argv);
}

auto run_comparative_experiment(int argc, char* argv[]) -> int
{
    using namespace comparative_autoencoder_experiment;

    try
    {
        const CliOptions cli = parse_cli(argc, argv);
        if (cli.help)
        {
            print_usage(argv[0]);
            return 0;
        }

        const ComparativeConfig cfg = load_config(resolve_profile_path(cli));
        const std::size_t cfg_hash = config_hash(cfg);

        std::filesystem::path out_dir =
            cfg.results_dir.empty() ? source_results_dir() : std::filesystem::path(cfg.results_dir);
        if (!std::filesystem::exists(out_dir)) out_dir = std::filesystem::path("results");
        std::filesystem::create_directories(out_dir);

        std::vector<ResultRow> all_rows;

        for (const auto& dataset : cfg.datasets)
        {
            const DatasetSplit split = build_split(cfg, dataset);

            for (const auto& encoding : cfg.encodings)
            {
                for (int run_id = 0; run_id < cfg.repeats; ++run_id)
                {
                    const std::uint32_t run_seed = cfg.seed;

                    {
                        auto lstm_cfg = make_lstm_cfg(cfg);
                        nn::models::lstm::LSTMAutoencoder lstm_model(lstm_cfg);
                        Adam lstm_opt(cfg.learning_rate);
                        lstm_opt.attach(lstm_model.params());

                        float train_ms = 0.0f;
                        float infer_ms = 0.0f;
                        RunMetrics metrics = train_with_early_stopping_lstm(lstm_model,
                            lstm_opt,
                            cfg,
                            split.train_samples,
                            split.val_samples,
                            encoding,
                            run_seed,
                            train_ms,
                            infer_ms);
                        metrics.train_ms = train_ms;

                        all_rows.push_back(ResultRow{dataset,
                            "lstm-ae",
                            encoding,
                            "lstm",
                            1,
                            0.0f,
                            0.0f,
                            run_id + 1,
                            run_seed,
                            cfg_hash,
                            metrics});
                    }

                    for (const auto& architecture : cfg.snn_architectures)
                    {
                        for (int layers : cfg.layers)
                        {
                            for (float v_th : cfg.v_th_values)
                            {
                                for (float alpha : cfg.alpha_values)
                                {
                                    AutoencoderConfig snn_cfg =
                                        make_snn_cfg(cfg, layers, alpha, v_th);

                                    ProtocolSpikingAutoencoder snn_model(snn_cfg);
                                    Adam snn_opt(cfg.learning_rate);
                                    snn_opt.attach(snn_model.params());

                                    float train_ms = 0.0f;
                                    float infer_ms = 0.0f;
                                    RunMetrics metrics = train_with_early_stopping_snn(snn_model,
                                        snn_opt,
                                        cfg,
                                        split.train_samples,
                                        split.val_samples,
                                        split.val_labels,
                                        encoding,
                                        architecture,
                                        layers,
                                        alpha,
                                        v_th,
                                        run_seed,
                                        train_ms,
                                        infer_ms);
                                    metrics.train_ms = train_ms;

                                    all_rows.push_back(ResultRow{dataset,
                                        "snn-ae",
                                        encoding,
                                        architecture,
                                        layers,
                                        v_th,
                                        alpha,
                                        run_id + 1,
                                        run_seed,
                                        cfg_hash,
                                        metrics});
                                }
                            }
                        }
                    }
                }
            }
        }

        validate_repeat_determinism(cfg, all_rows);

        const std::filesystem::path csv_path = out_dir / (cfg.run_tag + "_comparative_metrics.csv");
        write_rows_csv(csv_path, all_rows);

        const std::filesystem::path table_path = out_dir / (cfg.run_tag + "_publication_table.csv");
        write_publication_table(table_path, all_rows);

        const std::filesystem::path summary_json = out_dir / (cfg.run_tag + "_summary.json");
        write_summary_json(summary_json, cfg, cfg_hash, all_rows);

        std::cout << "[comparative] Results written to:\n"
                  << "  - " << csv_path << "\n"
                  << "  - " << table_path << "\n"
                  << "  - " << summary_json << "\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[comparative] Fatal error: " << ex.what() << "\n";
        return 1;
    }
}

} // namespace lstm_autoencoder_experiment

#ifndef NN_EXPERIMENT04_NO_MAIN
auto main(int argc, char* argv[]) -> int
{
    const bool wants_help = has_help_flag(argc, argv);

    Logger::instance().set_level(parse_log_level_from_env());
    std::unique_ptr<StreamRedirector> redirect;
    if (!wants_help)
    {
        redirect = std::make_unique<StreamRedirector>(true, true);
    }

    LstmAutoencoderExperiment experiment;
    return experiment.run(argc, argv);
}
#endif
