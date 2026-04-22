#include "../include/ComparativeCli.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nlohmann/json.hpp"

namespace comparative_autoencoder_experiment
{

constexpr const char* kDefaultComparativeProfileStem = "lstm-compare";

static_assert(sizeof(float) == 4, "Experiment requires 32-bit float.");

auto has_compare_marker(const std::string& arg) -> bool
{
    return arg == "--comparative" || arg == "--experiment=snn-lstm-compare" ||
           arg == "--experiment=compare" || arg == "--experiment=comparative";
}

auto source_profile_dir() -> std::filesystem::path
{
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "profiles";
}

auto source_results_dir() -> std::filesystem::path
{
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "results";
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

} // namespace lstm_autoencoder_experiment
