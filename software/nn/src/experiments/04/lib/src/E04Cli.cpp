#include "../include/ComparativeCli.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nlohmann/json.hpp"

namespace comparative_autoencoder_experiment
{

constexpr const char* kDefaultComparativeProfileStem = "lstm-compare";

static_assert(sizeof(float) == 4, "Experiment requires 32-bit float.");

void infer_dimensions_from_layer_specs(ComparativeConfig& cfg)
{
    // This function is now deprecated as layer_sizes is removed from ComparativeConfig.
    // Dimensions are inferred on-the-fly in Training.
}

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
              << "  --dataset-root <path>            Dataset root directory\n"
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
        else if (arg == "--comparative-config" || arg == "--profile")
        {
            opts.comparative_config = next();
        }
        else if (arg.rfind("--comparative-config=", 0) == 0)
        {
            opts.comparative_config = arg.substr(std::string("--comparative-config=").size());
        }
        else if (arg.rfind("--profile=", 0) == 0)
        {
            opts.comparative_config = arg.substr(std::string("--profile=").size());
        }
        else if (arg == "--dataset-root" || arg.rfind("--dataset-root=", 0) == 0)
        {
            opts.dataset_root = (arg == "--dataset-root")
                                    ? next()
                                    : arg.substr(std::string("--dataset-root=").size());
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

auto load_config(const std::filesystem::path& path, const CliOptions& cli_opts) -> ComparativeConfig
{
    ComparativeConfig cfg;

    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open comparative config: " + path.string());
    }

    nlohmann::json j;
    f >> j;

    auto has_nested_keys = [](const nlohmann::json& json) -> bool
    {
        return json.contains("experiment") && json.contains("dataset") &&
               json.contains("training") && json.contains("model") && json.contains("evaluation");
    };

    if (has_nested_keys(j))
    {
        cfg = ComparativeConfig::from_nested_json(j);
    }
    else
    {
        cfg = ComparativeConfig::from_flat_json(j);
    }

    if (!cli_opts.dataset_root.empty())
    {
        cfg.dataset.dataset_root = cli_opts.dataset_root;
    }

    return cfg;
}

auto config_hash(const ComparativeConfig& cfg) -> std::size_t
{
    nlohmann::json j;
    j["experiment"]["run_tag"] = cfg.experiment.run_tag;
    j["experiment"]["seed"] = cfg.experiment.seed;
    j["experiment"]["repeats"] = cfg.experiment.repeats;
    j["dataset"]["dataset_root"] = cfg.dataset.dataset_root;
    j["dataset"]["results_dir"] = cfg.dataset.results_dir;
    j["dataset"]["window_size"] = cfg.dataset.window_size;
    j["dataset"]["max_loaded_train_samples"] = cfg.dataset.max_loaded_train_samples;
    j["dataset"]["max_validation_samples"] = cfg.dataset.max_validation_samples;
    j["training"]["samples_per_batch"] = cfg.training.samples_per_batch;
    j["training"]["batches_per_epoch"] = cfg.training.batches_per_epoch;
    j["training"]["epochs"] = cfg.training.epochs;
    j["training"]["early_stop_patience"] = cfg.training.early_stop_patience;
    j["training"]["learning_rate"] = cfg.training.learning_rate;
    j["training"]["max_reconstruct_mean_deviation"] = cfg.training.max_reconstruct_mean_deviation;

    j["evaluation"]["datasets"] = cfg.evaluation.datasets;
    j["evaluation"]["encodings"] = cfg.evaluation.encodings;
    j["evaluation"]["snn_architectures"] = cfg.evaluation.snn_architectures;
    j["evaluation"]["v_th_values"] = cfg.evaluation.v_th_values;
    j["evaluation"]["alpha_values"] = cfg.evaluation.alpha_values;
    j["experiment"]["seed_deterministic"] = cfg.experiment.seed_deterministic;
    j["experiment"]["check_determinism"] = cfg.experiment.check_determinism;

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
