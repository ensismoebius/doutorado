/**
 * @file src/experiments/03/lib/src/experiment04.cpp
 * @brief Integrated Experiment04 runner implementation.
 */

#include "experiment04.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "experiment04/Experiment04Config.hpp"
#include "experiment04/LSTMAutoencoder.hpp"
#include "experiment04/Trainer.hpp"
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"

namespace lstm_autoencoder_experiment
{
namespace
{
constexpr const char* kDefaultProfileStem = "lstm-default";

struct CliOptions
{
    std::string profile_name = kDefaultProfileStem;
    std::string config_path;
    std::string dataset_root;
    int epochs = -1;
    float lr = -1.0f;
    int input_size = -1;
    int hidden_size = -1;
    int latent_size = -1;
    int seq_len = -1;
    int num_layers = -1;
    float grad_clip = -1.0f;
    bool help = false;
};

auto has_experiment04_marker(const std::string& arg) -> bool
{
    return arg == "--experiment04" || arg == "--lstm-autoencoder" || arg == "--experiment=04" ||
           arg == "--experiment=experiment04" || arg == "--experiment=lstm" ||
           arg == "--experiment=lstm-autoencoder";
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
    std::cout << "Usage: " << prog << " --experiment=lstm-autoencoder [options]\n"
              << "Options:\n"
              << "  --experiment04                Run integrated Experiment04 pipeline\n"
              << "  --lstm-profile <name|path>    LSTM profile stem or JSON path\n"
              << "  --config <path>               Alias for explicit JSON config path\n"
              << "  --dataset-root <path>         Override dataset root\n"
              << "  --epochs <n>                  Override epoch count\n"
              << "  --lr <f>                      Override learning rate\n"
              << "  --input-size <n>              Override input feature dimension\n"
              << "  --hidden-size <n>             Override hidden dimension\n"
              << "  --latent-size <n>             Override latent dimension\n"
              << "  --seq-len <n>                 Override sequence length\n"
              << "  --num-layers <n>              Override stacked LSTM layer count\n"
              << "  --grad-clip <f>               Override gradient clipping norm\n"
              << "  --help                        Print this message\n";
}

auto resolve_profile_path(const CliOptions& opts) -> std::filesystem::path
{
    namespace fs = std::filesystem;

    if (!opts.config_path.empty())
    {
        return fs::path(opts.config_path);
    }

    const fs::path source_dir = source_profile_dir();
    const fs::path runtime_dir = fs::path("profiles");
    const fs::path raw_profile = fs::path(opts.profile_name);

    if (raw_profile.has_parent_path() || raw_profile.extension() == ".json")
    {
        if (fs::exists(raw_profile)) return raw_profile;
        if (raw_profile.extension() == ".json")
        {
            const fs::path source_candidate = source_dir / raw_profile.filename();
            if (fs::exists(source_candidate)) return source_candidate;
            const fs::path runtime_candidate = runtime_dir / raw_profile.filename();
            if (fs::exists(runtime_candidate)) return runtime_candidate;
        }
    }

    const std::string stem = raw_profile.stem().string().empty() ? std::string(kDefaultProfileStem)
                                                                 : raw_profile.stem().string();
    const fs::path source_candidate = source_dir / (stem + ".json");
    if (fs::exists(source_candidate)) return source_candidate;
    const fs::path runtime_candidate = runtime_dir / (stem + ".json");
    if (fs::exists(runtime_candidate)) return runtime_candidate;

    throw std::runtime_error("Cannot resolve Experiment04 profile: " + opts.profile_name);
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
        else if (has_experiment04_marker(arg))
        {
            continue;
        }
        else if (arg == "--lstm-profile")
        {
            opts.profile_name = next();
        }
        else if (arg.rfind("--lstm-profile=", 0) == 0)
        {
            opts.profile_name = arg.substr(std::string("--lstm-profile=").size());
        }
        else if (arg == "--config")
        {
            opts.config_path = next();
        }
        else if (arg == "--dataset-root")
        {
            opts.dataset_root = next();
        }
        else if (arg == "--epochs")
        {
            opts.epochs = std::stoi(next());
        }
        else if (arg == "--lr")
        {
            opts.lr = std::stof(next());
        }
        else if (arg == "--input-size")
        {
            opts.input_size = std::stoi(next());
        }
        else if (arg == "--hidden-size")
        {
            opts.hidden_size = std::stoi(next());
        }
        else if (arg == "--latent-size")
        {
            opts.latent_size = std::stoi(next());
        }
        else if (arg == "--seq-len")
        {
            opts.seq_len = std::stoi(next());
        }
        else if (arg == "--num-layers")
        {
            opts.num_layers = std::stoi(next());
        }
        else if (arg == "--grad-clip")
        {
            opts.grad_clip = std::stof(next());
        }
        else if (arg == "--profile")
        {
            ++i;
        }
        else if (arg.rfind("--profile=", 0) == 0)
        {
            continue;
        }
        else
        {
            throw std::runtime_error("Unknown Experiment04 option: " + arg);
        }
    }

    return opts;
}

auto load_config(const std::filesystem::path& path) -> Experiment04Config
{
    Experiment04Config cfg;

    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open config: " + path.string());
    }

    nlohmann::json j;
    f >> j;

    auto get = [&](const std::string& key, auto& field)
    {
        if (j.contains(key)) field = j[key].get<std::decay_t<decltype(field)>>();
    };

    get("dataset_root", cfg.dataset_root);
    get("subject_regex", cfg.subject_regex);
    get("dataset_type", cfg.dataset_type);
    get("batch_size", cfg.batch_size);
    get("max_batches_per_epoch", cfg.max_batches_per_epoch);
    get("sampler_shuffle_seed", cfg.sampler_shuffle_seed);
    get("input_size", cfg.input_size);
    get("seq_len", cfg.seq_len);
    get("hidden_size", cfg.hidden_size);
    get("latent_size", cfg.latent_size);
    get("num_layers", cfg.num_layers);
    get("epochs", cfg.epochs);
    get("learning_rate", cfg.learning_rate);
    get("grad_clip_norm", cfg.grad_clip_norm);
    get("results_dir", cfg.results_dir);
    get("run_tag", cfg.run_tag);

    return cfg;
}

auto make_synthetic_dataset(int n_samples, int seq_len, int input_size, unsigned seed)
    -> std::vector<nn::Tensor>
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> freq_dist(0.5f, 4.0f);
    std::uniform_real_distribution<float> phase_dist(0.0f, 2.0f * 3.14159f);
    std::normal_distribution<float> noise_dist(0.0f, 0.05f);

    std::vector<nn::Tensor> samples;
    samples.reserve(static_cast<size_t>(n_samples));

    for (int s = 0; s < n_samples; ++s)
    {
        nn::Tensor sample(static_cast<nn::Index>(seq_len), static_cast<nn::Index>(input_size));
        for (int t = 0; t < seq_len; ++t)
        {
            for (int d = 0; d < input_size; ++d)
            {
                const float freq = freq_dist(rng);
                const float phase = phase_dist(rng);
                float val = std::sin(
                    2.0f * 3.14159f * freq * static_cast<float>(t) / static_cast<float>(seq_len) +
                    phase);
                val += noise_dist(rng);
                sample.at(static_cast<nn::Index>(t), static_cast<nn::Index>(d)) =
                    std::clamp(val, -1.0f, 1.0f);
            }
        }
        samples.push_back(std::move(sample));
    }

    return samples;
}

void write_results(const Experiment04Config& cfg,
    const LSTMAutoencoderConfig& arch,
    const std::vector<EpochResult>& history,
    int exit_code,
    const std::string& error_msg = "")
{
    namespace fs = std::filesystem;

    fs::path results_dir =
        cfg.results_dir.empty() ? source_results_dir() : fs::path(cfg.results_dir);
    if (!fs::exists(results_dir))
    {
        results_dir = fs::path("results");
    }
    fs::create_directories(results_dir);

    nlohmann::json j;
    j["run_tag"] = cfg.run_tag;
    j["dataset_type"] = cfg.dataset_type;
    j["model"] = "lstm-autoencoder";
    j["optimizer"] = "adam";
    j["loss"] = "mse";
    j["epochs"] = cfg.epochs;
    j["learning_rate"] = cfg.learning_rate;
    j["arch"]["input_size"] = arch.input_size;
    j["arch"]["seq_len"] = arch.seq_len;
    j["arch"]["hidden_size"] = arch.hidden_size;
    j["arch"]["latent_size"] = arch.latent_size;
    j["arch"]["num_layers"] = arch.num_layers;

    nlohmann::json train_losses = nlohmann::json::array();
    nlohmann::json val_losses = nlohmann::json::array();
    for (const auto& r : history)
    {
        train_losses.push_back(r.train_loss);
        val_losses.push_back(std::isnan(r.val_loss) ? nullptr : nlohmann::json(r.val_loss));
    }
    j["train_losses"] = train_losses;
    j["val_losses"] = val_losses;
    j["exit_code"] = exit_code;
    if (!error_msg.empty()) j["error"] = error_msg;

    const std::string run_tag = cfg.run_tag.empty() ? std::string("experiment04") : cfg.run_tag;
    const fs::path filename = results_dir / (run_tag + "_results.json");
    std::ofstream out(filename);
    out << j.dump(2) << "\n";
    std::cout << "[experiment04] Results written to " << filename.string() << "\n";
}
} // namespace

auto should_run_from_cli(int argc, char* argv[]) -> bool
{
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
    using lstm_autoencoder_experiment::CliOptions;
    using lstm_autoencoder_experiment::LSTMAutoencoder;
    using lstm_autoencoder_experiment::LSTMAutoencoderConfig;
    using lstm_autoencoder_experiment::Trainer;

    try
    {
        const CliOptions cli = lstm_autoencoder_experiment::parse_cli(argc, argv);
        if (cli.help)
        {
            lstm_autoencoder_experiment::print_usage(argv[0]);
            return 0;
        }

        Experiment04Config cfg = lstm_autoencoder_experiment::load_config(
            lstm_autoencoder_experiment::resolve_profile_path(cli));
        if (!cli.dataset_root.empty()) cfg.dataset_root = cli.dataset_root;
        if (cli.epochs > 0) cfg.epochs = cli.epochs;
        if (cli.lr > 0.0f) cfg.learning_rate = cli.lr;
        if (cli.input_size > 0) cfg.input_size = cli.input_size;
        if (cli.hidden_size > 0) cfg.hidden_size = cli.hidden_size;
        if (cli.latent_size > 0) cfg.latent_size = cli.latent_size;
        if (cli.seq_len > 0) cfg.seq_len = cli.seq_len;
        if (cli.num_layers > 0) cfg.num_layers = cli.num_layers;
        if (cli.grad_clip >= 0.0f) cfg.grad_clip_norm = cli.grad_clip;

        std::cout << "[experiment04] Configuration:\n"
                  << "  dataset_root : " << cfg.dataset_root << "\n"
                  << "  dataset_type : " << cfg.dataset_type << "\n"
                  << "  input_size   : " << cfg.input_size << "\n"
                  << "  seq_len      : " << cfg.seq_len << "\n"
                  << "  hidden_size  : " << cfg.hidden_size << "\n"
                  << "  latent_size  : " << cfg.latent_size << "\n"
                  << "  num_layers   : " << cfg.num_layers << "\n"
                  << "  epochs       : " << cfg.epochs << "\n"
                  << "  lr           : " << cfg.learning_rate << "\n"
                  << "  grad_clip    : " << cfg.grad_clip_norm << "\n";

        constexpr int kNTrain = 200;
        constexpr int kNVal = 40;
        std::cout << "[experiment04] Building synthetic dataset (" << kNTrain << " train, " << kNVal
                  << " val)\n";

        auto train_samples = lstm_autoencoder_experiment::make_synthetic_dataset(
            kNTrain, cfg.seq_len, cfg.input_size, cfg.sampler_shuffle_seed);
        auto val_samples = lstm_autoencoder_experiment::make_synthetic_dataset(
            kNVal, cfg.seq_len, cfg.input_size, cfg.sampler_shuffle_seed + 1u);

        LSTMAutoencoderConfig arch;
        arch.input_size = cfg.input_size;
        arch.seq_len = cfg.seq_len;
        arch.hidden_size = cfg.hidden_size;
        arch.latent_size = cfg.latent_size;
        arch.num_layers = cfg.num_layers;
        LSTMAutoencoder model(arch);

        std::cout << "[experiment04] Model built. Parameter count estimate: "
                  << model.params().size() << " tensors\n";

        Trainer trainer(model, cfg);
        const auto history = trainer.fit(train_samples, val_samples);

        lstm_autoencoder_experiment::write_results(cfg, arch, history, 0);
        std::cout << "[experiment04] Done.\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[experiment04] Fatal error: " << ex.what() << "\n";
        try
        {
            Experiment04Config cfg;
            lstm_autoencoder_experiment::LSTMAutoencoderConfig arch;
            lstm_autoencoder_experiment::write_results(cfg, arch, {}, 1, ex.what());
        }
        catch (...)
        {
        }
        return 1;
    }
}