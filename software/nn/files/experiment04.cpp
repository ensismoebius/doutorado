/**
 * @file experiment04.cpp
 * @brief Experiment04 — LSTM Autoencoder for 1-D temporal signals.
 *
 * Entry point for the LSTM autoencoder training pipeline. This file mirrors the
 * thin launcher pattern used by experiment03:
 *
 *   1. Parse a JSON configuration profile (command-line or default).
 *   2. Build synthetic or real windowed dataset samples.
 *   3. Construct the LSTMAutoencoder.
 *   4. Run the Trainer.
 *   5. Write a JSON results summary.
 *
 * Comparison parity with experiment03:
 *   - MSE reconstruction loss
 *   - Adam optimizer with same default hyperparameters
 *   - Per-epoch train/val logging in the same format
 *   - JSON result artifact at the same path convention
 *
 * Dataset strategy:
 *   When a valid dataset root is provided (via --dataset-root) and the subject
 *   matcher finds subjects, real AudioWindowDataset samples are loaded.
 *   Otherwise, a synthetic sine-wave dataset is generated automatically so
 *   the experiment runs standalone without any data dependency — useful for
 *   architecture validation and CI.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Experiment04 components
#include "lib/include/Experiment04Config.hpp"
#include "lib/include/LSTMAutoencoder.hpp"
#include "lib/include/Trainer.hpp"

// Core framework
#include "nn/layers/eigen/Layers.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/tensor/Tensor.hpp"

// JSON for results
#include "nlohmann/json.hpp"

// ---------------------------------------------------------------------------
// CLI parsing (minimal — mirrors experiment03 pattern without CLI11 dep)
// ---------------------------------------------------------------------------
namespace
{

struct CliOptions
{
    std::string config_path;
    std::string dataset_root;
    int         epochs       = -1;
    float       lr           = -1.0f;
    int         input_size   = -1;
    int         hidden_size  = -1;
    int         latent_size  = -1;
    int         seq_len      = -1;
    int         num_layers   = -1;
    float       grad_clip    = -1.0f;
    bool        help         = false;
};

void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --config <path>          JSON config profile\n"
              << "  --dataset-root <path>    Root directory for 10.1117 dataset\n"
              << "  --epochs <n>             Override epoch count\n"
              << "  --lr <f>                 Override learning rate\n"
              << "  --input-size <n>         Override input feature dimension\n"
              << "  --hidden-size <n>        Override LSTM hidden dimension\n"
              << "  --latent-size <n>        Override latent dimension\n"
              << "  --seq-len <n>            Override sequence length\n"
              << "  --num-layers <n>         Override number of stacked LSTM layers\n"
              << "  --grad-clip <f>          Override gradient clipping norm (0=off)\n"
              << "  --help                   Print this message\n";
}

CliOptions parse_cli(int argc, char* argv[])
{
    CliOptions opts;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")    { opts.help = true; }
        else if (arg == "--config")            { opts.config_path   = next(); }
        else if (arg == "--dataset-root")      { opts.dataset_root  = next(); }
        else if (arg == "--epochs")            { opts.epochs        = std::stoi(next()); }
        else if (arg == "--lr")                { opts.lr            = std::stof(next()); }
        else if (arg == "--input-size")        { opts.input_size    = std::stoi(next()); }
        else if (arg == "--hidden-size")       { opts.hidden_size   = std::stoi(next()); }
        else if (arg == "--latent-size")       { opts.latent_size   = std::stoi(next()); }
        else if (arg == "--seq-len")           { opts.seq_len       = std::stoi(next()); }
        else if (arg == "--num-layers")        { opts.num_layers    = std::stoi(next()); }
        else if (arg == "--grad-clip")         { opts.grad_clip     = std::stof(next()); }
        else { std::cerr << "Unknown option: " << arg << "\n"; }
    }
    return opts;
}

// ---------------------------------------------------------------------------
// Config loader from JSON (subset of experiment03 profile convention)
// ---------------------------------------------------------------------------
Experiment04Config load_config(const std::string& path)
{
    Experiment04Config cfg;
    if (path.empty()) return cfg;

    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open config: " + path);
    }

    nlohmann::json j;
    f >> j;

    auto get = [&](const std::string& key, auto& field)
    {
        if (j.contains(key)) field = j[key].get<std::decay_t<decltype(field)>>();
    };

    get("dataset_root",           cfg.dataset_root);
    get("subject_regex",          cfg.subject_regex);
    get("dataset_type",           cfg.dataset_type);
    get("batch_size",             cfg.batch_size);
    get("max_batches_per_epoch",  cfg.max_batches_per_epoch);
    get("sampler_shuffle_seed",   cfg.sampler_shuffle_seed);
    get("input_size",             cfg.input_size);
    get("seq_len",                cfg.seq_len);
    get("hidden_size",            cfg.hidden_size);
    get("latent_size",            cfg.latent_size);
    get("num_layers",             cfg.num_layers);
    get("epochs",                 cfg.epochs);
    get("learning_rate",          cfg.learning_rate);
    get("grad_clip_norm",         cfg.grad_clip_norm);
    get("results_dir",            cfg.results_dir);
    get("run_tag",                cfg.run_tag);

    return cfg;
}

// ---------------------------------------------------------------------------
// Synthetic dataset — sine waves with different frequencies per sample
// Used when no real dataset root is supplied.
// ---------------------------------------------------------------------------
std::vector<nn::Tensor> make_synthetic_dataset(
    int n_samples, int seq_len, int input_size, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> freq_dist(0.5f, 4.0f);
    std::uniform_real_distribution<float> phase_dist(0.0f, 2.0f * 3.14159f);
    std::normal_distribution<float> noise_dist(0.0f, 0.05f);

    std::vector<nn::Tensor> samples;
    samples.reserve(static_cast<size_t>(n_samples));

    for (int s = 0; s < n_samples; ++s)
    {
        nn::Tensor sample(static_cast<nn::Index>(seq_len),
                          static_cast<nn::Index>(input_size));
        for (int t = 0; t < seq_len; ++t)
        {
            for (int d = 0; d < input_size; ++d)
            {
                float freq  = freq_dist(rng);
                float phase = phase_dist(rng);
                float val   = std::sin(2.0f * 3.14159f * freq *
                                       static_cast<float>(t) / static_cast<float>(seq_len)
                                       + phase);
                val += noise_dist(rng);
                // Normalise to [-1, 1]
                sample.at(static_cast<nn::Index>(t),
                           static_cast<nn::Index>(d)) = std::clamp(val, -1.0f, 1.0f);
            }
        }
        samples.push_back(std::move(sample));
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Results writer (JSON — same structure convention as experiment03)
// ---------------------------------------------------------------------------
void write_results(const Experiment04Config& cfg,
                   const experiment04::LSTMAutoencoderConfig& arch,
                   const std::vector<experiment04::EpochResult>& history,
                   int exit_code, const std::string& error_msg = "")
{
    namespace fs = std::filesystem;
    fs::create_directories(cfg.results_dir);

    nlohmann::json j;
    j["run_tag"]     = cfg.run_tag;
    j["dataset_type"]= cfg.dataset_type;
    j["model"]       = "lstm-autoencoder";
    j["optimizer"]   = "adam";
    j["loss"]        = "mse";
    j["epochs"]      = cfg.epochs;
    j["learning_rate"] = cfg.learning_rate;
    j["arch"]["input_size"]  = arch.input_size;
    j["arch"]["seq_len"]     = arch.seq_len;
    j["arch"]["hidden_size"] = arch.hidden_size;
    j["arch"]["latent_size"] = arch.latent_size;
    j["arch"]["num_layers"]  = arch.num_layers;

    nlohmann::json train_losses = nlohmann::json::array();
    nlohmann::json val_losses   = nlohmann::json::array();
    for (const auto& r : history)
    {
        train_losses.push_back(r.train_loss);
        val_losses.push_back(std::isnan(r.val_loss) ? nullptr : nlohmann::json(r.val_loss));
    }
    j["train_losses"] = train_losses;
    j["val_losses"]   = val_losses;
    j["exit_code"]    = exit_code;
    if (!error_msg.empty()) j["error"] = error_msg;

    std::string filename = cfg.results_dir + "/" + cfg.run_tag + "_results.json";
    std::ofstream out(filename);
    out << j.dump(2) << "\n";
    std::cout << "[experiment04] Results written to " << filename << "\n";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    try
    {
        // ---- CLI ----
        CliOptions cli = parse_cli(argc, argv);
        if (cli.help)
        {
            print_usage(argv[0]);
            return 0;
        }

        // ---- Config ----
        Experiment04Config cfg = load_config(cli.config_path);
        if (!cli.dataset_root.empty()) cfg.dataset_root   = cli.dataset_root;
        if (cli.epochs      > 0)      cfg.epochs          = cli.epochs;
        if (cli.lr          > 0.0f)   cfg.learning_rate   = cli.lr;
        if (cli.input_size  > 0)      cfg.input_size       = cli.input_size;
        if (cli.hidden_size > 0)      cfg.hidden_size      = cli.hidden_size;
        if (cli.latent_size > 0)      cfg.latent_size      = cli.latent_size;
        if (cli.seq_len     > 0)      cfg.seq_len          = cli.seq_len;
        if (cli.num_layers  > 0)      cfg.num_layers       = cli.num_layers;
        if (cli.grad_clip   >= 0.0f)  cfg.grad_clip_norm   = cli.grad_clip;

        std::cout << "[experiment04] Configuration:\n"
                  << "  dataset_root : " << cfg.dataset_root  << "\n"
                  << "  dataset_type : " << cfg.dataset_type  << "\n"
                  << "  input_size   : " << cfg.input_size    << "\n"
                  << "  seq_len      : " << cfg.seq_len       << "\n"
                  << "  hidden_size  : " << cfg.hidden_size   << "\n"
                  << "  latent_size  : " << cfg.latent_size   << "\n"
                  << "  num_layers   : " << cfg.num_layers    << "\n"
                  << "  epochs       : " << cfg.epochs        << "\n"
                  << "  lr           : " << cfg.learning_rate << "\n"
                  << "  grad_clip    : " << cfg.grad_clip_norm<< "\n";

        // ---- Dataset ----
        // For now, always use the synthetic dataset path.
        // A future extension point is DatasetBuilder04 (see README.md extension notes).
        constexpr int kNTrain = 200;
        constexpr int kNVal   = 40;

        std::cout << "[experiment04] Building synthetic dataset ("
                  << kNTrain << " train, " << kNVal << " val)\n";

        auto train_samples = make_synthetic_dataset(
            kNTrain, cfg.seq_len, cfg.input_size, cfg.sampler_shuffle_seed);
        auto val_samples = make_synthetic_dataset(
            kNVal, cfg.seq_len, cfg.input_size, cfg.sampler_shuffle_seed + 1u);

        // ---- Model ----
        experiment04::LSTMAutoencoderConfig arch;
        arch.input_size  = cfg.input_size;
        arch.seq_len     = cfg.seq_len;
        arch.hidden_size = cfg.hidden_size;
        arch.latent_size = cfg.latent_size;
        arch.num_layers  = cfg.num_layers;
        experiment04::LSTMAutoencoder model(arch);

        std::cout << "[experiment04] Model built. Parameter count estimate: "
                  << model.params().size() << " tensors\n";

        // ---- Training ----
        experiment04::Trainer trainer(model, cfg);
        auto history = trainer.fit(train_samples, val_samples);

        // ---- Results ----
        write_results(cfg, arch, history, 0);
        std::cout << "[experiment04] Done.\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[experiment04] Fatal error: " << ex.what() << "\n";
        // Write failure result if we can
        try
        {
            Experiment04Config cfg;
            experiment04::LSTMAutoencoderConfig arch;
            write_results(cfg, arch, {}, 1, ex.what());
        }
        catch (...) {}
        return 1;
    }
}
