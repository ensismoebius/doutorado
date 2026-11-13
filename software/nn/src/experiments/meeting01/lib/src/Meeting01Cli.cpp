#include "../include/Meeting01Cli.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nlohmann/json.hpp"

namespace meeting01
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
              << "  --dataset-root <path>            Dataset root directory\n"
              << "  --cv-fold <n>                    Override the profile's nested-LOSO fold "
                 "index (0..cv_num_folds-1)\n"
              << "  --dataset <name>                 Run only this dataset from "
                 "evaluation.datasets (fsdd|audiomnist|eegmmidb|siena)\n"
              << "  --no-tui                          Disable the live progress TUI "
                 "(auto-off when stdout is not a terminal)\n"
              << "  --print-resolved-config          Print the validated, resolved profile as "
                 "JSON and exit (02_ --placeholders reads it)\n"
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
        else if (arg == "--cv-fold" || arg.rfind("--cv-fold=", 0) == 0)
        {
            const std::string value =
                (arg == "--cv-fold") ? next() : arg.substr(std::string("--cv-fold=").size());
            opts.cv_fold = std::stoi(value);
            opts.cv_fold_set = true;
        }
        else if (arg == "--dataset" || arg.rfind("--dataset=", 0) == 0)
        {
            opts.dataset =
                (arg == "--dataset") ? next() : arg.substr(std::string("--dataset=").size());
        }
        else if (arg == "--no-tui")
        {
            opts.no_tui = true;
        }
        else if (arg == "--print-resolved-config")
        {
            opts.print_resolved_config = true;
        }
        else
        {
            // An ignored typo is not harmless here: a misspelt option (say "--cv_fold 3")
            // would silently run the profile's own fold instead -- weeks of training that
            // truncate that fold's events log and rewrite its result files.
            throw std::runtime_error(
                "Unknown argument '" + arg + "'. Run with --help for the accepted options.");
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
    fs::path raw = fs::path(profile_name);

    if ((raw.has_parent_path() || raw.extension() == ".json") && fs::exists(raw))
    {
        return raw;
    }

    const std::string stem = raw.stem().string().empty()
                                 ? std::string(kDefaultComparativeProfileStem)
                                 : raw.stem().string();
    fs::path source_candidate = source_dir / (stem + ".json");
    if (fs::exists(source_candidate)) return source_candidate;

    const fs::path runtime_candidate = runtime_dir / (stem + ".json");
    if (fs::exists(runtime_candidate)) return runtime_candidate;

    throw std::runtime_error("Cannot resolve comparative profile: " + profile_name);
}

auto load_config(const std::filesystem::path& path, const CliOptions& cli_opts) -> Meeting01Config
{
    Meeting01Config cfg;

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
        cfg = Meeting01Config::from_nested_json(j);
    }
    else
    {
        cfg = Meeting01Config::from_flat_json(j);
    }

    if (!cli_opts.dataset_root.empty())
    {
        cfg.dataset.dataset_root = cli_opts.dataset_root;
    }

    if (cli_opts.cv_fold_set)
    {
        cfg.dataset.cv_fold = cli_opts.cv_fold;
    }

    if (!cli_opts.dataset.empty())
    {
        const auto& ds = cfg.evaluation.datasets;
        if (std::find(ds.begin(), ds.end(), cli_opts.dataset) == ds.end())
            throw std::runtime_error("--dataset '" + cli_opts.dataset +
                                     "' is not listed in the profile's evaluation.datasets");
        cfg.evaluation.datasets = {cli_opts.dataset};
    }

    return cfg;
}

auto dataset_source_json(const Meeting01Config::DatasetSource& s) -> nlohmann::json
{
    return {{"name", s.name},
        {"root", s.root},
        {"window_size", s.window_size},
        {"cv_num_folds", s.cv_num_folds},
        {"sample_rate", s.sample_rate},
        {"max_windows_per_recording", s.max_windows_per_recording},
        {"loso_max_train_windows", s.loso_max_train_windows},
        {"loso_max_val_windows", s.loso_max_val_windows},
        {"loso_max_test_windows", s.loso_max_test_windows},
        {"latent_dim", s.latent_dim}};
}

auto resolved_config_json(const Meeting01Config& cfg) -> nlohmann::json
{
    nlohmann::json j;
    const auto& e = cfg.experiment;
    j["experiment"] = {{"run_tag", e.run_tag},
        {"seed", e.seed},
        {"repeats", e.repeats},
        {"seed_deterministic", e.seed_deterministic},
        {"check_determinism", e.check_determinism}};

    const auto& d = cfg.dataset;
    j["dataset"] = {{"dataset_root", d.dataset_root},
        {"results_dir", d.results_dir},
        {"window_size", d.window_size},
        {"max_loaded_train_samples", d.max_loaded_train_samples},
        {"max_validation_samples", d.max_validation_samples},
        {"cv_fold", d.cv_fold},
        {"cv_num_folds", d.cv_num_folds},
        {"max_windows_per_recording", d.max_windows_per_recording},
        {"loso_max_train_windows", d.loso_max_train_windows},
        {"loso_max_val_windows", d.loso_max_val_windows},
        {"loso_max_test_windows", d.loso_max_test_windows},
        {"latent_dim", d.latent_dim},
        {"sources", nlohmann::json::array()}};
    for (const auto& s : d.sources) j["dataset"]["sources"].push_back(dataset_source_json(s));

    const auto& t = cfg.training;
    j["training"] = {{"samples_per_batch", t.samples_per_batch},
        {"batches_per_epoch", t.batches_per_epoch},
        {"epochs", t.epochs},
        {"early_stop_patience", t.early_stop_patience},
        {"early_stop_min_skill", t.early_stop_min_skill},
        {"learning_rate", t.learning_rate},
        {"lr_warmup_steps", t.lr_warmup_steps},
        {"learning_rate_biophysical", t.learning_rate_biophysical},
        {"beta1", t.beta1},
        {"beta2", t.beta2},
        {"epsilon", t.epsilon},
        {"max_reconstruct_mean_deviation", t.max_reconstruct_mean_deviation}};

    const auto& m = cfg.model;
    j["model"] = {{"latent_dim", m.latent_dim},
        {"lstm_hidden_size", m.lstm_hidden_size},
        {"lstm_frame_size", m.lstm_frame_size},
        {"time_steps", m.time_steps},
        {"recurrent_time_steps", m.recurrent_time_steps},
        {"snn_time_pool", m.snn_time_pool},
        {"snn_decoder_neuron", m.snn_decoder_neuron},
        {"initializer", m.initializer},
        {"branch_hidden_size", m.branch_hidden_size},
        {"fusion_hidden_size", m.fusion_hidden_size},
        {"transformer_d_model", m.transformer_d_model},
        {"transformer_heads", m.transformer_heads},
        {"transformer_layers", m.transformer_layers},
        {"transformer_d_ff", m.transformer_d_ff},
        {"transformer_norm_first", m.transformer_norm_first},
        {"denoising_noise_std", m.denoising_noise_std},
        {"loss_type", m.loss_type},
        {"encoder_layer_spec", m.encoder_layer_spec},
        {"decoder_layer_spec", m.decoder_layer_spec},
        {"branch_encoder_layer_spec", m.branch_encoder_layer_spec},
        {"branch_decoder_layer_spec", m.branch_decoder_layer_spec},
        {"fusion_encoder_layer_spec", m.fusion_encoder_layer_spec},
        {"fusion_decoder_layer_spec", m.fusion_decoder_layer_spec}};

    const auto& ev = cfg.evaluation;
    j["evaluation"] = {{"datasets", ev.datasets},
        {"encodings", ev.encodings},
        {"baselines", ev.baselines},
        {"snn_architectures", ev.snn_architectures}};
    // The search mechanics every family's block shares; checkpoint_every_generations is left
    // out (when the GA saves its state changes no result).
    auto mechanics = [](const auto& g)
    {
        return nlohmann::json{{"population_size", g.population_size},
            {"generations", g.generations},
            {"crossover_prob", g.crossover_prob},
            {"mutation_prob", g.mutation_prob},
            {"tournament_k", g.tournament_k},
            {"winner_seeds", g.winner_seeds},
            {"seed", g.seed}};
    };
    const auto& snn = ev.ga.snn;
    auto js = mechanics(snn);
    js.update({{"min_layers", snn.min_layers},
        {"max_layers", snn.max_layers},
        {"min_width", snn.min_width},
        {"max_width", snn.max_width},
        {"voltage_threshold_min", snn.voltage_threshold_min},
        {"voltage_threshold_max", snn.voltage_threshold_max},
        {"alpha_min", snn.alpha_min},
        {"alpha_max", snn.alpha_max},
        {"time_steps_min", snn.time_steps_min},
        {"time_steps_max", snn.time_steps_max}});
    j["evaluation"]["ga"]["snn"] = js;
    auto recurrent = [&](const Meeting01Config::RecurrentGa& g)
    {
        auto jr = mechanics(g);
        jr.update({{"min_hidden", g.min_hidden},
            {"max_hidden", g.max_hidden},
            {"min_layers", g.min_layers},
            {"max_layers", g.max_layers}});
        return jr;
    };
    j["evaluation"]["ga"]["lstm"] = recurrent(ev.ga.lstm);
    j["evaluation"]["ga"]["gru"] = recurrent(ev.ga.gru);
    const auto& tr = ev.ga.transformer;
    auto jt = mechanics(tr);
    jt.update({{"min_d_model", tr.min_d_model},
        {"max_d_model", tr.max_d_model},
        {"head_choices", tr.head_choices},
        {"min_layers", tr.min_layers},
        {"max_layers", tr.max_layers},
        {"min_d_ff", tr.min_d_ff},
        {"max_d_ff", tr.max_d_ff}});
    j["evaluation"]["ga"]["transformer"] = jt;
    return j;
}

// Until 2026-11-02 this hashed a hand-picked subset that left out most of model.* (the SNN's
// time pool, decoder neuron, initializer, latent width, the baselines' recurrent_time_steps),
// the learning rate of the biophysical parameters and every GA block's winner_seeds and
// operator probabilities: a profile edited in any of them between two RESUME=1 starts reused
// the checkpoints trained under the old values. Hashing resolved_config_json closes that.
auto config_hash(const Meeting01Config& cfg) -> std::size_t
{
    return std::hash<std::string>{}(resolved_config_json(cfg).dump());
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

} // namespace meeting01
