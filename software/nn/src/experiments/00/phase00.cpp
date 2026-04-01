/**
 * @file phase00.cpp
 * @brief PHASE 0 experiment entry point (frozen baseline pipeline).
 *
 * End-to-end flow:
 * - load YAML config (with frozen defaults for this phase)
 * - aggregate trials from the dataset
 * - extract features, normalize to [0,1], sanity-check
 * - compute paraconsistent metrics
 * - train a small classifier and write metrics/artifacts
 */

#include <filesystem>

#include "../../include/nn/logging/Logger.hpp"
#include "Config.hpp"
#include "nn/logging/StreamRedirector.hpp"
#include "phase00_data.hpp"
#include "phase00_features.hpp"
#include "phase00_training.hpp"

auto main(int argc, const char* argv[]) -> int
{
    nn::logging::StreamRedirector redirect(true, true);
    std::filesystem::path config_path;
    if (argc == 1)
    {
        config_path = phase00::default_config_path();
    }
    else if (argc == 2)
    {
        config_path = argv[1];
    }
    else
    {
        NN_LOG_ERROR(std::string("Usage: ") + argv[0] + " [config.json]");
        return 1;
    }

    // Load configuration
    auto cfg_opt = Config::load(config_path.string());
    if (!cfg_opt)
    {
        NN_LOG_ERROR(std::string("Failed to load config from ") + config_path.string());
        return 1;
    }

    // Extract config from optional
    const Config& cfg = cfg_opt.value();

    NN_LOG_INFO("PHASE 0: Frozen baseline (1.5 s window / 50% overlap / [0,1] normalization)");
    NN_LOG_INFO(std::string("Using config: ") + config_path.string());

    if (!std::filesystem::exists(cfg.dataset_base_path))
    {
        NN_LOG_ERROR(std::string("Dataset base path does not exist: ") + cfg.dataset_base_path);
        return 1;
    }

    auto trials = phase00::aggregate_trials(cfg);

    if (trials.empty())
    {
        NN_LOG_ERROR("No data processed for any subject. Exiting.");
        return 1;
    }

    std::vector<std::vector<double>> all_combined_features;
    std::vector<int> all_combined_labels;
    all_combined_features.reserve(trials.size());
    all_combined_labels.reserve(trials.size());

    for (auto& trial : trials)
    {
        all_combined_labels.push_back(trial.label);
        all_combined_features.push_back(std::move(trial.features));
    }

    phase00::normalize_features(all_combined_features, cfg.range);

    if (!phase00::verify_normalization(all_combined_features, cfg.range))
    {
        NN_LOG_ERROR("Normalization check failed: values are outside [0, 1].");
        return 1;
    }

    auto [normalized_labels, ordered_labels] = phase00::build_label_index(all_combined_labels);
    const int num_classes = static_cast<int>(ordered_labels.size());

    auto [alpha, beta, g1, g2] =
        phase00::compute_paraconsistent_metrics(all_combined_features, normalized_labels);

    auto train_result =
        phase00::train_resnet_snn(all_combined_features, normalized_labels, cfg, num_classes);

    std::filesystem::create_directories(cfg.results_dir);
    const auto metrics_path = std::filesystem::path(cfg.results_dir) / cfg.metrics_file;
    const auto torch_state_path = std::filesystem::path(cfg.results_dir) / cfg.torch_state_file;

    phase00::save_results(metrics_path, alpha, beta, g1, g2, train_result.accuracy);
    phase00::save_torch_state(torch_state_path, train_result);

    NN_LOG_INFO(std::string("Experiment completed. Metrics: ") + metrics_path.string() +
                " | torch state: " + torch_state_path.string());

    return 0;
}