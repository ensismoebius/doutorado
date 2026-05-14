/**
 * @file src/experiments/05/experiment05.cpp
 * @brief Experiment05 entry point: biometric authentication of severely dysphonic speakers.
 *
 * Usage:
 *   experiment05 --config <profile.json>
 */

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "lib/include/E05Classifiers.hpp"
#include "lib/include/E05Config.hpp"
#include "lib/include/E05Dataset.hpp"
#include "lib/include/E05FeatureExtraction.hpp"
#include "lib/include/E05Output.hpp"
#include "lib/include/E05Paraconsistent.hpp"
#include "progress/ProgressManager.hpp"
#include "utility/progress.hpp"

namespace
{
void print_usage(const char* prog)
{
    std::cerr << "Usage: " << prog << " --config <profile.json>\n";
}

std::string parse_config_path(int argc, char* argv[])
{
    for (int i = 1; i < argc - 1; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--config")
            return std::string(argv[i + 1]);
    }
    return {};
}

// Run one full pipeline iteration with the given config (seed + run_tag already set).
void run_once(const e05::E05Config& cfg)
{
    // ── 2. Load dataset ──────────────────────────────────────────────────────
    auto view = e05::load_dataset(cfg.dataset);
    std::cout << "[E05] Loaded " << view.samples.size() << " samples from "
              << view.n_subjects << " subjects, " << view.n_stimuli << " stimuli.\n";

    // ── 3. Feature extraction ────────────────────────────────────────────────
    auto feature_sets = e05::extract_features(
        view, cfg.feature_extraction, cfg.dataset.modality);
    std::cout << "[E05] Extracted " << feature_sets.size() << " feature set(s).\n";

    // ── 4. Paraconsistent ranking ────────────────────────────────────────────
    std::vector<e05::ParaconsistentScore> scores;
    if (cfg.paraconsistent.enabled)
    {
        scores = e05::rank_feature_sets(view.samples, feature_sets);
        std::cout << "[E05] Paraconsistent ranking:\n";
        for (const auto& s : scores)
        {
            std::cout << "  " << s.label
                      << " alpha=" << s.alpha
                      << " beta=" << s.beta
                      << " D_truth=" << s.d_truth << "\n";
        }
    }

    // ── 5. Classification ────────────────────────────────────────────────────
    int n_usable_fs = 0;
    for (const auto& fs : feature_sets)
        if (!fs.vectors.empty()) ++n_usable_fs;

    const int total_outer_folds = n_usable_fs * cfg.training.k_folds;
    const uint32_t global_bar =
        nn::progress::ProgressManager::instance().create_bar(
            "E05 | " + cfg.experiment.run_tag,
            static_cast<float>(total_outer_folds));
    nn::progress::ProgressManager::instance().set_description(
        global_bar,
        cfg.classifier.type + " | " + cfg.classifier.text_mode +
        " | " + std::to_string(cfg.training.k_folds) + "-fold CV");

    int global_completed = 0;
    std::vector<e05::ClassificationResult> results;
    for (const auto& fs : feature_sets)
    {
        if (fs.vectors.empty()) continue;
        auto result = e05::run_classifier(
            view, fs.vectors, fs.label, cfg, nullptr, global_bar, &global_completed);
        results.push_back(std::move(result));
    }

    nn::progress::ProgressManager::instance().complete_bar(global_bar);

    // ── 6. Output ────────────────────────────────────────────────────────────
    const std::string& results_dir = cfg.dataset.results_dir;
    const std::string& tag         = cfg.experiment.run_tag;

    e05::write_metrics_csv(results_dir, tag, results);
    e05::write_paraconsistent_csv(results_dir, tag, scores);
    e05::write_summary_json(results_dir, tag, cfg, results, scores);
    e05::write_comparison_dat(results_dir, tag, results);

    for (const auto& r : results)
    {
        std::cout << "[E05] " << r.feature_set_label
                  << "  acc="  << r.mean_accuracy
                  << " ±"      << r.std_accuracy
                  << "  EER="  << r.mean_eer
                  << "  spec=" << r.mean_specificity << "\n";
    }
    std::cout << "[E05] Done. Results written to " << results_dir << "\n";
}
} // namespace

auto main(int argc, char* argv[]) -> int
{
    try
    {
        std::string config_path = parse_config_path(argc, argv);
        if (config_path.empty())
        {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        // ── 1. Load and validate profile ─────────────────────────────────────
        auto cfg = e05::E05Config::from_file(config_path);
        cfg.validate();

        std::cout << "[E05] run_tag=" << cfg.experiment.run_tag
                  << " modality="    << cfg.dataset.modality
                  << " strategy="    << cfg.feature_extraction.strategy
                  << " repeats="     << cfg.experiment.repeats << "\n";

        // ── Repeat loop ───────────────────────────────────────────────────────
        for (int rep = 0; rep < cfg.experiment.repeats; ++rep)
        {
            e05::E05Config rep_cfg = cfg;

            // Each repeat uses a different seed unless seed_deterministic=true.
            if (!rep_cfg.experiment.seed_deterministic)
                rep_cfg.experiment.seed += static_cast<uint32_t>(rep);

            // Append repeat index to run_tag so output files don't overwrite.
            if (cfg.experiment.repeats > 1)
                rep_cfg.experiment.run_tag =
                    cfg.experiment.run_tag + "_rep" + std::to_string(rep);

            if (cfg.experiment.repeats > 1)
                std::cout << "[E05] === Repeat " << rep + 1
                          << "/" << cfg.experiment.repeats
                          << " (seed=" << rep_cfg.experiment.seed << ") ===\n";

            run_once(rep_cfg);
        }

        nn::progress::ProgressManager::instance().shutdown();
        flushProgressAsync();

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[E05] Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
