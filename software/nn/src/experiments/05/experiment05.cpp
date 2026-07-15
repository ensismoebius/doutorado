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
#include <sstream>
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
        if (arg == "--config") return std::string(argv[i + 1]);
    }
    return {};
}

// Run one full pipeline iteration with the given config (seed + run_tag already set).
// `view` is loaded once by the caller — dataset loading is seed-independent, so
// repeats share it. `cached_features` is non-null when feature extraction is also
// seed-independent (handcrafted strategy); AE strategies pass null and re-extract
// per repeat because AE training depends on the repeat's seed.
void run_once(const e05::E05Config& cfg,
    const e05::E05DatasetView& view,
    const std::vector<e05::FeatureSet>* cached_features)
{
    auto& pm = nn::progress::ProgressManager::instance();

    // ── 3. Feature extraction ────────────────────────────────────────────────
    std::vector<e05::FeatureSet> extracted;
    if (cached_features == nullptr)
    {
        extracted = e05::extract_features(view,
            cfg.feature_extraction,
            cfg.training,
            cfg.dataset.modality,
            cfg.dataset.fusion_mode,
            cfg.experiment.seed);
    }
    const auto& feature_sets = (cached_features != nullptr) ? *cached_features : extracted;
    pm.log("[E05] Extracted " + std::to_string(feature_sets.size()) + " feature set(s).");

    // ── 4. Paraconsistent ranking ────────────────────────────────────────────
    std::vector<e05::ParaconsistentScore> scores;
    if (cfg.paraconsistent.enabled)
    {
        scores = e05::rank_feature_sets(view.samples, feature_sets);
        pm.log("[E05] Paraconsistent ranking:");
        for (const auto& s : scores)
        {
            std::ostringstream oss;
            oss << "  " << s.label << " alpha=" << s.alpha << " beta=" << s.beta
                << " D_truth=" << s.d_truth;
            pm.log(oss.str());
        }
    }

    // ── 5. Classification (Phase 01 only) ────────────────────────────────────
    // When classifier.enabled is false, this is a Phase 00 run: stop after
    // paraconsistent ranking and emit only the ranking artefacts.
    std::vector<e05::ClassificationResult> results;
    if (cfg.classifier.enabled)
    {
        int n_usable_fs = 0;
        for (const auto& fs : feature_sets)
            if (!fs.vectors.empty()) ++n_usable_fs;

        const int total_outer_folds = n_usable_fs * cfg.training.k_folds;
        const uint32_t global_bar =
            pm.create_bar("E05 | " + cfg.experiment.run_tag, static_cast<float>(total_outer_folds));
        pm.set_description(global_bar,
            cfg.classifier.type + " | " + cfg.classifier.text_mode + " | " +
                std::to_string(cfg.training.k_folds) + "-fold CV");

        int global_completed = 0;
        for (const auto& fs : feature_sets)
        {
            if (fs.vectors.empty()) continue;
            auto result = e05::run_classifier(
                view, fs.vectors, fs.label, cfg, nullptr, global_bar, &global_completed);
            results.push_back(std::move(result));
        }

        pm.complete_bar(global_bar);
    }
    else
    {
        pm.log("[E05] classifier.enabled=false — Phase 00 run, stopping after ranking.");
    }

    // ── 6. Output ────────────────────────────────────────────────────────────
    const std::string& results_dir = cfg.dataset.results_dir;
    const std::string& tag = cfg.experiment.run_tag;

    // Ranking artefacts are always written; classifier artefacts only when it ran.
    e05::write_paraconsistent_csv(results_dir, tag, scores);
    e05::write_summary_json(results_dir,
        tag,
        cfg,
        results,
        scores,
        view.n_subjects,
        view.n_stimuli,
        view.samples.size());
    if (cfg.classifier.enabled)
    {
        e05::write_metrics_csv(results_dir, tag, results);
        e05::write_comparison_dat(results_dir, tag, results);
    }

    for (const auto& r : results)
    {
        std::ostringstream oss;
        oss << "[E05] " << r.feature_set_label << "  acc=" << r.mean_accuracy << " ±"
            << r.std_accuracy << "  EER=" << r.mean_eer << "  spec=" << r.mean_specificity;
        pm.log(oss.str());
    }
    pm.log("[E05] Done. Results written to " + results_dir);
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
                  << " modality=" << cfg.dataset.modality
                  << " strategy=" << cfg.feature_extraction.strategy
                  << " repeats=" << cfg.experiment.repeats << "\n";

        // ── 2. Load dataset (once — identical across repeats) ────────────────
        auto view = e05::load_dataset(cfg.dataset);
        {
            std::ostringstream oss;
            oss << "[E05] Loaded " << view.samples.size() << " samples from " << view.n_subjects
                << " subjects, " << view.n_stimuli << " stimuli.";
            nn::progress::ProgressManager::instance().log(oss.str());
        }

        // Handcrafted features are deterministic (no seed involved), so extract
        // once and share across repeats. AE-based extraction trains a network per
        // repeat seed and must stay inside the repeat loop.
        std::vector<e05::FeatureSet> shared_features;
        const bool features_seed_independent = (cfg.feature_extraction.strategy == "handcrafted");
        if (features_seed_independent)
            shared_features = e05::extract_features(view,
                cfg.feature_extraction,
                cfg.training,
                cfg.dataset.modality,
                cfg.dataset.fusion_mode,
                cfg.experiment.seed);

        // ── Repeat loop ───────────────────────────────────────────────────────
        for (int rep = 0; rep < cfg.experiment.repeats; ++rep)
        {
            e05::E05Config rep_cfg = cfg;

            // Each repeat uses a different seed unless seed_deterministic=true.
            if (!rep_cfg.experiment.seed_deterministic)
                rep_cfg.experiment.seed += static_cast<uint32_t>(rep);

            // Append repeat index to run_tag so output files don't overwrite.
            if (cfg.experiment.repeats > 1)
                rep_cfg.experiment.run_tag = cfg.experiment.run_tag + "_rep" + std::to_string(rep);

            if (cfg.experiment.repeats > 1)
            {
                std::ostringstream oss;
                oss << "[E05] === Repeat " << rep + 1 << "/" << cfg.experiment.repeats
                    << " (seed=" << rep_cfg.experiment.seed << ") ===";
                nn::progress::ProgressManager::instance().log(oss.str());
            }

            run_once(rep_cfg, view, features_seed_independent ? &shared_features : nullptr);
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
