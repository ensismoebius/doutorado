/**
 * @file src/experiments/thesis/thesis.cpp
 * @brief Experiment05 entry point: biometric authentication of severely dysphonic speakers.
 *
 * Usage:
 *   thesis --config <profile.json>
 */

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

#include "lib/include/ThesisClassifiers.hpp"
#include "lib/include/ThesisConfig.hpp"
#include "lib/include/ThesisDataset.hpp"
#include "lib/include/ThesisFeatureExtraction.hpp"
#include "lib/include/ThesisOutput.hpp"
#include "lib/include/ThesisParaconsistent.hpp"
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

// Provenance/determinism fingerprint of the run — the reproducibility-defining
// fields hashed to one value (the Guayaquil config_hash analog). Two runs with the same
// hash trained the same model on the same data pipeline with the same seed.
std::size_t config_fingerprint(const thesis::ThesisConfig& cfg)
{
    std::ostringstream s;
    s << cfg.experiment.run_tag << '|' << cfg.experiment.seed << '|' << cfg.experiment.repeats
      << '|' << cfg.dataset.modality << '|' << cfg.dataset.fusion_mode << '|'
      << cfg.feature_extraction.strategy << '|' << cfg.classifier.enabled << '|'
      << cfg.classifier.type << '|' << cfg.classifier.text_mode << "|layers:";
    for (const auto& l : cfg.classifier.layer_spec) s << l << ',';
    s << '|' << cfg.training.optimizer_type << '|' << cfg.training.effective_learning_rate() << '|'
      << cfg.training.epochs << '|' << cfg.training.samples_per_batch << '|'
      << cfg.training.weight_decay << '|' << cfg.training.k_folds << '|' << cfg.training.nested_cv;
    const auto& hc = cfg.feature_extraction.handcrafted;
    s << "|hc:" << hc.wavelet << ',' << hc.scale << ',' << hc.cepstral << ',' << hc.dtwpt_level;
    const auto& ae = cfg.feature_extraction.autoencoder;
    s << "|ae:" << ae.model << ',' << ae.encoding << ',' << ae.time_steps << ','
      << ae.voltage_threshold;
    return std::hash<std::string>{}(s.str());
}

// Run one full pipeline iteration with the given config (seed + run_tag already set).
// `view` is loaded once by the caller — dataset loading is seed-independent, so
// repeats share it. `cached_features` is non-null when feature extraction is also
// seed-independent (handcrafted strategy); AE strategies pass null and re-extract
// per repeat because AE training depends on the repeat's seed.
void run_once(const thesis::ThesisConfig& cfg,
    const thesis::ThesisDatasetView& view,
    const std::vector<thesis::FeatureSet>* cached_features)
{
    auto& pm = nn::progress::ProgressManager::instance();

    // ── 3. Feature extraction ────────────────────────────────────────────────
    std::vector<thesis::FeatureSet> extracted;
    if (cached_features == nullptr)
    {
        extracted = thesis::extract_features(view,
            cfg.feature_extraction,
            cfg.training,
            cfg.dataset.modality,
            cfg.dataset.fusion_mode,
            cfg.experiment.seed);
    }
    const auto& feature_sets = (cached_features != nullptr) ? *cached_features : extracted;
    pm.log("[Thesis] Extracted " + std::to_string(feature_sets.size()) + " feature set(s).");

    // ── 4. Paraconsistent ranking ────────────────────────────────────────────
    std::vector<thesis::ParaconsistentScore> scores;
    if (cfg.paraconsistent.enabled)
    {
        scores = thesis::rank_feature_sets(view.samples, feature_sets);
        pm.log("[Thesis] Paraconsistent ranking:");
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
    std::vector<thesis::ClassificationResult> results;
    if (cfg.classifier.enabled)
    {
        int n_usable_fs = 0;
        for (const auto& fs : feature_sets)
            if (!fs.vectors.empty()) ++n_usable_fs;

        const int total_outer_folds = n_usable_fs * cfg.training.k_folds;
        const uint32_t global_bar = pm.create_bar(
            "Thesis | " + cfg.experiment.run_tag, static_cast<float>(total_outer_folds));
        pm.set_description(global_bar,
            cfg.classifier.type + " | " + cfg.classifier.text_mode + " | " +
                std::to_string(cfg.training.k_folds) + "-fold CV");

        int global_completed = 0;
        for (const auto& fs : feature_sets)
        {
            if (fs.vectors.empty()) continue;
            auto result = thesis::run_classifier(
                view, fs.vectors, fs.label, cfg, nullptr, global_bar, &global_completed);
            results.push_back(std::move(result));
        }

        pm.complete_bar(global_bar);
    }
    else
    {
        pm.log("[Thesis] classifier.enabled=false — Phase 00 run, stopping after ranking.");
    }

    // ── 6. Output ────────────────────────────────────────────────────────────
    const std::string& results_dir = cfg.dataset.results_dir;
    const std::string& tag = cfg.experiment.run_tag;

    // Ranking artefacts are always written; classifier artefacts only when it ran.
    thesis::write_paraconsistent_csv(results_dir, tag, scores);
    thesis::write_summary_json(results_dir,
        tag,
        cfg,
        results,
        scores,
        view.n_subjects,
        view.n_stimuli,
        view.samples.size(),
        config_fingerprint(cfg));
    if (cfg.classifier.enabled)
    {
        thesis::write_metrics_csv(results_dir, tag, results);
        thesis::write_comparison_dat(results_dir, tag, results);
        thesis::write_learning_curves_dat(
            results_dir, tag, results); // Guayaquil epoch-history analog
    }

    for (const auto& r : results)
    {
        std::ostringstream oss;
        oss << "[Thesis] " << r.feature_set_label << "  acc=" << r.mean_accuracy << " ±"
            << r.std_accuracy << "  EER=" << r.mean_eer << "  spec=" << r.mean_specificity;
        pm.log(oss.str());
    }
    pm.log("[Thesis] Done. Results written to " + results_dir);
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
        auto cfg = thesis::ThesisConfig::from_file(config_path);
        cfg.validate();

        // Overall multi-profile banner. Each profile is a separate process and cannot know the
        // whole-run progress on its own, so the runner (run_thesis_profiles.sh) computes it and
        // hands the ready-made line in via THESIS_OVERALL. Logged first, it renders as a persistent
        // top line above the per-profile bars, mirroring the Guayaquil (Guayaquil) TUI. Empty/unset
        // when run standalone, so the TUI is unchanged.
        if (const char* overall = std::getenv("THESIS_OVERALL");
            overall != nullptr && overall[0] != '\0')
        {
            nn::progress::ProgressManager::instance().log(std::string(overall));
        }

        std::cout << "[Thesis] run_tag=" << cfg.experiment.run_tag
                  << " modality=" << cfg.dataset.modality
                  << " strategy=" << cfg.feature_extraction.strategy
                  << " repeats=" << cfg.experiment.repeats << "\n";

        // ── 2. Load dataset (once — identical across repeats) ────────────────
        auto view = thesis::load_dataset(cfg.dataset);
        {
            std::ostringstream oss;
            oss << "[Thesis] Loaded " << view.samples.size() << " samples from " << view.n_subjects
                << " subjects, " << view.n_stimuli << " stimuli.";
            nn::progress::ProgressManager::instance().log(oss.str());
        }

        // Handcrafted features are deterministic (no seed involved), so extract
        // once and share across repeats. AE-based extraction trains a network per
        // repeat seed and must stay inside the repeat loop.
        std::vector<thesis::FeatureSet> shared_features;
        const bool features_seed_independent = (cfg.feature_extraction.strategy == "handcrafted");
        if (features_seed_independent)
            shared_features = thesis::extract_features(view,
                cfg.feature_extraction,
                cfg.training,
                cfg.dataset.modality,
                cfg.dataset.fusion_mode,
                cfg.experiment.seed);

        // ── Repeat loop ───────────────────────────────────────────────────────
        for (int rep = 0; rep < cfg.experiment.repeats; ++rep)
        {
            thesis::ThesisConfig rep_cfg = cfg;

            // Each repeat uses a different seed unless seed_deterministic=true.
            if (!rep_cfg.experiment.seed_deterministic)
                rep_cfg.experiment.seed += static_cast<uint32_t>(rep);

            // Append repeat index to run_tag so output files don't overwrite.
            if (cfg.experiment.repeats > 1)
                rep_cfg.experiment.run_tag = cfg.experiment.run_tag + "_rep" + std::to_string(rep);

            if (cfg.experiment.repeats > 1)
            {
                std::ostringstream oss;
                oss << "[Thesis] === Repeat " << rep + 1 << "/" << cfg.experiment.repeats
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
        std::cerr << "[Thesis] Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
