/**
 * @file src/experiments/thesis/thesis.cpp
 * @brief Experiment05 entry point: biometric authentication of severely dysphonic speakers.
 *
 * Usage:
 *   thesis --config <profile.json> [--repeat <k>]
 *
 * --repeat k runs only repetition k, exactly as the full run would (seed, output names); the
 * GridUnesp array runs Phase 01 as one such process per (profile, repetition). ThesisCli.hpp.
 */

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "lib/include/ParaconsistentSubsets.hpp"
#include "lib/include/ThesisClassifiers.hpp"
#include "lib/include/ThesisCli.hpp"
#include "lib/include/ThesisConfig.hpp"
#include "lib/include/ThesisDataset.hpp"
#include "lib/include/ThesisFeatureExtraction.hpp"
#include "lib/include/ThesisOutput.hpp"
#include "lib/include/ThesisParaconsistent.hpp"
#include "lib/include/VoiceDegradation.hpp"
#include "progress/ProgressManager.hpp"
#include "statistics/GenuineImpostorEERScorer.hpp"
#include "utility/progress.hpp"

namespace
{
// Provenance/determinism fingerprint of the run: a hash of every declared key (the
// Meeting01 config_hash analog). ThesisConfig refuses undeclared keys the run reads and
// declared keys it ignores, so the declared keys are exactly the run's configuration —
// two runs with the same hash trained the same model on the same pipeline with the same
// seed, and a key that changes nothing cannot change the hash.
std::size_t config_fingerprint(const thesis::ThesisConfig& cfg)
{
    return std::hash<std::string>{}(cfg.declared_json().dump());
}

// Run one full pipeline iteration with the given config (seed + run_tag already set).
// `view` is loaded once by the caller — dataset loading is seed-independent, so
// repeats share it. `cached_features` is non-null when feature extraction is also
// seed-independent (handcrafted strategy); AE strategies pass null and re-extract
// per repeat because AE training depends on the repeat's seed.
// Verification protocol: per test speaker, the first utterance is the enrolment
// template and the rest are probes scored against every template (cosine).
constexpr std::size_t kEnrolUtterancesPerSpeaker = 1;

// The voice of every sample whispered by degradation.level (VoiceDegradation.hpp), in place.
// One random stream per sample (seed mixed with its index), so the result does not depend
// on the thread schedule.
void degrade_voices(thesis::ThesisDatasetView& view, const thesis::ThesisConfig::Degradation& dg)
{
    const double voice_rate_hz = thesis::require_rate(view.voice_sample_rate, "voice");
    const auto n = static_cast<long>(view.samples.size());
#pragma omp parallel for schedule(dynamic, 8)
    for (long i = 0; i < n; ++i)
    {
        nn::Tensor& audio = view.samples[static_cast<std::size_t>(i)].audio;
        const float* src = audio.data_ptr();
        const std::vector<float> x(src, src + audio.size());
        const std::uint32_t seed = dg.seed.get() ^ (static_cast<std::uint32_t>(i) * 2654435761U);
        const std::vector<float> y = thesis::degrade_voice(x, voice_rate_hz, dg.level, seed);
        std::copy(y.begin(), y.end(), audio.mutable_data_ptr());
    }
}

void run_once(const thesis::ThesisConfig& cfg,
    const thesis::ThesisDatasetView& view,
    const std::vector<thesis::FeatureSet>* cached_features,
    const std::vector<thesis::FeatureSet>* degraded_features)
{
    auto& pm = nn::progress::ProgressManager::instance();

    // ── 3. Feature extraction ────────────────────────────────────────────────
    std::vector<thesis::FeatureSet> extracted;
    if (cached_features == nullptr)
    {
        extracted = thesis::extract_features(view, cfg, /*on_model_trained=*/nullptr);
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
        if (cfg.paraconsistent.speaker_subsets)
            thesis::write_subsets_csv(cfg.dataset.results_dir,
                cfg.experiment.run_tag,
                thesis::score_speaker_subsets(view.samples,
                    feature_sets,
                    cfg.paraconsistent.fold_k,
                    cfg.paraconsistent.fold_seed,
                    cfg.paraconsistent.fold_repeats));
    }

    // ── 5. Classification (Phase 01 only) ────────────────────────────────────
    // When classifier.enabled is false, this is a Phase 00 run: stop after
    // paraconsistent ranking and emit only the ranking artefacts.
    std::vector<thesis::ClassificationResult> results;
    if (cfg.classifier.enabled)
    {
        const int total_outer_folds = static_cast<int>(feature_sets.size()) * cfg.training.k_folds;
        const uint32_t global_bar = pm.create_bar(
            "Thesis | " + cfg.experiment.run_tag, static_cast<float>(total_outer_folds));
        pm.set_description(global_bar,
            cfg.classifier.type + " | " + cfg.classifier.text_mode + " | " +
                std::to_string(cfg.training.k_folds) + "-fold CV");

        const statistics::GenuineImpostorEERScorer scorer(kEnrolUtterancesPerSpeaker);
        if (degraded_features != nullptr && degraded_features->size() != feature_sets.size())
            throw std::logic_error("thesis: degraded and clean feature sets differ in number");

        int global_completed = 0;
        for (const auto& fs : feature_sets)
        {
            // An extractor that produced nothing used to be skipped here, leaving the
            // run one feature set short with no sign of it.
            if (fs.vectors.empty())
                throw std::runtime_error(
                    "thesis: feature set \"" + fs.label + "\" has no vectors; nothing to classify");
            const std::size_t fs_index = static_cast<std::size_t>(&fs - feature_sets.data());
            std::optional<thesis::DegradedTestFeatures> degraded;
            if (degraded_features != nullptr)
            {
                const auto& dfs = (*degraded_features)[fs_index];
                if (dfs.label != fs.label)
                    throw std::logic_error("thesis: degraded feature set \"" + dfs.label +
                                           "\" is not the twin of \"" + fs.label + "\"");
                degraded = thesis::DegradedTestFeatures{&dfs.vectors,
                    cfg.degradation.scope.get() == "probes",
                    kEnrolUtterancesPerSpeaker};
            }
            auto result = thesis::run_classifier(view,
                fs.vectors,
                fs.label,
                cfg,
                scorer,
                global_bar,
                &global_completed,
                degraded ? &*degraded : nullptr);
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
            results_dir, tag, results); // Meeting01 epoch-history analog
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
        const thesis::ThesisCli cli = thesis::parse_cli(argc, argv);

        // ── 1. Load and validate profile ─────────────────────────────────────
        auto cfg = thesis::ThesisConfig::from_file(cli.config_path);
        cfg.validate();
        const thesis::RepeatRange repeats = thesis::repeat_range(cli, cfg.experiment.repeats);

        // Overall multi-profile banner. Each profile is a separate process and cannot know the
        // whole-run progress on its own, so the runner (run_thesis_profiles.sh) computes it and
        // hands the ready-made line in via THESIS_OVERALL. Logged first, it renders as a persistent
        // top line above the per-profile bars, mirroring the Meeting01 (Meeting01) TUI. Empty/unset
        // when run standalone, so the TUI is unchanged.
        if (const char* overall = std::getenv("THESIS_OVERALL");
            overall != nullptr && overall[0] != '\0')
        {
            nn::progress::ProgressManager::instance().log(std::string(overall));
        }

        std::cout << "[Thesis] run_tag=" << cfg.experiment.run_tag
                  << " modality=" << cfg.dataset.modality
                  << " strategy=" << cfg.feature_extraction.strategy
                  << " repeats=" << cfg.experiment.repeats << " running=[" << repeats.first << ","
                  << repeats.last << ")\n";

        // ── 2. Load dataset (once — identical across repeats) ────────────────
        auto view = thesis::load_dataset(cfg.dataset);
        {
            std::ostringstream oss;
            oss << "[Thesis] Loaded " << view.samples.size() << " samples from " << view.n_subjects
                << " subjects, " << view.n_stimuli << " stimuli.";
            nn::progress::ProgressManager::instance().log(oss.str());
        }

        // Handcrafted and PCA features are deterministic (no seed involved), so extract
        // once and share across repeats. AE-based extraction trains a network per
        // repeat seed and must stay inside the repeat loop.
        std::vector<thesis::FeatureSet> shared_features;
        const bool features_seed_independent = (cfg.feature_extraction.strategy != "autoencoder");
        if (features_seed_independent)
            shared_features = thesis::extract_features(view, cfg, /*on_model_trained=*/nullptr);

        // Simulated dysphonia: the same extractor over the degraded voices. The clean
        // features are taken first; the view's audio is then replaced (nothing after this
        // reads the clean audio). ThesisConfig refuses degradation with an autoencoder.
        std::optional<std::vector<thesis::FeatureSet>> degraded_features;
        if (cfg.degradation.enabled.declared() && cfg.degradation.enabled)
        {
            degrade_voices(view, cfg.degradation);
            degraded_features = thesis::extract_features(view, cfg, /*on_model_trained=*/nullptr);
        }

        // ── Repeat loop ───────────────────────────────────────────────────────
        for (int rep = repeats.first; rep < repeats.last; ++rep)
        {
            thesis::ThesisConfig rep_cfg = cfg;

            // Each repeat uses a different seed unless seed_deterministic=true. The key
            // exists only for repeats > 1 (one run has nothing to decide).
            if (cfg.experiment.repeats > 1 && !rep_cfg.experiment.seed_deterministic)
                rep_cfg.experiment.seed = rep_cfg.experiment.seed + static_cast<uint32_t>(rep);

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

            run_once(rep_cfg,
                view,
                features_seed_independent ? &shared_features : nullptr,
                degraded_features ? &*degraded_features : nullptr);
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
