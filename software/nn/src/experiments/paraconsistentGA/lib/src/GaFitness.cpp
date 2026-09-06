#include "GaFitness.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "ThesisConfig.hpp"
#include "ThesisFeatureExtraction.hpp"
#include "ThesisParaconsistent.hpp"

namespace pga
{

namespace
{
// Mean per-dimension standard deviation of a set of latent vectors. Near-zero means
// the latent has collapsed (every sample maps to the same point) — the α=β=1
// degeneracy the reconstruction sanity filter defends against
// (.wiki/Experiments/ParaconsistentGA-Design.md §3.3).
double mean_latent_std(const std::vector<std::vector<double>>& vectors)
{
    if (vectors.empty() || vectors[0].empty()) return 0.0;
    const size_t n = vectors.size();
    const size_t dim = vectors[0].size();

    double acc_std = 0.0;
    for (size_t d = 0; d < dim; ++d)
    {
        double mean = 0.0;
        for (const auto& v : vectors)
            if (d < v.size()) mean += v[d];
        mean /= static_cast<double>(n);

        double var = 0.0;
        for (const auto& v : vectors)
            if (d < v.size())
            {
                const double e = v[d] - mean;
                var += e * e;
            }
        var /= static_cast<double>(n);
        acc_std += std::sqrt(var);
    }
    return acc_std / static_cast<double>(dim);
}

// Build a thesis FeatureExtraction that runs this genome's AE.
thesis::ThesisConfig::FeatureExtraction make_feature_extraction(
    const Genome& g, const GaConfig& cfg)
{
    thesis::ThesisConfig::FeatureExtraction fe = cfg.base.feature_extraction;
    fe.strategy = "autoencoder";
    fe.autoencoder = to_ae_config(g, cfg.base.feature_extraction.autoencoder);
    return fe;
}

// Winning (lowest d_penalized) representation found across the n_seeds training/scoring
// sweep, plus the per-seed d_penalized series used to report mean/std.
struct SeedSweepResult
{
    std::vector<double> d_pen_per_seed;
    thesis::ParaconsistentScore best_score;
    double best_latent_activity = 0.0;
    int best_seed_offset = -1;
    std::map<std::string, nn::Tensor> best_weights;
};

// Trains + scores this genome's feature extraction across cfg.ga.n_seeds independent seeds
// (.wiki/Experiments/ParaconsistentGA-Design.md §5.5), keeping the best (lowest d_penalized)
// feature set/model snapshot seen across all seeds.
SeedSweepResult run_seed_sweep(const thesis::ThesisDatasetView& view,
    const thesis::ThesisConfig::FeatureExtraction& fe,
    const std::string& modality,
    const std::string& fusion_mode,
    const GaConfig& cfg)
{
    SeedSweepResult result;
    result.d_pen_per_seed.reserve(static_cast<size_t>(cfg.ga.n_seeds));
    double best_d_pen = std::numeric_limits<double>::max();

    for (int s = 0; s < cfg.ga.n_seeds; ++s)
    {
        const std::uint32_t seed = cfg.base.experiment.seed + static_cast<std::uint32_t>(s);

        // Captures every model this seed trains (one, or two for late fusion — see
        // ModelSnapshotFn), prefixed by part_tag so late fusion's voice+eeg models don't
        // collide. Only kept in `best_weights` if this seed turns out to be the winner.
        std::map<std::string, nn::Tensor> captured_this_seed;
        auto on_model_trained =
            [&](const std::string& part_tag, const std::map<std::string, nn::Tensor>& sd)
        {
            const std::string prefix = part_tag.empty() ? "" : part_tag + ".";
            for (const auto& [k, v] : sd) captured_this_seed[prefix + k] = v;
        };

        auto feature_sets = thesis::extract_features(
            view, fe, cfg.base.training, modality, fusion_mode, seed, on_model_trained);

        // Score every non-empty feature set this genome produced; the individual's
        // value for the seed is the best (lowest d_penalized) representation.
        double seed_best = std::numeric_limits<double>::max();
        for (const auto& fs : feature_sets)
        {
            if (fs.vectors.empty()) continue;
            const auto score = thesis::score_feature_set(view.samples, fs);
            if (score.d_penalized < seed_best)
            {
                seed_best = score.d_penalized;
                if (score.d_penalized < best_d_pen)
                {
                    best_d_pen = score.d_penalized;
                    result.best_score = score;
                    result.best_latent_activity = mean_latent_std(fs.vectors);
                    result.best_seed_offset = s;
                    result.best_weights = captured_this_seed;
                }
            }
        }

        if (seed_best != std::numeric_limits<double>::max())
            result.d_pen_per_seed.push_back(seed_best);
    }

    return result;
}

// Mean and (population) standard deviation of `values`.
std::pair<double, double> mean_and_std(const std::vector<double>& values)
{
    double mean = 0.0;
    for (double v : values) mean += v;
    mean /= static_cast<double>(values.size());
    double var = 0.0;
    for (double v : values) var += (v - mean) * (v - mean);
    var /= static_cast<double>(values.size());
    return {mean, std::sqrt(var)};
}

// Marks `ind` infeasible and accumulates the constraint-violation magnitude for any
// constraint it breaches (latency ceiling, minimum latent activity).
void apply_feasibility_constraints(Individual& ind, const GaConfig& cfg)
{
    ind.feasible = true;
    ind.constraint_violation = 0.0;

    if (ind.est_latency_ms > cfg.constraints.latency_ceiling_ms)
    {
        ind.feasible = false;
        ind.constraint_violation += (ind.est_latency_ms - cfg.constraints.latency_ceiling_ms) /
                                    std::max(1.0, cfg.constraints.latency_ceiling_ms);
    }
    if (ind.latent_activity < cfg.constraints.tau_rec)
    {
        ind.feasible = false;
        const double denom = std::max(cfg.constraints.tau_rec, 1e-12);
        ind.constraint_violation += (cfg.constraints.tau_rec - ind.latent_activity) / denom;
    }
}

} // namespace

bool exceeds_latency_budget(const Genome& g, const GaConfig& cfg)
{
    const double est_ae_ms =
        static_cast<double>(inference_cost_proxy(g)) * cfg.constraints.ns_per_mac / 1e6;
    return est_ae_ms > cfg.autoencoder_budget_ms();
}

void evaluate_individual(
    Individual& ind, const thesis::ThesisDatasetView& view, const GaConfig& cfg)
{
    const Genome& g = ind.genome;

    ind.param_count = estimated_params(g);
    ind.inference_cost = inference_cost_proxy(g);
    ind.est_latency_ms = cfg.constraints.fixed_pipeline_cost_ms +
                         static_cast<double>(ind.inference_cost) * cfg.constraints.ns_per_mac / 1e6;

    // ── Cheap pre-training screen (.wiki/Experiments/ParaconsistentGA-Design.md §4)
    // ───────────────────────────────── Over-budget genomes are discarded without training: mark
    // infeasible, give a worst-case sentinel objective, skip the expensive extract/score entirely.
    if (exceeds_latency_budget(g, cfg))
    {
        ind.d_penalized_mean = 2.0; // worst-vertex sentinel; unused under constrained dominance
        ind.d_penalized_std = 0.0;
        ind.feasible = false;
        ind.constraint_violation = (ind.est_latency_ms - cfg.constraints.latency_ceiling_ms) /
                                   std::max(1.0, cfg.constraints.latency_ceiling_ms);
        ind.objectives = {ind.d_penalized_mean, static_cast<double>(ind.inference_cost)};
        ind.evaluated = true;
        return;
    }

    // ── Train + score across n_seeds (.wiki/Experiments/ParaconsistentGA-Design.md §5.5)
    // ────────────────────────────
    const thesis::ThesisConfig::FeatureExtraction fe = make_feature_extraction(g, cfg);
    const std::string& modality = cfg.base.dataset.modality;
    const std::string& fusion_mode = cfg.base.dataset.fusion_mode;

    SeedSweepResult sweep = run_seed_sweep(view, fe, modality, fusion_mode, cfg);

    if (sweep.d_pen_per_seed.empty())
        throw std::runtime_error(
            "GaFitness: no usable feature set was produced for this genome across any of "
            "the " +
            std::to_string(cfg.ga.n_seeds) +
            " seeds. Scoring it as merely 'infeasible' would hide a broken extraction "
            "behind a normal-looking GA result.");

    // Mean + std of d_penalized across seeds.
    const auto [mean, std_dev] = mean_and_std(sweep.d_pen_per_seed);

    ind.d_penalized_mean = mean;
    ind.d_penalized_std = std_dev;
    ind.alpha = sweep.best_score.alpha;
    ind.beta = sweep.best_score.beta;
    ind.g1 = sweep.best_score.g1;
    ind.g2 = sweep.best_score.g2;
    ind.d_truth = sweep.best_score.d_truth;
    ind.latent_activity = sweep.best_latent_activity;
    ind.winning_seed_offset = sweep.best_seed_offset;
    if (!sweep.best_weights.empty()) ind.weights_snapshot = std::move(sweep.best_weights);

    // ── Feasibility (constrained dominance) ──────────────────────────────────
    apply_feasibility_constraints(ind, cfg);

    ind.objectives = {ind.d_penalized_mean, static_cast<double>(ind.inference_cost)};
    ind.evaluated = true;
}

} // namespace pga
