/**
 * Unit tests for the paraconsistentGA experiment. Covers the dataset-independent
 * acceptance criteria from ga.md §7: d_penalized reference reproduction, constant
 * output ranked worst, NSGA-II constrained dominance / crowding correctness, plus
 * genome mapping and config validation.
 */
#include <cmath>
#include <limits>
#include <vector>

#include "ThesisDataset.hpp"
#include "ThesisFeatureExtraction.hpp"
#include "ThesisParaconsistent.hpp"
#include "gtest/gtest.h"
#include "lib/include/GaConfig.hpp"
#include "lib/include/GaFitness.hpp"
#include "lib/include/GaGenome.hpp"
#include "lib/include/GaNsga2.hpp"

namespace
{
using pga::Genome;
using pga::Individual;

Individual make_ind(std::vector<double> objectives, bool feasible, double violation = 0.0)
{
    Individual ind;
    ind.objectives = std::move(objectives);
    ind.feasible = feasible;
    ind.constraint_violation = violation;
    ind.evaluated = true;
    return ind;
}

// d_penalized computed straight from (alpha, beta) via the real thesis constant.
double d_penalized_from(double alpha, double beta)
{
    const double g1 = alpha - beta;
    const double g2 = alpha + beta - 1.0;
    const double d_truth = std::sqrt((g1 - 1.0) * (g1 - 1.0) + g2 * g2);
    return d_truth + thesis::kContradictionPenalty * std::abs(g2);
}
} // namespace

// ── ga.md §7: d_penalized reference cases ────────────────────────────────────
TEST(PgaParaconsistent, ReferenceCaseAmbiguityVertex)
{
    // (alpha,beta) = (1,1): the Ambiguity vertex. kContradictionPenalty = 2-sqrt(2)
    // is chosen so every non-Truth vertex scores exactly 2.0 (NOT 2.4142).
    EXPECT_NEAR(d_penalized_from(1.0, 1.0), 2.0, 1e-4);
}

TEST(PgaParaconsistent, ReferenceCaseGoodFeatures)
{
    // (alpha,beta) = (0.92, 0.075) → ~0.1580 (ga.md §7).
    EXPECT_NEAR(d_penalized_from(0.92, 0.075), 0.1580, 1e-3);
}

// ── ga.md §7: constant output must rank worst, not best ──────────────────────
TEST(PgaParaconsistent, ConstantLatentRanksWorst)
{
    const int n_subjects = 4;
    const int per_subject = 5;

    std::vector<thesis::ThesisSample> samples;
    for (int s = 0; s < n_subjects; ++s)
        for (int k = 0; k < per_subject; ++k)
        {
            thesis::ThesisSample smp;
            smp.subject_id = s;
            samples.push_back(smp);
        }

    // Informative: each subject clusters around a distinct point with tiny jitter.
    thesis::FeatureSet informative;
    informative.label = "informative";
    // Constant: every sample identical → collapsed latent (alpha=beta=1 degeneracy).
    thesis::FeatureSet constant;
    constant.label = "constant";

    for (int s = 0; s < n_subjects; ++s)
        for (int k = 0; k < per_subject; ++k)
        {
            const double base = static_cast<double>(s);
            const double jit = 0.01 * k;
            informative.vectors.push_back({base + jit, base * 0.5 + jit, base * 0.25 + jit});
            constant.vectors.push_back({0.5, 0.5, 0.5});
        }

    const auto s_info = thesis::score_feature_set(samples, informative);
    const auto s_const = thesis::score_feature_set(samples, constant);

    EXPECT_NEAR(s_const.d_penalized, 2.0, 1e-4);        // degenerate vertex
    EXPECT_LT(s_info.d_penalized, s_const.d_penalized); // informative strictly better
}

// ── NSGA-II constrained dominance (ga.md §3.4) ───────────────────────────────
TEST(PgaNsga2, FeasibleDominatesInfeasible)
{
    // Infeasible individual has better objectives but must NOT dominate the feasible one.
    Individual feasible = make_ind({1.0, 1.0}, true);
    Individual infeasible = make_ind({0.0, 0.0}, false, 0.5);
    EXPECT_TRUE(pga::constrained_dominates(feasible, infeasible));
    EXPECT_FALSE(pga::constrained_dominates(infeasible, feasible));
}

TEST(PgaNsga2, InfeasibleOrderedByViolation)
{
    Individual less = make_ind({9.0, 9.0}, false, 0.1);
    Individual more = make_ind({0.0, 0.0}, false, 0.9);
    EXPECT_TRUE(pga::constrained_dominates(less, more));
    EXPECT_FALSE(pga::constrained_dominates(more, less));
}

TEST(PgaNsga2, ParetoDominanceBetweenFeasible)
{
    Individual a = make_ind({1.0, 1.0}, true);
    Individual b = make_ind({2.0, 2.0}, true);
    Individual c =
        make_ind({1.0, 3.0}, true); // non-dominated vs a (worse obj1, equal obj0? no, equal obj0)
    EXPECT_TRUE(pga::constrained_dominates(a, b));
    EXPECT_FALSE(pga::constrained_dominates(b, a));
    // a has equal obj0 and better obj1 → a dominates c.
    EXPECT_TRUE(pga::constrained_dominates(a, c));
}

TEST(PgaNsga2, NonDominatedSortFronts)
{
    std::vector<Individual> pop = {
        make_ind({1.0, 4.0}, true), // A
        make_ind({2.0, 3.0}, true), // B  (A,B mutually non-dominated → front 0)
        make_ind({3.0, 3.0}, true), // C  dominated by B → front 1
        make_ind({5.0, 5.0}, true), // D  dominated by all → front 2
    };
    auto fronts = pga::fast_non_dominated_sort(pop);
    ASSERT_EQ(fronts.size(), 3u);
    EXPECT_EQ(fronts[0].size(), 2u);
    EXPECT_EQ(fronts[1].size(), 1u);
    EXPECT_EQ(fronts[2].size(), 1u);
    EXPECT_EQ(pop[0].rank, 0);
    EXPECT_EQ(pop[3].rank, 2);
}

TEST(PgaNsga2, CrowdingBoundariesInfinite)
{
    std::vector<Individual> pop = {
        make_ind({1.0, 3.0}, true),
        make_ind({2.0, 2.0}, true),
        make_ind({3.0, 1.0}, true),
    };
    std::vector<int> front = {0, 1, 2};
    pga::assign_crowding_distance(pop, front);
    // Two extremes per objective get +inf; the interior point stays finite.
    int n_inf = 0;
    for (const auto& ind : pop)
        if (ind.crowding == std::numeric_limits<double>::infinity()) ++n_inf;
    EXPECT_EQ(n_inf, 2);
    EXPECT_TRUE(std::isfinite(pop[1].crowding));
    EXPECT_GT(pop[1].crowding, 0.0);
}

// ── Genome → AE config mapping (ga.md §5.1) ──────────────────────────────────
TEST(PgaGenome, SnnMappingCouplesTemporalToEncoding)
{
    Genome g;
    g.hidden = 64;
    g.latent = 32;
    g.encoding = "latency";
    pga::apply_phase00_temporal_coupling(g);
    EXPECT_EQ(g.time_steps, 16);
    EXPECT_FLOAT_EQ(g.voltage_threshold, 0.2f);

    auto ae = pga::to_ae_config(g, "snn-ae");
    EXPECT_EQ(ae.model, "snn-ae");
    ASSERT_EQ(ae.encoder_layer_spec.size(), 2u);
    EXPECT_EQ(ae.encoder_layer_spec[0], "linear:64:leaky");
    EXPECT_EQ(ae.encoder_layer_spec[1], "linear:32:identity");
    EXPECT_EQ(ae.decoder_layer_spec.back(), "linear:output:identity");
    EXPECT_EQ(ae.encoding, "latency");
    EXPECT_FLOAT_EQ(ae.firing_rate_reg_lambda, 0.5f);
}

TEST(PgaGenome, AnnMappingForcesDirect)
{
    Genome g;
    g.hidden = 32;
    g.latent = 16;
    g.encoding = "poisson"; // must be ignored for ann-ae
    auto ae = pga::to_ae_config(g, "ann-ae");
    EXPECT_EQ(ae.encoding, "direct");
    EXPECT_EQ(ae.time_steps, 1);
    EXPECT_FLOAT_EQ(ae.voltage_threshold, 1.0f);
}

TEST(PgaGenome, ProxiesMonotonic)
{
    Genome small;
    small.hidden = 16;
    small.latent = 8;
    Genome big;
    big.hidden = 128;
    big.latent = 64;
    EXPECT_LT(pga::estimated_params(small), pga::estimated_params(big));
    EXPECT_LT(pga::inference_cost_proxy(small), pga::inference_cost_proxy(big));
}

TEST(PgaGenome, RandomGenomeKeepsBottleneck)
{
    std::mt19937 rng(123);
    pga::GenomeBounds bounds;
    for (int i = 0; i < 200; ++i)
    {
        Genome g = pga::random_genome(rng, bounds, /*is_snn=*/true);
        EXPECT_LT(g.latent, g.hidden) << "latent must stay below hidden (bottleneck)";
    }
}

// ── Latency pre-screen (ga.md §4) ────────────────────────────────────────────
TEST(PgaFitness, LatencyPreScreenRejectsOverBudget)
{
    pga::GaConfig cfg;
    cfg.constraints.latency_ceiling_ms = 0.001; // absurdly tight → everything over budget
    cfg.constraints.fixed_pipeline_cost_ms = 0.0;
    cfg.constraints.ns_per_mac = 2.0;
    Genome g;
    g.hidden = 128;
    g.latent = 64;
    g.time_steps = 16;
    EXPECT_TRUE(pga::exceeds_latency_budget(g, cfg));

    cfg.constraints.latency_ceiling_ms = 1e9; // huge budget → nothing over budget
    EXPECT_FALSE(pga::exceeds_latency_budget(g, cfg));
}

// ── Config validation ────────────────────────────────────────────────────────
TEST(PgaConfig, RejectsHandcraftedStrategy)
{
    pga::GaConfig cfg;
    cfg.base.experiment.run_tag = "t";
    cfg.base.dataset.root = "/x";
    cfg.base.feature_extraction.strategy = "handcrafted";
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(PgaConfig, AcceptsValidAutoencoderConfig)
{
    pga::GaConfig cfg;
    cfg.base.experiment.run_tag = "t";
    cfg.base.dataset.root = "/x";
    cfg.base.dataset.modality = "eeg";
    cfg.base.feature_extraction.strategy = "autoencoder";
    cfg.base.feature_extraction.autoencoder.model = "snn-ae";
    cfg.base.feature_extraction.autoencoder.encoder_layer_spec = {
        "linear:64:leaky", "linear:32:identity"};
    cfg.base.feature_extraction.autoencoder.decoder_layer_spec = {
        "linear:64:leaky", "linear:output:identity"};
    cfg.base.classifier.enabled = false; // as GaConfig::from_json forces (Phase-00-like run)
    cfg.run_tag = "t";
    EXPECT_NO_THROW(cfg.validate());
}
