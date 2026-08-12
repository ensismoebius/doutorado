/**
 * Unit tests for the paraconsistentGA experiment. Covers the dataset-independent
 * acceptance criteria from .wiki/Experiments/ParaconsistentGA-Design.md §7: d_penalized reference
 * reproduction, constant output ranked worst, NSGA-II constrained dominance / crowding correctness,
 * plus genome mapping and config validation.
 */
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include "ThesisDataset.hpp"
#include "ThesisFeatureExtraction.hpp"
#include "ThesisParaconsistent.hpp"
#include "gtest/gtest.h"
#include "lib/include/GaCheckpoint.hpp"
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

// ── .wiki/Experiments/ParaconsistentGA-Design.md §7: d_penalized reference cases
// ────────────────────────────────────
TEST(PgaParaconsistent, ReferenceCaseAmbiguityVertex)
{
    // (alpha,beta) = (1,1): the Ambiguity vertex. kContradictionPenalty = 2-sqrt(2)
    // is chosen so every non-Truth vertex scores exactly 2.0 (NOT 2.4142).
    EXPECT_NEAR(d_penalized_from(1.0, 1.0), 2.0, 1e-4);
}

TEST(PgaParaconsistent, ReferenceCaseGoodFeatures)
{
    // (alpha,beta) = (0.92, 0.075) → ~0.1580 (.wiki/Experiments/ParaconsistentGA-Design.md §7).
    EXPECT_NEAR(d_penalized_from(0.92, 0.075), 0.1580, 1e-3);
}

// ── .wiki/Experiments/ParaconsistentGA-Design.md §7: constant output must rank worst, not best
// ──────────────────────
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

// ── NSGA-II constrained dominance (.wiki/Experiments/ParaconsistentGA-Design.md §3.4)
// ───────────────────────────────
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

namespace
{
// A profile-shaped base config, as ThesisConfig::from_json would produce.
thesis::ThesisConfig::AutoencoderConfig base_ae(const char* model, const char* loss)
{
    thesis::ThesisConfig::AutoencoderConfig ae;
    ae.model = model;
    ae.ae_loss_type = loss;
    ae.firing_rate_reg_lambda = 0.5f;
    ae.firing_rate_min = 0.1f;
    ae.firing_rate_max = 0.8f;
    return ae;
}
} // namespace

// ── Genome → AE config mapping (.wiki/Experiments/ParaconsistentGA-Design.md §5.1)
// ──────────────────────────────────
TEST(PgaGenome, SnnMappingCouplesTemporalToEncoding)
{
    Genome g;
    g.encoder_widths = {10, 5, 4, 2}; // free depth + per-layer widths from the DNA
    g.encoding = "latency";
    pga::apply_phase00_temporal_coupling(g);
    EXPECT_EQ(g.time_steps, 16);
    EXPECT_FLOAT_EQ(g.voltage_threshold, 0.2f);
    EXPECT_EQ(g.latent(), 2);
    EXPECT_EQ(g.depth(), 4);

    auto ae = pga::to_ae_config(g, base_ae("snn-ae", "mse"));
    EXPECT_EQ(ae.model, "snn-ae");
    // encoder: every width leaky except the latent (identity)
    ASSERT_EQ(ae.encoder_layer_spec.size(), 4u);
    EXPECT_EQ(ae.encoder_layer_spec[0], "linear:10:leaky");
    EXPECT_EQ(ae.encoder_layer_spec[1], "linear:5:leaky");
    EXPECT_EQ(ae.encoder_layer_spec[2], "linear:4:leaky");
    EXPECT_EQ(ae.encoder_layer_spec[3], "linear:2:identity");
    // decoder mirrors the hidden widths back up, then projects to output
    ASSERT_EQ(ae.decoder_layer_spec.size(), 4u);
    EXPECT_EQ(ae.decoder_layer_spec[0], "linear:4:leaky");
    EXPECT_EQ(ae.decoder_layer_spec[1], "linear:5:leaky");
    EXPECT_EQ(ae.decoder_layer_spec[2], "linear:10:leaky");
    EXPECT_EQ(ae.decoder_layer_spec[3], "linear:output:identity");
    EXPECT_EQ(ae.encoding, "latency");
    EXPECT_FLOAT_EQ(ae.firing_rate_reg_lambda, 0.5f);
}

TEST(PgaGenome, AnnMappingForcesDirect)
{
    Genome g;
    g.encoder_widths = {32, 16};
    g.encoding = "poisson"; // must be ignored for ann-ae
    auto ae = pga::to_ae_config(g, base_ae("ann-ae", "mse"));
    EXPECT_EQ(ae.encoding, "direct");
    EXPECT_EQ(ae.time_steps, 1);
    EXPECT_FLOAT_EQ(ae.voltage_threshold, 1.0f);
}

TEST(PgaGenome, ProxiesMonotonic)
{
    Genome small;
    small.encoder_widths = {16, 8};
    Genome big;
    big.encoder_widths = {128, 64};
    EXPECT_LT(pga::estimated_params(small), pga::estimated_params(big));
    EXPECT_LT(pga::inference_cost_proxy(small), pga::inference_cost_proxy(big));
}

TEST(PgaGenome, RandomGenomeKeepsBottleneck)
{
    std::mt19937 rng(123);
    pga::GenomeBounds bounds;
    for (int i = 0; i < 500; ++i)
    {
        Genome g = pga::random_genome(rng, bounds, /*is_snn=*/true);
        ASSERT_GE(g.depth(), bounds.min_layers);
        ASSERT_LE(g.depth(), bounds.max_layers);
        // strictly decreasing across EVERY layer (each compresses toward the bottleneck)
        for (size_t k = 0; k + 1 < g.encoder_widths.size(); ++k)
            EXPECT_GT(g.encoder_widths[k], g.encoder_widths[k + 1])
                << "widths must strictly decrease; layer " << k;
        for (int w : g.encoder_widths)
        {
            EXPECT_GE(w, bounds.min_width);
            EXPECT_LE(w, bounds.max_width);
        }
    }
}

// REGRESSION: the genome→config mapping must never discard a profile field.
// It previously rebuilt the config from scratch, silently resetting `ae_loss_type`
// to its "mse" default — so an mse run and an mae run produced bit-identical
// results, and spikecount/spiketime profiles secretly trained under MSE.
TEST(PgaGenome, ToAeConfigPreservesProfileFields)
{
    Genome g;
    g.encoder_widths = {32, 16};
    g.encoding = "latency";
    pga::apply_phase00_temporal_coupling(g);

    for (const char* loss : {"mse", "mae", "spiketime"})
    {
        auto base = base_ae("snn-ae", loss);
        auto ae = pga::to_ae_config(g, base);
        EXPECT_EQ(ae.ae_loss_type, loss) << "profile ae_loss_type was dropped by to_ae_config";
        EXPECT_EQ(ae.model, "snn-ae");
        // firing-rate band belongs to the profile, not the genome
        EXPECT_FLOAT_EQ(ae.firing_rate_reg_lambda, 0.5f);
        EXPECT_FLOAT_EQ(ae.firing_rate_min, 0.1f);
        EXPECT_FLOAT_EQ(ae.firing_rate_max, 0.8f);
        // genome-owned fields still applied
        EXPECT_EQ(ae.encoding, "latency");
        EXPECT_EQ(ae.encoder_layer_spec.front(), "linear:32:leaky");
        EXPECT_EQ(ae.encoder_layer_spec.back(), "linear:16:identity");
    }
}

TEST(PgaGenome, ToAeConfigPreservesLossForAnnToo)
{
    Genome g;
    g.encoder_widths = {8, 4};
    auto ae = pga::to_ae_config(g, base_ae("ann-ae", "mae"));
    EXPECT_EQ(ae.ae_loss_type, "mae");
    EXPECT_EQ(ae.encoding, "direct"); // model-implied override still wins
}

// ── Free architecture: depth + width both vary, repaired to a legal shape ────
TEST(PgaGenome, RepairForcesStrictlyDecreasing)
{
    pga::GenomeBounds bounds; // min_width=1, max_width=128
    Genome g;
    g.encoder_widths = {3, 3, 5, 1, 5}; // unsorted, duplicates, rising
    pga::repair_widths(g, bounds);
    ASSERT_GE(g.depth(), 1);
    for (size_t k = 0; k + 1 < g.encoder_widths.size(); ++k)
        EXPECT_GT(g.encoder_widths[k], g.encoder_widths[k + 1]);
    for (int w : g.encoder_widths)
    {
        EXPECT_GE(w, bounds.min_width);
        EXPECT_LE(w, bounds.max_width);
    }
}

TEST(PgaGenome, MutationCanChangeDepthAndStaysLegal)
{
    std::mt19937 rng(7);
    pga::GenomeBounds bounds;
    bounds.min_layers = 1;
    bounds.max_layers = 6;
    Genome g = pga::random_genome(rng, bounds, /*is_snn=*/true);
    bool saw_grow = false, saw_shrink = false;
    int prev = g.depth();
    for (int i = 0; i < 300; ++i)
    {
        pga::mutate(g, rng, bounds, /*prob=*/0.6, /*is_snn=*/true);
        ASSERT_GE(g.depth(), bounds.min_layers);
        ASSERT_LE(g.depth(), bounds.max_layers);
        for (size_t k = 0; k + 1 < g.encoder_widths.size(); ++k)
            ASSERT_GT(g.encoder_widths[k], g.encoder_widths[k + 1]);
        if (g.depth() > prev) saw_grow = true;
        if (g.depth() < prev) saw_shrink = true;
        prev = g.depth();
    }
    EXPECT_TRUE(saw_grow) << "add-layer mutation never fired";
    EXPECT_TRUE(saw_shrink) << "remove-layer mutation never fired";
}

TEST(PgaGenome, CrossoverOfDifferentDepthsIsLegal)
{
    std::mt19937 rng(11);
    pga::GenomeBounds bounds;
    Genome a;
    a.encoder_widths = {3, 2, 1};
    Genome b;
    b.encoder_widths = {10, 5, 4, 2};
    for (int i = 0; i < 200; ++i)
    {
        Genome c = pga::crossover(a, b, rng, bounds, /*is_snn=*/false);
        ASSERT_GE(c.depth(), 1);
        for (size_t k = 0; k + 1 < c.encoder_widths.size(); ++k)
            ASSERT_GT(c.encoder_widths[k], c.encoder_widths[k + 1]);
    }
}

// ── True diploid genetics ────────────────────────────────────────────────────
TEST(PgaDiploid, RandomDiploidBothHaplotypesLegal)
{
    std::mt19937 rng(321);
    pga::GenomeBounds bounds;
    for (int i = 0; i < 300; ++i)
    {
        pga::DiploidGenome d = pga::random_diploid(rng, bounds, /*is_snn=*/true);
        for (const pga::Genome* h : {&d.hap_a, &d.hap_b})
        {
            ASSERT_GE(h->depth(), bounds.min_layers);
            ASSERT_LE(h->depth(), bounds.max_layers);
            for (size_t k = 0; k + 1 < h->encoder_widths.size(); ++k)
                EXPECT_GT(h->encoder_widths[k], h->encoder_widths[k + 1]);
        }
        EXPECT_GE(d.dom_a, 0.0f);
        EXPECT_LE(d.dom_a, 1.0f);
        EXPECT_GE(d.dom_b, 0.0f);
        EXPECT_LE(d.dom_b, 1.0f);
    }
}

TEST(PgaDiploid, ExpressedIsHigherDominanceHaplotype)
{
    pga::DiploidGenome d;
    d.hap_a.encoder_widths = {64, 32};
    d.hap_b.encoder_widths = {10, 5, 2};
    d.dom_a = 0.7f;
    d.dom_b = 0.3f;
    EXPECT_EQ(d.expressed().encoder_widths, d.hap_a.encoder_widths);
    d.dom_a = 0.2f;
    d.dom_b = 0.9f;
    EXPECT_EQ(d.expressed().encoder_widths, d.hap_b.encoder_widths);
    // Tie resolves to A, deterministically.
    d.dom_a = d.dom_b = 0.5f;
    EXPECT_EQ(d.expressed().encoder_widths, d.hap_a.encoder_widths);
}

TEST(PgaDiploid, MeiosisProducesLegalGameteAndBoundedDominance)
{
    std::mt19937 rng(99);
    pga::GenomeBounds bounds;
    pga::DiploidGenome d = pga::random_diploid(rng, bounds, /*is_snn=*/true);
    for (int i = 0; i < 300; ++i)
    {
        pga::Gamete g = pga::meiosis(
            d, rng, bounds, /*recomb_prob=*/0.9, /*mutation_prob=*/0.3, /*is_snn=*/true);
        ASSERT_GE(g.haplotype.depth(), bounds.min_layers);
        ASSERT_LE(g.haplotype.depth(), bounds.max_layers);
        for (size_t k = 0; k + 1 < g.haplotype.encoder_widths.size(); ++k)
            ASSERT_GT(g.haplotype.encoder_widths[k], g.haplotype.encoder_widths[k + 1]);
        EXPECT_GE(g.dominance, 0.0f);
        EXPECT_LE(g.dominance, 1.0f);
    }
}

TEST(PgaDiploid, FuseMapsGametesToHaplotypes)
{
    pga::Gamete g1;
    g1.haplotype.encoder_widths = {8, 4};
    g1.dominance = 0.6f;
    pga::Gamete g2;
    g2.haplotype.encoder_widths = {30, 20, 10};
    g2.dominance = 0.1f;
    pga::DiploidGenome child = pga::fuse(g1, g2);
    EXPECT_EQ(child.hap_a.encoder_widths, g1.haplotype.encoder_widths);
    EXPECT_FLOAT_EQ(child.dom_a, 0.6f);
    EXPECT_EQ(child.hap_b.encoder_widths, g2.haplotype.encoder_widths);
    EXPECT_FLOAT_EQ(child.dom_b, 0.1f);
    // The dominant gamete (g1) is the one expressed.
    EXPECT_EQ(child.expressed().encoder_widths, g1.haplotype.encoder_widths);
}

// A recessive haplotype (never expressed) must still be able to reach a child's
// expressed slot via meiosis+fusion — the whole point of the diploid reservoir.
TEST(PgaDiploid, RecessiveAlleleCanResurface)
{
    std::mt19937 rng(2024);
    pga::GenomeBounds bounds;
    pga::DiploidGenome parent;
    parent.hap_a.encoder_widths = {120};     // dominant, expressed
    parent.hap_b.encoder_widths = {3, 2, 1}; // recessive, hidden
    parent.dom_a = 0.99f;
    parent.dom_b = 0.01f;

    bool recessive_expressed = false;
    for (int i = 0; i < 2000 && !recessive_expressed; ++i)
    {
        // No mutation/recombination so haplotypes pass through intact; only the dominance
        // coin flips decide which gamete allele + dominance the child expresses.
        pga::Gamete g1 = pga::meiosis(parent, rng, bounds, 0.0, 0.0, /*is_snn=*/false);
        pga::Gamete g2 = pga::meiosis(parent, rng, bounds, 0.0, 0.0, /*is_snn=*/false);
        pga::DiploidGenome child = pga::fuse(g1, g2);
        if (child.expressed().encoder_widths == std::vector<int>{3, 2, 1})
            recessive_expressed = true;
    }
    EXPECT_TRUE(recessive_expressed)
        << "the hidden (recessive) haplotype never reached expression across 2000 crosses";
}

// ── Checkpoint / resume serialization ────────────────────────────────────────
namespace
{
Individual make_evaluated_individual()
{
    Individual ind;
    ind.genotype.hap_a.encoder_widths = {64, 32, 8};
    ind.genotype.hap_a.encoding = "latency";
    ind.genotype.hap_a.time_steps = 16;
    ind.genotype.hap_a.voltage_threshold = 0.2f;
    ind.genotype.hap_b.encoder_widths = {50, 3};
    ind.genotype.hap_b.encoding = "direct";
    ind.genotype.dom_a = 0.8f;
    ind.genotype.dom_b = 0.2f;
    ind.genome = ind.genotype.expressed(); // hap_a
    ind.d_penalized_mean = 0.37;
    ind.d_penalized_std = 0.02;
    ind.alpha = 0.44;
    ind.beta = 0.61;
    ind.g1 = -0.17;
    ind.g2 = 0.05;
    ind.d_truth = 0.31;
    ind.latent_activity = 0.5;
    ind.param_count = 12345;
    ind.inference_cost = 6789;
    ind.est_latency_ms = 13.5;
    ind.evaluated = true;
    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {0.37, 6789.0};
    ind.born_generation = 3;
    return ind;
}

std::string unique_tmp_dir(const char* tag)
{
    auto dir = std::filesystem::temp_directory_path() /
               ("pga_ckpt_" + std::string(tag) + "_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    return dir.string();
}
} // namespace

TEST(PgaCheckpoint, IndividualJsonRoundTripPreservesEveryField)
{
    const Individual a = make_evaluated_individual();
    const Individual b =
        pga::individual_from_checkpoint_json(pga::individual_to_checkpoint_json(a));

    // Diploid genotype (needed for reproduction on resume).
    EXPECT_EQ(b.genotype.hap_a.encoder_widths, a.genotype.hap_a.encoder_widths);
    EXPECT_EQ(b.genotype.hap_a.encoding, a.genotype.hap_a.encoding);
    EXPECT_EQ(b.genotype.hap_a.time_steps, a.genotype.hap_a.time_steps);
    EXPECT_FLOAT_EQ(b.genotype.hap_a.voltage_threshold, a.genotype.hap_a.voltage_threshold);
    EXPECT_EQ(b.genotype.hap_b.encoder_widths, a.genotype.hap_b.encoder_widths);
    EXPECT_FLOAT_EQ(b.genotype.dom_a, a.genotype.dom_a);
    EXPECT_FLOAT_EQ(b.genotype.dom_b, a.genotype.dom_b);
    EXPECT_EQ(b.genotype.expressed().encoder_widths, a.genotype.expressed().encoder_widths);
    // Phenotype + metrics.
    EXPECT_EQ(b.genome.encoder_widths, a.genome.encoder_widths);
    EXPECT_DOUBLE_EQ(b.d_penalized_mean, a.d_penalized_mean);
    EXPECT_DOUBLE_EQ(b.beta, a.beta);
    EXPECT_EQ(b.param_count, a.param_count);
    EXPECT_EQ(b.inference_cost, a.inference_cost);
    EXPECT_EQ(b.feasible, a.feasible);
    EXPECT_EQ(b.objectives, a.objectives);
    EXPECT_EQ(b.born_generation, a.born_generation);
}

TEST(PgaCheckpoint, RngStateRoundTripReproducesDraws)
{
    std::mt19937 rng(12345);
    for (int i = 0; i < 100; ++i) rng(); // advance to a nontrivial state
    const std::string state = pga::rng_to_string(rng);

    std::mt19937 restored(999);
    pga::rng_from_string(restored, state);

    // The restored engine must produce the identical subsequent sequence.
    for (int i = 0; i < 50; ++i) EXPECT_EQ(rng(), restored());
}

TEST(PgaCheckpoint, CacheAppendAndReloadRoundTrips)
{
    const std::string dir = unique_tmp_dir("cache");
    const std::string path = dir + "/cache.jsonl";

    Individual a = make_evaluated_individual();
    Individual b = make_evaluated_individual();
    b.genome.encoder_widths = {16, 4};
    b.d_penalized_mean = 0.9;

    pga::append_cache_entry(path, a);
    pga::append_cache_entry(path, b);

    auto loaded = pga::load_cache_entries(path);
    ASSERT_EQ(loaded.size(), 2u);
    EXPECT_EQ(loaded[0].genome.encoder_widths, a.genome.encoder_widths);
    EXPECT_EQ(loaded[1].genome.encoder_widths, b.genome.encoder_widths);
    EXPECT_DOUBLE_EQ(loaded[1].d_penalized_mean, 0.9);

    std::filesystem::remove_all(dir);
}

TEST(PgaCheckpoint, TornTrailingCacheLineIsDroppedNotFatal)
{
    const std::string dir = unique_tmp_dir("torn");
    const std::string path = dir + "/cache.jsonl";

    pga::append_cache_entry(path, make_evaluated_individual());
    // Simulate a crash mid-append: a partial, unparseable trailing line.
    {
        std::ofstream f(path, std::ios::app);
        f << "{\"genotype\": {\"hap_a\": {\"enc"; // truncated JSON, no newline terminator
    }
    auto loaded = pga::load_cache_entries(path); // must not throw
    EXPECT_EQ(loaded.size(), 1u);                // the one complete record survives

    std::filesystem::remove_all(dir);
}

TEST(PgaCheckpoint, GenerationCheckpointRoundTrips)
{
    const std::string dir = unique_tmp_dir("gen");
    const std::string tag = "unit";

    std::mt19937 rng(42);
    for (int i = 0; i < 20; ++i) rng();
    std::vector<Individual> parents = {make_evaluated_individual(), make_evaluated_individual()};
    parents[1].genome.encoder_widths = {7, 2};

    EXPECT_FALSE(pga::state_checkpoint_exists(dir, tag));
    pga::save_generation_checkpoint(dir, tag, 5, rng, parents);
    EXPECT_TRUE(pga::state_checkpoint_exists(dir, tag));

    auto ck = pga::load_generation_checkpoint(dir, tag);
    EXPECT_EQ(ck.generation, 5);
    ASSERT_EQ(ck.parents.size(), 2u);
    EXPECT_EQ(ck.parents[1].genome.encoder_widths, (std::vector<int>{7, 2}));

    // RNG restored from the checkpoint reproduces the engine's next draws.
    std::mt19937 restored(1);
    pga::rng_from_string(restored, ck.rng_state);
    for (int i = 0; i < 30; ++i) EXPECT_EQ(rng(), restored());

    pga::remove_checkpoint_artifacts(dir, tag);
    EXPECT_FALSE(pga::state_checkpoint_exists(dir, tag));
    std::filesystem::remove_all(dir);
}

// ── Latency pre-screen (.wiki/Experiments/ParaconsistentGA-Design.md §4)
// ────────────────────────────────────────────
TEST(PgaFitness, LatencyPreScreenRejectsOverBudget)
{
    pga::GaConfig cfg;
    cfg.constraints.latency_ceiling_ms = 0.001; // absurdly tight → everything over budget
    cfg.constraints.fixed_pipeline_cost_ms = 0.0;
    cfg.constraints.ns_per_mac = 2.0;
    Genome g;
    g.encoder_widths = {128, 64};
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
