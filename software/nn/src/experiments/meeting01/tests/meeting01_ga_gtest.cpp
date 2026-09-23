// meeting01_ga_gtest.cpp — coverage for the NSGA-II SNN-AE architecture search
// (Meeting01GaGenome/Fitness/Checkpoint/Search): genome repair/crossover/mutate stay
// legal, to_ae_config reproduces today's fixed profile as one reachable point in the
// free-form space, and the checkpoint round-trips.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#include "Meeting01Config.hpp"
#include "Meeting01GaCheckpoint.hpp"
#include "Meeting01GaFitness.hpp"
#include "Meeting01GaGenome.hpp"
#include "Meeting01GaSearch.hpp"
#include "Meeting01Metrics.hpp"
#include "ga/Nsga2Core.hpp"

namespace
{

using meeting01::ga::Genome;
using meeting01::ga::GenomeBounds;

GenomeBounds make_bounds()
{
    GenomeBounds b;
    b.min_layers = 1;
    b.max_layers = 4;
    b.min_width = 4;
    b.max_width = 128;
    b.voltage_threshold_min = 0.1f;
    b.voltage_threshold_max = 2.0f;
    b.alpha_min = 0.5f;
    b.alpha_max = 0.99f;
    b.encoding_choices = {"direct", "poisson", "latency"};
    b.architecture_choices = {"dense", "conv1d", "recurrent"};
    return b;
}

bool strictly_decreasing(const std::vector<int>& w)
{
    for (std::size_t i = 1; i < w.size(); ++i)
        if (w[i - 1] <= w[i]) return false;
    return true;
}

TEST(Meeting01GaGenome, RandomGenomeIsLegal)
{
    std::mt19937 rng(1);
    const auto bounds = make_bounds();
    for (int trial = 0; trial < 200; ++trial)
    {
        const Genome g = meeting01::ga::random_genome(rng, bounds);
        EXPECT_GE(g.depth(), bounds.min_layers);
        EXPECT_LE(g.depth(), bounds.max_layers);
        EXPECT_TRUE(strictly_decreasing(g.encoder_widths));
        for (int w : g.encoder_widths)
        {
            EXPECT_GE(w, bounds.min_width);
            EXPECT_LE(w, bounds.max_width);
        }
        EXPECT_NE(
            std::find(bounds.encoding_choices.begin(), bounds.encoding_choices.end(), g.encoding),
            bounds.encoding_choices.end());
        EXPECT_NE(std::find(bounds.architecture_choices.begin(),
                      bounds.architecture_choices.end(),
                      g.architecture),
            bounds.architecture_choices.end());
        EXPECT_GE(g.voltage_threshold, bounds.voltage_threshold_min);
        EXPECT_LE(g.voltage_threshold, bounds.voltage_threshold_max);
        EXPECT_GE(g.alpha, bounds.alpha_min);
        EXPECT_LE(g.alpha, bounds.alpha_max);
    }
}

TEST(Meeting01GaGenome, RepairForcesStrictlyDecreasing)
{
    const auto bounds = make_bounds();
    Genome g;
    g.encoder_widths = {5, 5, 5, 200, -3};
    meeting01::ga::repair_widths(g, bounds);
    EXPECT_TRUE(strictly_decreasing(g.encoder_widths));
    for (int w : g.encoder_widths)
    {
        EXPECT_GE(w, bounds.min_width);
        EXPECT_LE(w, bounds.max_width);
    }
    EXPECT_LE(static_cast<int>(g.encoder_widths.size()), bounds.max_layers);
    EXPECT_FALSE(g.encoder_widths.empty());
}

TEST(Meeting01GaGenome, CrossoverOfDifferentDepthsIsLegal)
{
    std::mt19937 rng(7);
    const auto bounds = make_bounds();
    const Genome a{{100, 60, 30}, "direct", "dense", 1.0f, 0.9f};
    const Genome b{{40}, "latency", "recurrent", 0.5f, 0.7f};
    for (int trial = 0; trial < 50; ++trial)
    {
        const Genome c = meeting01::ga::crossover(a, b, rng, bounds);
        EXPECT_TRUE(strictly_decreasing(c.encoder_widths));
        EXPECT_GE(c.depth(), bounds.min_layers);
        EXPECT_LE(c.depth(), bounds.max_layers);
        // Arithmetic-mean crossover for continuous genes: child is between the parents.
        EXPECT_GE(c.voltage_threshold, std::min(a.voltage_threshold, b.voltage_threshold) - 1e-4f);
        EXPECT_LE(c.voltage_threshold, std::max(a.voltage_threshold, b.voltage_threshold) + 1e-4f);
    }
}

TEST(Meeting01GaGenome, MutationCanChangeDepthAndStaysLegal)
{
    std::mt19937 rng(13);
    const auto bounds = make_bounds();
    bool depth_changed = false;
    bool encoding_changed = false;
    bool architecture_changed = false;
    for (int trial = 0; trial < 500; ++trial)
    {
        Genome g{{80, 40}, "direct", "dense", 1.0f, 0.9f};
        const int before_depth = g.depth();
        meeting01::ga::mutate(g, rng, bounds, /*prob=*/0.5);

        EXPECT_TRUE(strictly_decreasing(g.encoder_widths));
        EXPECT_GE(g.depth(), bounds.min_layers);
        EXPECT_LE(g.depth(), bounds.max_layers);
        EXPECT_NE(
            std::find(bounds.encoding_choices.begin(), bounds.encoding_choices.end(), g.encoding),
            bounds.encoding_choices.end());
        EXPECT_NE(std::find(bounds.architecture_choices.begin(),
                      bounds.architecture_choices.end(),
                      g.architecture),
            bounds.architecture_choices.end());
        EXPECT_GE(g.voltage_threshold, bounds.voltage_threshold_min);
        EXPECT_LE(g.voltage_threshold, bounds.voltage_threshold_max);
        EXPECT_GE(g.alpha, bounds.alpha_min);
        EXPECT_LE(g.alpha, bounds.alpha_max);

        if (g.depth() != before_depth) depth_changed = true;
        if (g.encoding != "direct") encoding_changed = true;
        if (g.architecture != "dense") architecture_changed = true;
    }
    EXPECT_TRUE(depth_changed) << "500 trials at prob=0.5 never changed depth";
    EXPECT_TRUE(encoding_changed) << "500 trials at prob=0.5 never changed encoding";
    EXPECT_TRUE(architecture_changed) << "500 trials at prob=0.5 never changed architecture";
}

TEST(Meeting01GaGenome, GenomeKeyIsStableAndDiscriminating)
{
    const Genome a{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    const Genome b = a;
    Genome c = a;
    c.alpha = 0.8f;
    EXPECT_EQ(meeting01::ga::genome_key(a), meeting01::ga::genome_key(b));
    EXPECT_NE(meeting01::ga::genome_key(a), meeting01::ga::genome_key(c));
}

TEST(Meeting01GaGenome, ToAeConfigReproducesTheFixedProfileShape)
{
    // The free-form path must reproduce today's fixed encoder_layer_spec as one
    // reachable point in the space — a regression anchor against the grid-search
    // profile's own convention (linear:64:leaky, linear:32:identity).
    meeting01::Meeting01Config cfg;
    cfg.model.loss_type = "mse";
    cfg.model.encoder_layer_spec = {"linear:64:leaky", "linear:32:identity"};
    cfg.model.decoder_layer_spec = {"linear:64:leaky", "linear:output:identity"};
    cfg.dataset.window_size = 256;

    const Genome g{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    const auto ae = meeting01::ga::to_ae_config(g, cfg);

    ASSERT_EQ(ae.encoder_layer_spec.size(), 2u);
    EXPECT_EQ(ae.encoder_layer_spec[0], "linear:64:leaky");
    EXPECT_EQ(ae.encoder_layer_spec[1], "linear:32:identity");
    ASSERT_EQ(ae.decoder_layer_spec.size(), 2u);
    EXPECT_EQ(ae.decoder_layer_spec[0], "linear:64:leaky");
    EXPECT_EQ(ae.decoder_layer_spec[1], "linear:output:identity");
    EXPECT_EQ(ae.input_features, 256);
    EXPECT_EQ(ae.loss_type, "mse");
}

TEST(Meeting01GaGenome, ToAeConfigRendersFreeFormWidths)
{
    meeting01::Meeting01Config cfg;
    cfg.model.loss_type = "mse";
    cfg.model.encoder_layer_spec = {"linear:64:leaky", "linear:32:identity"};
    cfg.model.decoder_layer_spec = {"linear:64:leaky", "linear:output:identity"};
    cfg.dataset.window_size = 128;

    const Genome g{{96, 40, 12}, "poisson", "recurrent", 0.4f, 0.85f};
    const auto ae = meeting01::ga::to_ae_config(g, cfg);

    ASSERT_EQ(ae.encoder_layer_spec.size(), 3u);
    EXPECT_EQ(ae.encoder_layer_spec[0], "linear:96:leaky");
    EXPECT_EQ(ae.encoder_layer_spec[1], "linear:40:leaky");
    EXPECT_EQ(ae.encoder_layer_spec[2], "linear:12:identity");
    ASSERT_EQ(ae.decoder_layer_spec.size(), 3u);
    EXPECT_EQ(ae.decoder_layer_spec[0], "linear:40:leaky");
    EXPECT_EQ(ae.decoder_layer_spec[1], "linear:96:leaky");
    EXPECT_EQ(ae.decoder_layer_spec[2], "linear:output:identity");
}

TEST(Meeting01GaCheckpoint, IndividualJsonRoundTripPreservesEveryField)
{
    meeting01::ga::Meeting01GaIndividual ind;
    ind.genome = Genome{{72, 24}, "latency", "conv1d", 0.6f, 0.77f};
    ind.val_mse = 0.1234f;
    ind.param_count = 4242;
    ind.inference_cost = 99999;
    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {0.1234, 99999.0};
    ind.born_generation = 3;

    const auto j = meeting01::ga::individual_to_checkpoint_json(ind);
    const auto back =
        meeting01::ga::individual_from_checkpoint_json<meeting01::ga::Meeting01GaIndividual>(j);

    EXPECT_EQ(back.genome.encoder_widths, ind.genome.encoder_widths);
    EXPECT_EQ(back.genome.encoding, ind.genome.encoding);
    EXPECT_EQ(back.genome.architecture, ind.genome.architecture);
    EXPECT_FLOAT_EQ(back.genome.voltage_threshold, ind.genome.voltage_threshold);
    EXPECT_FLOAT_EQ(back.genome.alpha, ind.genome.alpha);
    EXPECT_FLOAT_EQ(back.val_mse, ind.val_mse);
    EXPECT_EQ(back.param_count, ind.param_count);
    EXPECT_EQ(back.inference_cost, ind.inference_cost);
    EXPECT_EQ(back.feasible, ind.feasible);
    EXPECT_EQ(back.objectives, ind.objectives);
    EXPECT_EQ(back.born_generation, ind.born_generation);
}

TEST(Meeting01GaCheckpoint, RngStateRoundTripReproducesDraws)
{
    std::mt19937 rng(2024);
    (void) rng(); // advance past the seed
    const std::string state = meeting01::ga::rng_to_string(rng);
    const std::uint32_t next_from_original = rng();

    std::mt19937 restored;
    meeting01::ga::rng_from_string(restored, state);
    EXPECT_EQ(restored(), next_from_original);
}

TEST(Meeting01GaCheckpoint, CacheAppendAndReloadRoundTrips)
{
    const std::string path =
        std::filesystem::temp_directory_path() / "meeting01_ga_gtest_cache.jsonl";
    std::filesystem::remove(path);

    meeting01::ga::Meeting01GaIndividual a;
    a.genome = Genome{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    a.val_mse = 0.5f;
    a.objectives = {0.5, 100.0};
    meeting01::ga::Meeting01GaIndividual b;
    b.genome = Genome{{48, 16}, "poisson", "recurrent", 0.3f, 0.6f};
    b.val_mse = 0.3f;
    b.objectives = {0.3, 80.0};

    meeting01::ga::append_cache_entry(path, a);
    meeting01::ga::append_cache_entry(path, b);

    const auto loaded =
        meeting01::ga::load_cache_entries<meeting01::ga::Meeting01GaIndividual>(path);
    ASSERT_EQ(loaded.size(), 2u);
    EXPECT_EQ(loaded[0].genome.encoder_widths, a.genome.encoder_widths);
    EXPECT_EQ(loaded[1].genome.encoder_widths, b.genome.encoder_widths);

    std::filesystem::remove(path);
}

TEST(Meeting01GaCheckpoint, TornTrailingCacheLineIsDroppedNotFatal)
{
    const std::string path =
        std::filesystem::temp_directory_path() / "meeting01_ga_gtest_torn_cache.jsonl";
    std::filesystem::remove(path);

    meeting01::ga::Meeting01GaIndividual a;
    a.genome = Genome{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    a.objectives = {0.5, 100.0};
    meeting01::ga::append_cache_entry(path, a);

    {
        std::ofstream f(path, std::ios::app);
        f << "{\"genome\":{\"encoder_widths\":[1"; // torn: no closing braces
    }

    const auto loaded =
        meeting01::ga::load_cache_entries<meeting01::ga::Meeting01GaIndividual>(path);
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded[0].genome.encoder_widths, a.genome.encoder_widths);

    std::filesystem::remove(path);
}

TEST(Meeting01GaCheckpoint, GenerationCheckpointRoundTrips)
{
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string tag = "gtest_meeting01_ga";
    meeting01::ga::remove_checkpoint_artifacts(dir, tag);

    std::mt19937 rng(99);
    std::vector<meeting01::ga::Meeting01GaIndividual> parents(2);
    parents[0].genome = Genome{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    parents[0].objectives = {0.5, 100.0};
    parents[1].genome = Genome{{48, 16}, "poisson", "recurrent", 0.3f, 0.6f};
    parents[1].objectives = {0.3, 80.0};

    meeting01::ga::save_generation_checkpoint(dir, tag, 2, rng, parents);
    EXPECT_TRUE(meeting01::ga::state_checkpoint_exists(dir, tag));

    const auto ck =
        meeting01::ga::load_generation_checkpoint<meeting01::ga::Meeting01GaIndividual>(dir, tag);
    EXPECT_EQ(ck.generation, 2);
    ASSERT_EQ(ck.parents.size(), 2u);
    EXPECT_EQ(ck.parents[0].genome.encoder_widths, parents[0].genome.encoder_widths);
    EXPECT_EQ(ck.parents[1].genome.encoder_widths, parents[1].genome.encoder_widths);

    meeting01::ga::remove_checkpoint_artifacts(dir, tag);
    EXPECT_FALSE(meeting01::ga::state_checkpoint_exists(dir, tag));
}

TEST(Meeting01GaNsga2Core, SharedCoreRanksFeasibleAboveInfeasible)
{
    // Exercises the shared ga::Nsga2Core template through Meeting01GaIndividual, the
    // same generic contract paraconsistentGA's pga::Individual satisfies.
    std::vector<meeting01::ga::Meeting01GaIndividual> pop(2);
    pop[0].feasible = true;
    pop[0].objectives = {1.0, 1.0};
    pop[1].feasible = false;
    pop[1].constraint_violation = 5.0;
    pop[1].objectives = {0.5, 0.5}; // would dominate on objectives alone if feasibility ignored

    EXPECT_TRUE(::ga::constrained_dominates(pop[0], pop[1]));
    EXPECT_FALSE(::ga::constrained_dominates(pop[1], pop[0]));

    const auto fronts = ::ga::fast_non_dominated_sort(pop);
    ASSERT_FALSE(fronts.empty());
    EXPECT_EQ(fronts[0].size(), 1u);
    EXPECT_EQ(fronts[0][0], 0);
}

// ---------------------------------------------------------------------------
// NSGA-II search core (added 2026-09-22). run_ga_search / select_next_generation /
// tournament / pick_winner previously had NO direct coverage at all — the code that
// chooses the architecture the paper publishes was the least-tested code in the
// experiment. These exercise the selection machinery through the public surface plus
// the shared core, without training anything.
// ---------------------------------------------------------------------------

namespace
{
meeting01::ga::Meeting01GaIndividual make_ind(double mse, double cost, bool feasible = true)
{
    meeting01::ga::Meeting01GaIndividual ind;
    ind.genome = Genome{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    ind.val_mse = static_cast<float>(mse);
    ind.inference_cost = static_cast<std::size_t>(cost);
    ind.feasible = feasible;
    ind.objectives = {mse, cost};
    return ind;
}
} // namespace

TEST(Meeting01GaSearch, PickWinnerTakesLowestMseThenLowestCost)
{
    meeting01::ga::GaSearchResult r;
    r.pareto_front = {make_ind(0.20, 10.0), make_ind(0.10, 90.0), make_ind(0.10, 30.0)};
    // pick_winner relies on run_ga_search having sorted the front; sort the same way.
    std::sort(r.pareto_front.begin(),
        r.pareto_front.end(),
        [](const auto& a, const auto& b)
        {
            if (a.val_mse != b.val_mse) return a.val_mse < b.val_mse;
            return a.inference_cost < b.inference_cost;
        });

    const auto& w = meeting01::ga::pick_winner(r);
    EXPECT_FLOAT_EQ(w.val_mse, 0.10f);
    EXPECT_EQ(w.inference_cost, 30u); // cheaper of the two tied on MSE
}

TEST(Meeting01GaSearch, PickWinnerThrowsOnEmptyFrontInsteadOfReturningGarbage)
{
    meeting01::ga::GaSearchResult empty;
    EXPECT_THROW((void) meeting01::ga::pick_winner(empty), std::runtime_error);
}

TEST(Meeting01GaSearch, NonDominatedSortSeparatesDominatedIndividuals)
{
    // (0.1, 10) dominates (0.2, 20); (0.1, 50) and (0.3, 5) are mutually non-dominated.
    std::vector<meeting01::ga::Meeting01GaIndividual> pop{
        make_ind(0.1, 10.0), make_ind(0.2, 20.0), make_ind(0.3, 5.0)};

    const auto fronts = ::ga::fast_non_dominated_sort(pop);
    ASSERT_GE(fronts.size(), 2u);
    // Front 0 holds the two mutually non-dominated ones (indices 0 and 2).
    EXPECT_EQ(fronts[0].size(), 2u);
    EXPECT_NE(std::find(fronts[0].begin(), fronts[0].end(), 0), fronts[0].end());
    EXPECT_NE(std::find(fronts[0].begin(), fronts[0].end(), 2), fronts[0].end());
    // The dominated one is not in front 0.
    EXPECT_EQ(std::find(fronts[0].begin(), fronts[0].end(), 1), fronts[0].end());
}

TEST(Meeting01GaSearch, CrowdingKeepsBoundarySolutionsInfinite)
{
    std::vector<meeting01::ga::Meeting01GaIndividual> pop{
        make_ind(0.1, 100.0), make_ind(0.2, 50.0), make_ind(0.3, 10.0)};
    const auto fronts = ::ga::fast_non_dominated_sort(pop);
    ::ga::assign_crowding_distance(pop, fronts[0]);

    // Extremes must be preserved by elitism, so their crowding is infinite and the
    // interior point is finite — that is what stops the front collapsing to one corner.
    int infinite = 0;
    for (int idx : fronts[0])
        if (std::isinf(pop[static_cast<std::size_t>(idx)].crowding)) ++infinite;
    EXPECT_GE(infinite, 2);
}

TEST(Meeting01GaSearch, ConfigRejectsPopulationOneWithGenerations)
{
    // Regression: population_size == 1 with generations >= 1 used to pass validation and
    // then hang forever inside tournament()'s rejection loop — a cluster job that burns
    // days producing nothing. Validation must reject it up front.
    meeting01::Meeting01Config cfg;
    cfg.experiment.run_tag = "t";
    cfg.experiment.seed = 42;
    cfg.experiment.repeats = 1;
    cfg.dataset.dataset_root = "/tmp";
    cfg.dataset.window_size = 64;
    cfg.dataset.max_loaded_train_samples = 10;
    cfg.dataset.max_validation_samples = 5;
    cfg.training.samples_per_batch = 1;
    cfg.training.epochs = 1;
    cfg.training.early_stop_patience = 0;
    cfg.training.learning_rate = 0.001f;
    cfg.model.encoder_layer_spec = {"linear:16:leaky", "linear:8:identity"};
    cfg.model.decoder_layer_spec = {"linear:8:leaky", "linear:output:identity"};
    cfg.evaluation.datasets = {"fsdd"};
    cfg.evaluation.encodings = {"direct"};
    cfg.evaluation.snn_architectures = {"dense"};
    cfg.evaluation.ga.snn.population_size = 1;
    cfg.evaluation.ga.snn.generations = 1;

    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    // population 1 with 0 generations is legal: one random genome, no breeding.
    cfg.evaluation.ga.snn.generations = 0;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01GaSearch, ConfigRejectsSingleTimeStep)
{
    meeting01::Meeting01Config cfg;
    cfg.experiment.run_tag = "t";
    cfg.experiment.seed = 42;
    cfg.experiment.repeats = 1;
    cfg.dataset.dataset_root = "/tmp";
    cfg.dataset.window_size = 64;
    cfg.dataset.max_loaded_train_samples = 10;
    cfg.dataset.max_validation_samples = 5;
    cfg.training.samples_per_batch = 1;
    cfg.training.epochs = 1;
    cfg.training.early_stop_patience = 0;
    cfg.training.learning_rate = 0.001f;
    cfg.model.encoder_layer_spec = {"linear:16:leaky", "linear:8:identity"};
    cfg.model.decoder_layer_spec = {"linear:8:leaky", "linear:output:identity"};
    cfg.model.time_steps = 1;
    cfg.evaluation.datasets = {"fsdd"};
    cfg.evaluation.encodings = {"direct"};
    cfg.evaluation.snn_architectures = {"dense"};

    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(Meeting01GaMetrics, MacsDistinguishWidthsTheOldProxyCollapsed)
{
    // The (first_width, depth) proxy gave these two genomes an identical cost, so the
    // GA's second objective could not tell a cheap architecture from an expensive one.
    const auto narrow = meeting01::estimate_snn_macs(256, std::vector<int>{128, 8}, 16);
    const auto wide = meeting01::estimate_snn_macs(256, std::vector<int>{128, 120}, 16);
    EXPECT_GT(wide, narrow);

    // Cost scales with the simulation steps, which the window-flattened version ignored.
    EXPECT_GT(meeting01::estimate_snn_macs(256, std::vector<int>{64, 32}, 16),
        meeting01::estimate_snn_macs(256, std::vector<int>{64, 32}, 4));
}

} // namespace
