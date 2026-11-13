// meeting01_ga_gtest.cpp — coverage for the NSGA-II SNN-AE architecture search
// (Meeting01GaGenome/Fitness/Checkpoint/Search): genome repair/crossover/mutate stay
// legal, to_ae_config reproduces today's fixed profile as one reachable point in the
// free-form space, and the checkpoint round-trips.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01GaCheckpoint.hpp"
#include "Meeting01GaFitness.hpp"
#include "Meeting01GaGenome.hpp"
#include "Meeting01GaSearch.hpp"
#include "Meeting01Metrics.hpp"
#include "Meeting01Training.hpp"
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
    b.time_steps_min = 2;
    b.time_steps_max = 32;
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
        EXPECT_GE(g.time_steps, bounds.time_steps_min);
        EXPECT_LE(g.time_steps, bounds.time_steps_max);
    }
}

// T is a free integer gene: over many draws every value of a small range shows up, and
// nothing outside it does.
TEST(Meeting01GaGenome, RandomTimeStepsCoverTheWholeRange)
{
    std::mt19937 rng(3);
    auto bounds = make_bounds();
    bounds.time_steps_min = 3;
    bounds.time_steps_max = 6;
    std::set<int> seen;
    for (int trial = 0; trial < 400; ++trial)
        seen.insert(meeting01::ga::random_genome(rng, bounds).time_steps);
    EXPECT_EQ(seen, (std::set<int>{3, 4, 5, 6}));
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
    const Genome a{{100, 60, 30}, "direct", "dense", 1.0f, 0.9f, 4};
    const Genome b{{40}, "latency", "recurrent", 0.5f, 0.7f, 20};
    for (int trial = 0; trial < 50; ++trial)
    {
        const Genome c = meeting01::ga::crossover(a, b, rng, bounds);
        EXPECT_TRUE(strictly_decreasing(c.encoder_widths));
        EXPECT_GE(c.depth(), bounds.min_layers);
        EXPECT_LE(c.depth(), bounds.max_layers);
        // Arithmetic-mean crossover for continuous genes: child is between the parents.
        EXPECT_GE(c.voltage_threshold, std::min(a.voltage_threshold, b.voltage_threshold) - 1e-4f);
        EXPECT_LE(c.voltage_threshold, std::max(a.voltage_threshold, b.voltage_threshold) + 1e-4f);
        // T is inherited whole from one parent, never averaged into a third value.
        EXPECT_TRUE(c.time_steps == a.time_steps || c.time_steps == b.time_steps) << c.time_steps;
    }
}

TEST(Meeting01GaGenome, MutationCanChangeDepthAndStaysLegal)
{
    std::mt19937 rng(13);
    const auto bounds = make_bounds();
    bool depth_changed = false;
    bool encoding_changed = false;
    bool architecture_changed = false;
    bool time_steps_changed = false;
    for (int trial = 0; trial < 500; ++trial)
    {
        Genome g{{80, 40}, "direct", "dense", 1.0f, 0.9f, 16};
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
        EXPECT_GE(g.time_steps, bounds.time_steps_min);
        EXPECT_LE(g.time_steps, bounds.time_steps_max);

        if (g.depth() != before_depth) depth_changed = true;
        if (g.time_steps != 16) time_steps_changed = true;
        if (g.encoding != "direct") encoding_changed = true;
        if (g.architecture != "dense") architecture_changed = true;
    }
    EXPECT_TRUE(depth_changed) << "500 trials at prob=0.5 never changed depth";
    EXPECT_TRUE(encoding_changed) << "500 trials at prob=0.5 never changed encoding";
    EXPECT_TRUE(architecture_changed) << "500 trials at prob=0.5 never changed architecture";
    EXPECT_TRUE(time_steps_changed) << "500 trials at prob=0.5 never changed T";
}

TEST(Meeting01GaGenome, GenomeKeyIsStableAndDiscriminating)
{
    const Genome a{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    const Genome& b = a;
    Genome c = a;
    c.alpha = 0.8f;
    Genome d = a;
    d.time_steps = a.time_steps + 1; // another T is another network: never one cache entry
    EXPECT_EQ(meeting01::ga::genome_key(a), meeting01::ga::genome_key(b));
    EXPECT_NE(meeting01::ga::genome_key(a), meeting01::ga::genome_key(c));
    EXPECT_NE(meeting01::ga::genome_key(a), meeting01::ga::genome_key(d));
    EXPECT_FALSE(a == d);
}

// The genome's T reaches the network it builds; a T the profile's range does not allow (a
// genome or checkpoint from another profile) is refused instead of built.
TEST(Meeting01GaGenome, ToAeConfigUsesTheGenomesTimeSteps)
{
    meeting01::Meeting01Config cfg;
    cfg.model.loss_type = "mse";
    cfg.model.encoder_layer_spec = {"linear:64:leaky", "linear:32:identity"};
    cfg.model.decoder_layer_spec = {"linear:64:leaky", "linear:output:identity"};
    cfg.dataset.window_size = 256;
    cfg.model.time_steps = 16;
    cfg.evaluation.ga.snn.time_steps_min = 2;
    cfg.evaluation.ga.snn.time_steps_max = 32;

    const Genome g{{64, 32}, "direct", "dense", 1.0f, 0.9f, 7};
    EXPECT_EQ(meeting01::ga::to_ae_config(g, cfg).time_steps, 7);
    EXPECT_EQ(meeting01::with_snn_time_steps(cfg, 7).model.time_steps, 7);
    EXPECT_EQ(cfg.model.time_steps, 16) << "the profile's own value is left alone";

    const Genome outside{{64, 32}, "direct", "dense", 1.0f, 0.9f, 33};
    EXPECT_THROW((void) meeting01::ga::to_ae_config(outside, cfg), std::invalid_argument);

    // Both bounds 0: T is not searched, the range is model.time_steps alone.
    cfg.evaluation.ga.snn.time_steps_min = 0;
    cfg.evaluation.ga.snn.time_steps_max = 0;
    EXPECT_EQ(meeting01::snn_time_steps_range(cfg), (std::pair<int, int>{16, 16}));
    EXPECT_THROW((void) meeting01::ga::to_ae_config(g, cfg), std::invalid_argument);
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
    ind.genome = Genome{{72, 24}, "latency", "conv1d", 0.6f, 0.77f, 11};
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
    EXPECT_EQ(back.genome.time_steps, ind.genome.time_steps);
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

    meeting01::ga::append_cache_entry(path, a, 7u);
    meeting01::ga::append_cache_entry(path, b, 7u);

    const auto loaded =
        meeting01::ga::load_cache_entries<meeting01::ga::Meeting01GaIndividual>(path, 7u);
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
    meeting01::ga::append_cache_entry(path, a, 7u);

    {
        std::ofstream f(path, std::ios::app);
        f << "{\"genome\":{\"encoder_widths\":[1"; // torn: no closing braces
    }

    const auto loaded =
        meeting01::ga::load_cache_entries<meeting01::ga::Meeting01GaIndividual>(path, 7u);
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded[0].genome.encoder_widths, a.genome.encoder_widths);

    std::filesystem::remove(path);
}

// The torn line has no '\n'. Left in the file, the resumed search's next append would be
// glued onto it ("{\"genome\":{...[1{\"genome\":...}\n"), and once a second entry follows,
// that glued line is no longer the last one: the next resume refuses the whole cache as
// corrupt. So dropping the torn line must also remove it from the file.
TEST(Meeting01GaCheckpoint, TornTrailingCacheLineDoesNotCorruptTheNextResume)
{
    using Ind = meeting01::ga::Meeting01GaIndividual;
    const std::string path =
        std::filesystem::temp_directory_path() / "meeting01_ga_gtest_torn_then_resume.jsonl";
    std::filesystem::remove(path);

    Ind a, b, c;
    a.genome = Genome{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    b.genome = Genome{{48, 24}, "direct", "dense", 1.0f, 0.9f};
    c.genome = Genome{{32, 16}, "direct", "dense", 1.0f, 0.9f};
    meeting01::ga::append_cache_entry(path, a, 7u);
    {
        std::ofstream f(path, std::ios::app);
        f << "{\"genome\":{\"encoder_widths\":[1"; // the crash: no closing braces, no '\n'
    }

    ASSERT_EQ(meeting01::ga::load_cache_entries<Ind>(path, 7u).size(), 1u); // first resume
    meeting01::ga::append_cache_entry(path, b, 7u); // the resumed search scores two genomes,
    meeting01::ga::append_cache_entry(path, c, 7u); // then is interrupted again

    const auto loaded = meeting01::ga::load_cache_entries<Ind>(path, 7u); // second resume
    ASSERT_EQ(loaded.size(), 3u);
    EXPECT_EQ(loaded[1].genome.encoder_widths, b.genome.encoder_widths);
    EXPECT_EQ(loaded[2].genome.encoder_widths, c.genome.encoder_widths);

    std::filesystem::remove(path);
}

// Only an unparseable line can be a torn write (a cut JSON object always lacks its closing
// brace). A last line that parses but cannot be read as an individual is a schema change,
// and is refused like any other line instead of being dropped and retrained.
TEST(Meeting01GaCheckpoint, ParseableTrailingLineOfAnotherSchemaIsRefusedNotDropped)
{
    using Ind = meeting01::ga::Meeting01GaIndividual;
    const std::string path =
        std::filesystem::temp_directory_path() / "meeting01_ga_gtest_schema_cache.jsonl";
    std::filesystem::remove(path);

    Ind a;
    a.genome = Genome{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    meeting01::ga::append_cache_entry(path, a, 7u);
    {
        nlohmann::json j = meeting01::ga::individual_to_checkpoint_json(a);
        j.update(meeting01::ga::checkpoint_stamp(7u));
        j.erase("born_generation");
        std::ofstream f(path, std::ios::app);
        f << j.dump() << '\n';
    }
    EXPECT_THROW(meeting01::ga::load_cache_entries<Ind>(path, 7u), std::runtime_error);

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

    meeting01::ga::save_generation_checkpoint(dir, tag, 2, rng, parents, 7u);
    EXPECT_TRUE(meeting01::ga::state_checkpoint_exists(dir, tag));

    const auto ck = meeting01::ga::load_generation_checkpoint<meeting01::ga::Meeting01GaIndividual>(
        dir, tag, 7u);
    EXPECT_EQ(ck.generation, 2);
    ASSERT_EQ(ck.parents.size(), 2u);
    EXPECT_EQ(ck.parents[0].genome.encoder_widths, parents[0].genome.encoder_widths);
    EXPECT_EQ(ck.parents[1].genome.encoder_widths, parents[1].genome.encoder_widths);

    meeting01::ga::remove_checkpoint_artifacts(dir, tag);
    EXPECT_FALSE(meeting01::ga::state_checkpoint_exists(dir, tag));
}

// A resume must not rank genomes scored under one configuration against genomes scored
// under another: a checkpoint with another stamp, or none (a binary before 2026-11-02), is
// refused -- not dropped, and not mistaken for a torn trailing line.
TEST(Meeting01GaCheckpoint, ResumeRefusesAnotherConfigurationsCheckpoint)
{
    using Ind = meeting01::ga::Meeting01GaIndividual;
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string tag = "gtest_meeting01_ga_stamp";
    meeting01::ga::remove_checkpoint_artifacts(dir, tag);

    std::mt19937 rng(99);
    std::vector<Ind> parents(1);
    parents[0].genome = Genome{{64, 32}, "direct", "dense", 1.0f, 0.9f};
    parents[0].objectives = {0.5, 100.0};
    meeting01::ga::save_generation_checkpoint(dir, tag, 1, rng, parents, 7u);
    EXPECT_NO_THROW(meeting01::ga::load_generation_checkpoint<Ind>(dir, tag, 7u));
    EXPECT_THROW(meeting01::ga::load_generation_checkpoint<Ind>(dir, tag, 8u), std::runtime_error);

    const std::string cache = meeting01::ga::checkpoint_cache_path(dir, tag);
    meeting01::ga::append_cache_entry(cache, parents[0], 7u);
    EXPECT_EQ(meeting01::ga::load_cache_entries<Ind>(cache, 7u).size(), 1u);
    // Last line, parseable, another stamp: refused, not dropped as a torn write.
    EXPECT_THROW(meeting01::ga::load_cache_entries<Ind>(cache, 8u), std::runtime_error);

    {
        std::ofstream f(cache, std::ios::app); // a pre-stamp line: no config_hash at all
        f << meeting01::ga::individual_to_checkpoint_json(parents[0]).dump() << '\n';
    }
    EXPECT_THROW(meeting01::ga::load_cache_entries<Ind>(cache, 7u), std::runtime_error);

    meeting01::ga::remove_checkpoint_artifacts(dir, tag);
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

namespace rescore
{
// Scores a member with its seed, so the mean shows exactly which seeds were averaged.
struct Ind
{
    int genome = 0;
    float val_mse = 0.0f;
    std::vector<double> objectives;
};
void evaluate_individual(Ind& ind,
    const meeting01::Meeting01Config&,
    const meeting01::DatasetSplit&,
    std::uint32_t seed,
    int,
    int)
{
    ind.val_mse = static_cast<float>(seed);
}
} // namespace rescore

TEST(Meeting01GaSearch, RescoreAveragesFreshSeedsOnlyNeverTheSearchScore)
{
    const meeting01::Meeting01Config cfg;
    const meeting01::DatasetSplit split;
    constexpr float kLuckySearchScore = -1000.0f; // would pull a mean that included it far down
    std::vector<rescore::Ind> front{{1, kLuckySearchScore, {kLuckySearchScore, 5.0}}};
    std::vector<std::uint32_t> seeds;
    const std::function<void(const rescore::Ind&)> on_eval = [&](const rescore::Ind& probe)
    { seeds.push_back(static_cast<std::uint32_t>(probe.val_mse)); };

    meeting01::ga::detail::rescore_front(front, cfg, split, 100u, 3, on_eval);

    EXPECT_EQ(
        seeds, (std::vector<std::uint32_t>{100u + 7919u, 100u + 2u * 7919u, 100u + 3u * 7919u}))
        << "three trainings, none on the search's seed";
    EXPECT_FLOAT_EQ(front[0].val_mse, 100.0f + 2.0f * 7919.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(front[0].objectives[0]), front[0].val_mse);
}

// Grid 5321956 seeded the gene draws with the run seed alone: the searches of seed 42 on
// FSDD fold 0 and AudioMNIST fold 2 evaluated the same 47 genomes. Each (dataset, fold) now
// draws its own population; the same stream reproduces it, and an unnamed stream is refused.
TEST(Meeting01GaSearch, EachFoldDrawsItsOwnPopulation)
{
    const auto bounds = make_bounds();
    const auto population = [&](std::uint32_t seed, const std::string& stream)
    {
        std::mt19937 rng(meeting01::ga::search_stream_seed(seed, stream));
        std::vector<std::string> keys;
        keys.reserve(8);
        for (int i = 0; i < 8; ++i)
            keys.push_back(meeting01::ga::genome_key(meeting01::ga::random_genome(rng, bounds)));
        return keys;
    };
    const auto fsdd0 = population(42u, "fsdd_fold0");
    EXPECT_EQ(fsdd0, population(42u, "fsdd_fold0"));
    for (const auto& [seed, stream] : std::vector<std::pair<std::uint32_t, std::string>>{
             {42u, "fsdd_fold1"}, {42u, "audiomnist_fold0"}, {43u, "fsdd_fold0"}})
    {
        const auto other = population(seed, stream);
        int shared = 0;
        for (const auto& k : other)
            shared += static_cast<int>(std::count(fsdd0.begin(), fsdd0.end(), k));
        EXPECT_EQ(shared, 0) << seed << " " << stream << " repeats fsdd_fold0's genomes";
    }
    // Pinned: a change of the derivation would silently change every search.
    EXPECT_EQ(meeting01::ga::search_stream_seed(42u, "fsdd_fold0"), 318140021u);
    EXPECT_THROW(meeting01::ga::search_stream_seed(42u, ""), std::invalid_argument);
}

TEST(Meeting01GaSearch, OneWinnerSeedLeavesTheFrontAsItIs)
{
    const meeting01::Meeting01Config cfg;
    const meeting01::DatasetSplit split;
    std::vector<rescore::Ind> front{{1, 0.5f, {0.5, 5.0}}};
    int calls = 0;
    const std::function<void(const rescore::Ind&)> on_eval = [&](const rescore::Ind&) { ++calls; };
    meeting01::ga::detail::rescore_front(front, cfg, split, 100u, 1, on_eval);
    EXPECT_EQ(calls, 0);
    EXPECT_FLOAT_EQ(front[0].val_mse, 0.5f);
}

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
    cfg.dataset.cv_fold = 0;
    cfg.dataset.cv_num_folds = 6;
    cfg.training.samples_per_batch = 1;
    cfg.training.epochs = 1;
    cfg.training.early_stop_patience = 0;
    cfg.training.early_stop_min_skill = 0.1f;
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
    cfg.training.early_stop_min_skill = 0.1f;
    cfg.training.learning_rate = 0.001f;
    cfg.model.encoder_layer_spec = {"linear:16:leaky", "linear:8:identity"};
    cfg.model.decoder_layer_spec = {"linear:8:leaky", "linear:output:identity"};
    cfg.model.time_steps = 1;
    cfg.evaluation.datasets = {"fsdd"};
    cfg.evaluation.encodings = {"direct"};
    cfg.evaluation.snn_architectures = {"dense"};

    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(Meeting01GaSearch, ConfigChecksTheTimeStepsRange)
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
    cfg.training.early_stop_min_skill = 0.1f;
    cfg.training.learning_rate = 0.001f;
    cfg.model.encoder_layer_spec = {"linear:16:leaky", "linear:8:identity"};
    cfg.model.decoder_layer_spec = {"linear:8:leaky", "linear:output:identity"};
    cfg.evaluation.datasets = {"fsdd"};
    cfg.evaluation.encodings = {"direct"};
    cfg.evaluation.snn_architectures = {"dense"};
    cfg.dataset.cv_fold = 0;
    auto& ga = cfg.evaluation.ga.snn;

    ga.time_steps_min = 2;
    ga.time_steps_max = 32;
    EXPECT_NO_THROW(cfg.validate());
    ga.time_steps_min = 0; // one bound alone: not "unset", a half-written profile
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    ga.time_steps_min = 1; // one step: no time axis
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    ga.time_steps_min = 33; // min above max
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    // alpha and v_th reach the neuron unclamped (make_snn_cfg throws instead), so the
    // profile's ranges must stay where beta = alpha holds and the threshold is usable.
    ga.time_steps_min = 2;
    ga.alpha_max = 0.9999f;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    ga.alpha_max = 0.99f;
    ga.voltage_threshold_min = 1e-4f;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    ga.voltage_threshold_min = 0.5f;
    EXPECT_NO_THROW(cfg.validate());
}

// Until 2026-11-02 make_snn_cfg clamped alpha into [0.001, 0.999] and v_th to >= 0.001 in
// silence: a genome outside them trained a neuron with another beta / threshold than the
// one its manifest reports. It now refuses.
TEST(Meeting01GaSearch, MakeSnnCfgRefusesAGeneItCannotUseAsGiven)
{
    meeting01::Meeting01Config cfg;
    cfg.dataset.window_size = 64;
    cfg.model.loss_type = "mse";
    cfg.model.time_steps = 4;
    cfg.model.latent_dim = 8;
    const std::vector<int> widths{16, 8};
    EXPECT_NO_THROW(meeting01::make_snn_cfg(cfg, 0.9f, 1.0f, widths));
    EXPECT_THROW(meeting01::make_snn_cfg(cfg, 0.9995f, 1.0f, widths), std::invalid_argument);
    EXPECT_THROW(meeting01::make_snn_cfg(cfg, 0.0f, 1.0f, widths), std::invalid_argument);
    EXPECT_THROW(meeting01::make_snn_cfg(cfg, 0.9f, 1e-4f, widths), std::invalid_argument);

    const auto ae = meeting01::make_snn_cfg(cfg, 0.9f, 1.0f, widths);
    EXPECT_FLOAT_EQ(std::exp(-1.0f / (ae.resistance * ae.capacitance)), 0.9f); // beta = alpha
    EXPECT_FLOAT_EQ(ae.voltage_threshold, 1.0f);
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
