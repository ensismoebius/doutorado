// meeting01_transformer_ga_gtest.cpp — coverage for the Transformer-AE architecture
// search (Meeting01TransformerGaGenome/Fitness): genome repair/crossover/mutate stay
// legal AND keep the d_model % n_heads == 0 invariant multi-head attention requires
// (the one constraint no other family's genome has), to_transformer_cfg renders
// overrides correctly (including the window_size*time_steps seq_len formula — getting
// this wrong doesn't just under-cost the MAC estimate like LSTM/GRU, it overruns the
// positional-encoding buffer and crashes), and the generic checkpoint layer round-trips
// TransformerGaIndividual.
//
// Mirrors meeting01_ga_gtest.cpp's coverage shape for the SNN genome.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <random>

#include "Meeting01Config.hpp"
#include "Meeting01GaCheckpoint.hpp"
#include "Meeting01Training.hpp"
#include "Meeting01TransformerGaFitness.hpp"
#include "Meeting01TransformerGaGenome.hpp"

namespace
{

using meeting01::ga::TransformerGenome;
using meeting01::ga::TransformerGenomeBounds;

TransformerGenomeBounds make_bounds()
{
    TransformerGenomeBounds b;
    b.min_d_model = 16;
    b.max_d_model = 128;
    b.head_choices = {1, 2, 4, 8};
    b.min_layers = 1;
    b.max_layers = 4;
    b.min_d_ff = 32;
    b.max_d_ff = 256;
    b.encoding_choices = {"direct", "poisson", "latency"};
    return b;
}

TEST(Meeting01TransformerGaGenome, RandomGenomeIsLegal)
{
    std::mt19937 rng(1);
    const auto bounds = make_bounds();
    for (int trial = 0; trial < 200; ++trial)
    {
        const TransformerGenome g = meeting01::ga::random_genome(rng, bounds);
        EXPECT_NE(std::find(bounds.head_choices.begin(), bounds.head_choices.end(), g.n_heads),
            bounds.head_choices.end());
        EXPECT_GE(g.n_layers, bounds.min_layers);
        EXPECT_LE(g.n_layers, bounds.max_layers);
        EXPECT_GE(g.d_ff, bounds.min_d_ff);
        EXPECT_LE(g.d_ff, bounds.max_d_ff);
        EXPECT_GE(g.d_model, bounds.min_d_model);
        EXPECT_LE(g.d_model, bounds.max_d_model);
        // The one constraint unique to this family: multi-head attention requires an
        // exact division, not an approximation.
        EXPECT_EQ(g.d_model % g.n_heads, 0) << "d_model=" << g.d_model << " n_heads=" << g.n_heads;
        EXPECT_NE(
            std::find(bounds.encoding_choices.begin(), bounds.encoding_choices.end(), g.encoding),
            bounds.encoding_choices.end());
    }
}

TEST(Meeting01TransformerGaGenome, RepairFixesIllegalHeadCountAndDivisibility)
{
    const auto bounds = make_bounds();
    // n_heads=3 is not in head_choices at all, and 100 is not a multiple of the
    // repaired n_heads — repair must fix both independently.
    TransformerGenome g{/*d_model=*/100, /*n_heads=*/3, /*n_layers=*/2, /*d_ff=*/64, "direct"};
    meeting01::ga::repair_transformer(g, bounds);

    EXPECT_NE(std::find(bounds.head_choices.begin(), bounds.head_choices.end(), g.n_heads),
        bounds.head_choices.end());
    EXPECT_EQ(g.d_model % g.n_heads, 0);
    EXPECT_GE(g.d_model, bounds.min_d_model);
    EXPECT_LE(g.d_model, bounds.max_d_model);
}

TEST(Meeting01TransformerGaGenome, RepairClampsLayersAndFeedForwardWidth)
{
    const auto bounds = make_bounds();
    TransformerGenome g{64, 4, /*n_layers=*/999, /*d_ff=*/-5, "direct"};
    meeting01::ga::repair_transformer(g, bounds);
    EXPECT_EQ(g.n_layers, bounds.max_layers);
    EXPECT_EQ(g.d_ff, bounds.min_d_ff);
}

TEST(Meeting01TransformerGaGenome, CrossoverAlwaysStaysDivisible)
{
    std::mt19937 rng(7);
    const auto bounds = make_bounds();
    const TransformerGenome a{128, 8, 4, 256, "direct"};
    const TransformerGenome b{16, 1, 1, 32, "latency"};
    for (int trial = 0; trial < 100; ++trial)
    {
        const TransformerGenome c = meeting01::ga::crossover(a, b, rng, bounds);
        EXPECT_EQ(c.d_model % c.n_heads, 0);
        EXPECT_GE(c.d_model, bounds.min_d_model);
        EXPECT_LE(c.d_model, bounds.max_d_model);
        EXPECT_GE(c.n_layers, bounds.min_layers);
        EXPECT_LE(c.n_layers, bounds.max_layers);
        EXPECT_GE(c.d_ff, bounds.min_d_ff);
        EXPECT_LE(c.d_ff, bounds.max_d_ff);
    }
}

TEST(Meeting01TransformerGaGenome, MutationCanChangeEveryFieldAndStaysDivisible)
{
    std::mt19937 rng(13);
    const auto bounds = make_bounds();
    bool d_model_changed = false;
    bool n_heads_changed = false;
    bool n_layers_changed = false;
    bool d_ff_changed = false;
    bool encoding_changed = false;
    for (int trial = 0; trial < 500; ++trial)
    {
        TransformerGenome g{64, 4, 2, 128, "direct"};
        meeting01::ga::mutate(g, rng, bounds, /*prob=*/0.5);

        EXPECT_EQ(g.d_model % g.n_heads, 0) << "d_model=" << g.d_model << " n_heads=" << g.n_heads;
        EXPECT_GE(g.n_layers, bounds.min_layers);
        EXPECT_LE(g.n_layers, bounds.max_layers);
        EXPECT_GE(g.d_ff, bounds.min_d_ff);
        EXPECT_LE(g.d_ff, bounds.max_d_ff);

        if (g.d_model != 64) d_model_changed = true;
        if (g.n_heads != 4) n_heads_changed = true;
        if (g.n_layers != 2) n_layers_changed = true;
        if (g.d_ff != 128) d_ff_changed = true;
        if (g.encoding != "direct") encoding_changed = true;
    }
    EXPECT_TRUE(d_model_changed) << "500 trials at prob=0.5 never changed d_model";
    EXPECT_TRUE(n_heads_changed) << "500 trials at prob=0.5 never changed n_heads";
    EXPECT_TRUE(n_layers_changed) << "500 trials at prob=0.5 never changed n_layers";
    EXPECT_TRUE(d_ff_changed) << "500 trials at prob=0.5 never changed d_ff";
    EXPECT_TRUE(encoding_changed) << "500 trials at prob=0.5 never changed encoding";
}

TEST(Meeting01TransformerGaGenome, GenomeKeyIsStableAndDiscriminating)
{
    const TransformerGenome a{64, 4, 2, 128, "direct"};
    const TransformerGenome b = a;
    TransformerGenome c = a;
    c.d_ff = 129;
    EXPECT_EQ(meeting01::ga::genome_key(a), meeting01::ga::genome_key(b));
    EXPECT_NE(meeting01::ga::genome_key(a), meeting01::ga::genome_key(c));
}

meeting01::Meeting01Config base_cfg()
{
    meeting01::Meeting01Config cfg;
    cfg.model.loss_type = "mse";
    cfg.model.encoder_layer_spec = {"linear:64:leaky", "linear:32:identity"};
    cfg.model.decoder_layer_spec = {"linear:64:leaky", "linear:output:identity"};
    cfg.model.lstm_frame_size = 8;
    cfg.model.time_steps = 4;
    cfg.model.latent_dim = 32;
    cfg.dataset.window_size = 256;
    return cfg;
}

// Regression anchor for the 2026-09-22 seq_len fix: unlike LSTM/GRU (which infer
// seq_len dynamically at forward time), TransformerAutoencoder sizes a FIXED
// positional-encoding buffer from this field once at construction — a wrong seq_len
// here doesn't just under-cost the MAC estimate, it makes the very first real forward
// pass throw "Block indices out of range". The genome-driven path must use the same
// (window_size * time_steps) / lstm_frame_size formula as the profile-driven one.
TEST(Meeting01TransformerGaGenome, ToTransformerCfgUsesWindowTimesTimeStepsForSeqLen)
{
    const auto cfg = base_cfg();
    const TransformerGenome g{32, 8, 3, 64, "poisson"};
    const auto arch = meeting01::ga::to_transformer_cfg(g, cfg);

    EXPECT_EQ(arch.input_size, cfg.model.lstm_frame_size);
    EXPECT_EQ(
        arch.seq_len, (cfg.dataset.window_size * cfg.model.time_steps) / cfg.model.lstm_frame_size);
    EXPECT_EQ(arch.d_model, 32);
    EXPECT_EQ(arch.n_heads, 8);
    EXPECT_EQ(arch.n_layers, 3);
    EXPECT_EQ(arch.d_ff, 64);
    // latent_dim is never genome-driven — every family compared at the same
    // compression ratio (user decision, 2026-09-22).
    EXPECT_EQ(arch.latent_size, cfg.model.latent_dim);
}

TEST(Meeting01TransformerGaGenome, ToTransformerCfgZeroOverrideFallsBackToProfile)
{
    auto cfg = base_cfg();
    cfg.model.transformer_d_model = 48;
    cfg.model.transformer_heads = 6;
    cfg.model.transformer_layers = 5;
    cfg.model.transformer_d_ff = 96;

    // A default-constructed genome's fields are non-zero, so exercise the override
    // convention directly via make_transformer_cfg(cfg, 0, 0, 0, 0) — the same "0 means
    // use the profile field" contract to_transformer_cfg relies on for every family.
    const auto arch = meeting01::make_transformer_cfg(cfg, 0, 0, 0, 0);
    EXPECT_EQ(arch.d_model, 48);
    EXPECT_EQ(arch.n_heads, 6);
    EXPECT_EQ(arch.n_layers, 5);
    EXPECT_EQ(arch.d_ff, 96);
}

TEST(Meeting01GaCheckpoint, TransformerIndividualJsonRoundTripPreservesEveryField)
{
    meeting01::ga::TransformerGaIndividual ind;
    ind.genome = TransformerGenome{64, 4, 3, 128, "latency"};
    ind.val_mse = 0.3456f;
    ind.param_count = 8181;
    ind.inference_cost = 55555;
    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {0.3456, 55555.0};
    ind.born_generation = 2;

    const auto j = meeting01::ga::individual_to_checkpoint_json(ind);
    const auto back =
        meeting01::ga::individual_from_checkpoint_json<meeting01::ga::TransformerGaIndividual>(j);

    EXPECT_EQ(back.genome.d_model, ind.genome.d_model);
    EXPECT_EQ(back.genome.n_heads, ind.genome.n_heads);
    EXPECT_EQ(back.genome.n_layers, ind.genome.n_layers);
    EXPECT_EQ(back.genome.d_ff, ind.genome.d_ff);
    EXPECT_EQ(back.genome.encoding, ind.genome.encoding);
    EXPECT_FLOAT_EQ(back.val_mse, ind.val_mse);
    EXPECT_EQ(back.param_count, ind.param_count);
    EXPECT_EQ(back.inference_cost, ind.inference_cost);
    EXPECT_EQ(back.objectives, ind.objectives);
    EXPECT_EQ(back.born_generation, ind.born_generation);
}

TEST(Meeting01GaCheckpoint, TransformerCacheAppendAndReloadRoundTrips)
{
    const std::string path =
        std::filesystem::temp_directory_path() / "meeting01_transformer_ga_gtest_cache.jsonl";
    std::filesystem::remove(path);

    meeting01::ga::TransformerGaIndividual a;
    a.genome = TransformerGenome{64, 4, 2, 128, "direct"};
    a.objectives = {0.5, 100.0};
    meeting01::ga::TransformerGaIndividual b;
    b.genome = TransformerGenome{32, 8, 1, 32, "poisson"};
    b.objectives = {0.3, 50.0};

    meeting01::ga::append_cache_entry(path, a);
    meeting01::ga::append_cache_entry(path, b);

    const auto loaded =
        meeting01::ga::load_cache_entries<meeting01::ga::TransformerGaIndividual>(path);
    ASSERT_EQ(loaded.size(), 2u);
    EXPECT_EQ(loaded[0].genome.d_model, a.genome.d_model);
    EXPECT_EQ(loaded[1].genome.d_model, b.genome.d_model);

    std::filesystem::remove(path);
}

TEST(Meeting01GaCheckpoint, TransformerGenerationCheckpointRoundTrips)
{
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string tag = "gtest_meeting01_transformer_ga";
    meeting01::ga::remove_checkpoint_artifacts(dir, tag);

    std::mt19937 rng(99);
    std::vector<meeting01::ga::TransformerGaIndividual> parents(2);
    parents[0].genome = TransformerGenome{64, 4, 2, 128, "direct"};
    parents[0].objectives = {0.5, 100.0};
    parents[1].genome = TransformerGenome{16, 1, 1, 32, "latency"};
    parents[1].objectives = {0.3, 30.0};

    meeting01::ga::save_generation_checkpoint(dir, tag, 3, rng, parents);
    EXPECT_TRUE(meeting01::ga::state_checkpoint_exists(dir, tag));

    const auto ck =
        meeting01::ga::load_generation_checkpoint<meeting01::ga::TransformerGaIndividual>(dir, tag);
    EXPECT_EQ(ck.generation, 3);
    ASSERT_EQ(ck.parents.size(), 2u);
    EXPECT_EQ(ck.parents[0].genome.d_model, parents[0].genome.d_model);
    EXPECT_EQ(ck.parents[1].genome.d_model, parents[1].genome.d_model);

    meeting01::ga::remove_checkpoint_artifacts(dir, tag);
    EXPECT_FALSE(meeting01::ga::state_checkpoint_exists(dir, tag));
}

} // namespace
