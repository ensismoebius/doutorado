// meeting01_recurrent_ga_gtest.cpp — coverage for the shared LSTM-AE / GRU-AE
// architecture search (Meeting01RecurrentGaGenome/Fitness): genome repair/crossover/
// mutate stay legal, to_lstm_cfg/to_gru_cfg render overrides correctly (including the
// window_size*time_steps seq_len formula and the fixed-latent_dim rule), and the
// generic checkpoint layer round-trips both LstmGaIndividual and GruGaIndividual.
//
// Mirrors meeting01_ga_gtest.cpp's coverage shape for the SNN genome — see that file's
// header comment for the rationale of each category.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <random>

#include "Meeting01Config.hpp"
#include "Meeting01GaCheckpoint.hpp"
#include "Meeting01RecurrentGaFitness.hpp"
#include "Meeting01RecurrentGaGenome.hpp"

namespace
{

using meeting01::ga::RecurrentGenome;
using meeting01::ga::RecurrentGenomeBounds;

RecurrentGenomeBounds make_bounds()
{
    RecurrentGenomeBounds b;
    b.min_hidden = 8;
    b.max_hidden = 256;
    b.min_layers = 1;
    b.max_layers = 3;
    b.encoding_choices = {"direct", "poisson", "latency"};
    return b;
}

TEST(Meeting01RecurrentGaGenome, RandomGenomeIsLegal)
{
    std::mt19937 rng(1);
    const auto bounds = make_bounds();
    for (int trial = 0; trial < 200; ++trial)
    {
        const RecurrentGenome g = meeting01::ga::random_genome(rng, bounds);
        EXPECT_GE(g.hidden_size, bounds.min_hidden);
        EXPECT_LE(g.hidden_size, bounds.max_hidden);
        EXPECT_GE(g.num_layers, bounds.min_layers);
        EXPECT_LE(g.num_layers, bounds.max_layers);
        EXPECT_NE(
            std::find(bounds.encoding_choices.begin(), bounds.encoding_choices.end(), g.encoding),
            bounds.encoding_choices.end());
    }
}

TEST(Meeting01RecurrentGaGenome, RepairClampsOutOfRangeValues)
{
    const auto bounds = make_bounds();
    RecurrentGenome g{/*hidden_size=*/100000, /*num_layers=*/-7, "direct"};
    meeting01::ga::repair_recurrent(g, bounds);
    EXPECT_EQ(g.hidden_size, bounds.max_hidden);
    EXPECT_EQ(g.num_layers, bounds.min_layers);

    RecurrentGenome g2{/*hidden_size=*/-5, /*num_layers=*/999, "direct"};
    meeting01::ga::repair_recurrent(g2, bounds);
    EXPECT_EQ(g2.hidden_size, bounds.min_hidden);
    EXPECT_EQ(g2.num_layers, bounds.max_layers);
}

TEST(Meeting01RecurrentGaGenome, CrossoverStaysWithinBounds)
{
    std::mt19937 rng(7);
    const auto bounds = make_bounds();
    const RecurrentGenome a{200, 3, "direct"};
    const RecurrentGenome b{16, 1, "latency"};
    for (int trial = 0; trial < 50; ++trial)
    {
        const RecurrentGenome c = meeting01::ga::crossover(a, b, rng, bounds);
        EXPECT_GE(c.hidden_size, bounds.min_hidden);
        EXPECT_LE(c.hidden_size, bounds.max_hidden);
        EXPECT_GE(c.num_layers, bounds.min_layers);
        EXPECT_LE(c.num_layers, bounds.max_layers);
        // Inherited by coin flip from one parent or the other, never averaged/invented.
        EXPECT_TRUE(c.hidden_size == a.hidden_size || c.hidden_size == b.hidden_size);
        EXPECT_TRUE(c.num_layers == a.num_layers || c.num_layers == b.num_layers);
        EXPECT_TRUE(c.encoding == a.encoding || c.encoding == b.encoding);
    }
}

TEST(Meeting01RecurrentGaGenome, MutationCanChangeEveryFieldAndStaysLegal)
{
    std::mt19937 rng(13);
    const auto bounds = make_bounds();
    bool hidden_changed = false;
    bool layers_changed = false;
    bool encoding_changed = false;
    for (int trial = 0; trial < 500; ++trial)
    {
        RecurrentGenome g{64, 1, "direct"};
        meeting01::ga::mutate(g, rng, bounds, /*prob=*/0.5);

        EXPECT_GE(g.hidden_size, bounds.min_hidden);
        EXPECT_LE(g.hidden_size, bounds.max_hidden);
        EXPECT_GE(g.num_layers, bounds.min_layers);
        EXPECT_LE(g.num_layers, bounds.max_layers);
        EXPECT_NE(
            std::find(bounds.encoding_choices.begin(), bounds.encoding_choices.end(), g.encoding),
            bounds.encoding_choices.end());

        if (g.hidden_size != 64) hidden_changed = true;
        if (g.num_layers != 1) layers_changed = true;
        if (g.encoding != "direct") encoding_changed = true;
    }
    EXPECT_TRUE(hidden_changed) << "500 trials at prob=0.5 never changed hidden_size";
    EXPECT_TRUE(layers_changed) << "500 trials at prob=0.5 never changed num_layers";
    EXPECT_TRUE(encoding_changed) << "500 trials at prob=0.5 never changed encoding";
}

TEST(Meeting01RecurrentGaGenome, GenomeKeyIsStableAndDiscriminating)
{
    const RecurrentGenome a{64, 2, "direct"};
    const RecurrentGenome b = a;
    RecurrentGenome c = a;
    c.hidden_size = 65;
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

// Regression anchor for the 2026-09-22 seq_len fix (Meeting01Training.cpp): the encoder
// expands every window into a (time_steps, window_size) tensor BEFORE framing, so the
// real sequence length to_lstm_frames()/to_gru_frames() produce is
// (window_size * time_steps) / lstm_frame_size, not window_size / lstm_frame_size. A
// genome-driven config must inherit the same formula as the profile-driven one, since
// the genome never owns seq_len — only hidden_size/num_layers do.
TEST(Meeting01RecurrentGaGenome, ToLstmCfgUsesWindowTimesTimeStepsForSeqLen)
{
    const auto cfg = base_cfg();
    const RecurrentGenome g{96, 2, "poisson"};
    const auto arch = meeting01::ga::to_lstm_cfg(g, cfg);

    EXPECT_EQ(arch.input_size, cfg.model.lstm_frame_size);
    EXPECT_EQ(
        arch.seq_len, (cfg.dataset.window_size * cfg.model.time_steps) / cfg.model.lstm_frame_size);
    EXPECT_EQ(arch.hidden_size, 96);
    EXPECT_EQ(arch.num_layers, 2);
    // latent_dim is never genome-driven — every family compared at the same
    // compression ratio (user decision, 2026-09-22).
    EXPECT_EQ(arch.latent_size, cfg.model.latent_dim);
}

TEST(Meeting01RecurrentGaGenome, ToGruCfgUsesWindowTimesTimeStepsForSeqLen)
{
    const auto cfg = base_cfg();
    const RecurrentGenome g{48, 3, "latency"};
    const auto arch = meeting01::ga::to_gru_cfg(g, cfg);

    EXPECT_EQ(arch.input_size, cfg.model.lstm_frame_size);
    EXPECT_EQ(
        arch.seq_len, (cfg.dataset.window_size * cfg.model.time_steps) / cfg.model.lstm_frame_size);
    EXPECT_EQ(arch.hidden_size, 48);
    EXPECT_EQ(arch.num_layers, 3);
    EXPECT_EQ(arch.latent_size, cfg.model.latent_dim);
}

TEST(Meeting01GaCheckpoint, LstmIndividualJsonRoundTripPreservesEveryField)
{
    meeting01::ga::LstmGaIndividual ind;
    ind.genome = RecurrentGenome{128, 2, "latency"};
    ind.val_mse = 0.2345f;
    ind.param_count = 5151;
    ind.inference_cost = 77777;
    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {0.2345, 77777.0};
    ind.born_generation = 4;

    const auto j = meeting01::ga::individual_to_checkpoint_json(ind);
    const auto back =
        meeting01::ga::individual_from_checkpoint_json<meeting01::ga::LstmGaIndividual>(j);

    EXPECT_EQ(back.genome.hidden_size, ind.genome.hidden_size);
    EXPECT_EQ(back.genome.num_layers, ind.genome.num_layers);
    EXPECT_EQ(back.genome.encoding, ind.genome.encoding);
    EXPECT_FLOAT_EQ(back.val_mse, ind.val_mse);
    EXPECT_EQ(back.param_count, ind.param_count);
    EXPECT_EQ(back.inference_cost, ind.inference_cost);
    EXPECT_EQ(back.objectives, ind.objectives);
    EXPECT_EQ(back.born_generation, ind.born_generation);
}

TEST(Meeting01GaCheckpoint, GruIndividualJsonRoundTripPreservesEveryField)
{
    meeting01::ga::GruGaIndividual ind;
    ind.genome = RecurrentGenome{40, 1, "direct"};
    ind.val_mse = 0.6789f;
    ind.objectives = {0.6789, 12345.0};
    ind.born_generation = 1;

    const auto j = meeting01::ga::individual_to_checkpoint_json(ind);
    const auto back =
        meeting01::ga::individual_from_checkpoint_json<meeting01::ga::GruGaIndividual>(j);

    EXPECT_EQ(back.genome.hidden_size, ind.genome.hidden_size);
    EXPECT_EQ(back.genome.num_layers, ind.genome.num_layers);
    EXPECT_EQ(back.genome.encoding, ind.genome.encoding);
    EXPECT_FLOAT_EQ(back.val_mse, ind.val_mse);
    EXPECT_EQ(back.objectives, ind.objectives);
}

TEST(Meeting01GaCheckpoint, LstmCacheAppendAndReloadRoundTrips)
{
    const std::string path =
        std::filesystem::temp_directory_path() / "meeting01_recurrent_ga_gtest_cache.jsonl";
    std::filesystem::remove(path);

    meeting01::ga::LstmGaIndividual a;
    a.genome = RecurrentGenome{64, 1, "direct"};
    a.objectives = {0.5, 100.0};
    meeting01::ga::LstmGaIndividual b;
    b.genome = RecurrentGenome{160, 2, "poisson"};
    b.objectives = {0.3, 200.0};

    meeting01::ga::append_cache_entry(path, a);
    meeting01::ga::append_cache_entry(path, b);

    const auto loaded = meeting01::ga::load_cache_entries<meeting01::ga::LstmGaIndividual>(path);
    ASSERT_EQ(loaded.size(), 2u);
    EXPECT_EQ(loaded[0].genome.hidden_size, a.genome.hidden_size);
    EXPECT_EQ(loaded[1].genome.hidden_size, b.genome.hidden_size);

    std::filesystem::remove(path);
}

TEST(Meeting01GaCheckpoint, GruGenerationCheckpointRoundTrips)
{
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string tag = "gtest_meeting01_recurrent_ga";
    meeting01::ga::remove_checkpoint_artifacts(dir, tag);

    std::mt19937 rng(99);
    std::vector<meeting01::ga::GruGaIndividual> parents(2);
    parents[0].genome = RecurrentGenome{64, 1, "direct"};
    parents[0].objectives = {0.5, 100.0};
    parents[1].genome = RecurrentGenome{200, 3, "latency"};
    parents[1].objectives = {0.3, 300.0};

    meeting01::ga::save_generation_checkpoint(dir, tag, 2, rng, parents);
    EXPECT_TRUE(meeting01::ga::state_checkpoint_exists(dir, tag));

    const auto ck =
        meeting01::ga::load_generation_checkpoint<meeting01::ga::GruGaIndividual>(dir, tag);
    EXPECT_EQ(ck.generation, 2);
    ASSERT_EQ(ck.parents.size(), 2u);
    EXPECT_EQ(ck.parents[0].genome.hidden_size, parents[0].genome.hidden_size);
    EXPECT_EQ(ck.parents[1].genome.hidden_size, parents[1].genome.hidden_size);

    meeting01::ga::remove_checkpoint_artifacts(dir, tag);
    EXPECT_FALSE(meeting01::ga::state_checkpoint_exists(dir, tag));
}

} // namespace
