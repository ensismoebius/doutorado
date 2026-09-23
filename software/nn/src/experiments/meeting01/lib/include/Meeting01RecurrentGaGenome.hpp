#pragma once

#include <random>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "models/gru/GRUAutoencoderConfig.hpp"
#include "models/lstm/LSTMAutoencoderConfig.hpp"
#include "nlohmann/json.hpp"

namespace meeting01::ga
{

// Bounds for the shared LSTM-AE / GRU-AE architecture search. `encoding_choices` draws
// from the profile's evaluation.encodings list, same whitelist source as the SNN genome
// (Meeting01GaGenome.hpp) — one legal set for every family, no new whitelist invented.
struct RecurrentGenomeBounds
{
    int min_hidden = 8;
    int max_hidden = 256;
    int min_layers = 1;
    int max_layers = 3;
    std::vector<std::string> encoding_choices;
};

// LSTM-AE and GRU-AE vary over exactly the same axes (hidden width, depth, input
// encoding), so ONE genome serves both instead of two near-identical copies. Which
// recurrent cell a given search trains is decided by which evaluate_individual overload
// is called (Meeting01RecurrentGaFitness.hpp: LstmGaIndividual vs GruGaIndividual, two
// distinct types wrapping this same genome) — not a field here, because the user's
// decision was one search PER family, never a mixed population choosing its own cell.
//
// latent_dim is NEVER a gene, here or in any family (user decision, 2026-09-22): it
// stays fixed at cfg.model.latent_dim so every family is compared at the same
// compression ratio, not "whichever family got the more generous bottleneck".
struct RecurrentGenome
{
    int hidden_size = 64;
    int num_layers = 1;
    std::string encoding = "direct";

    bool operator==(const RecurrentGenome& o) const noexcept
    {
        return hidden_size == o.hidden_size && num_layers == o.num_layers && encoding == o.encoding;
    }
};

// Clamps hidden_size/num_layers into bounds. No structural invariant to repair beyond
// clamping (unlike the SNN's strictly-decreasing encoder_widths) — every (hidden_size,
// num_layers) pair inside bounds is a legal recurrent stack.
void repair_recurrent(RecurrentGenome& g, const RecurrentGenomeBounds& bounds);

RecurrentGenome random_genome(std::mt19937& rng, const RecurrentGenomeBounds& bounds);
RecurrentGenome crossover(const RecurrentGenome& a,
    const RecurrentGenome& b,
    std::mt19937& rng,
    const RecurrentGenomeBounds& bounds);
void mutate(
    RecurrentGenome& g, std::mt19937& rng, const RecurrentGenomeBounds& bounds, double prob);

// Renders the genome into the real model config, starting from the profile's base
// config and overriding only the genome-owned fields (hidden_size, num_layers) —
// mirrors Meeting01GaGenome::to_ae_config's "override a copy, never rebuild from
// scratch" rule, for the same reason: fields the genome doesn't own must survive
// untouched.
auto to_lstm_cfg(const RecurrentGenome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::lstm::LSTMAutoencoderConfig;
auto to_gru_cfg(const RecurrentGenome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::gru::GRUAutoencoderConfig;

auto genome_key(const RecurrentGenome& g) -> std::string;

// ADL hooks for the generic checkpoint layer (Meeting01GaCheckpoint.hpp) — see
// Meeting01GaGenome.hpp's identical pair for the full rationale.
auto genome_to_json(const RecurrentGenome& g) -> nlohmann::json;
void genome_from_json(const nlohmann::json& j, RecurrentGenome& out);

} // namespace meeting01::ga
