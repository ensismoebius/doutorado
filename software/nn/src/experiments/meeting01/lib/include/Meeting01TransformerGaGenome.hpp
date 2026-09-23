#pragma once

#include <random>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "models/transformer/TransformerAutoencoderConfig.hpp"
#include "nlohmann/json.hpp"

namespace meeting01::ga
{

// `head_choices` is the legal POOL n_heads is drawn from and repaired into — multi-head
// attention needs d_model % n_heads == 0, so n_heads can't be an arbitrary integer in a
// range the way hidden_size/d_ff are; it has to come from a short list of divisor
// candidates that repair_transformer then reconciles against d_model.
struct TransformerGenomeBounds
{
    int min_d_model = 16;
    int max_d_model = 128;
    std::vector<int> head_choices = {1, 2, 4, 8};
    int min_layers = 1;
    int max_layers = 4;
    int min_d_ff = 32;
    int max_d_ff = 256;
    std::vector<std::string> encoding_choices;
};

// latent_dim is NEVER a gene (see Meeting01RecurrentGaGenome.hpp's identical note) —
// fixed at cfg.model.latent_dim for every family.
struct TransformerGenome
{
    int d_model = 64;
    int n_heads = 4;
    int n_layers = 2;
    int d_ff = 128;
    std::string encoding = "direct";
};

// Enforces d_model % n_heads == 0: picks the legal n_heads closest to the current
// value, then adjusts d_model to the nearest multiple of that n_heads that still fits
// [min_d_model, max_d_model]. Never left to chance — applied after random draw,
// crossover and mutation alike, same discipline repair_widths uses to keep the SNN's
// encoder_widths strictly decreasing (Meeting01GaGenome.cpp).
void repair_transformer(TransformerGenome& g, const TransformerGenomeBounds& bounds);

TransformerGenome random_genome(std::mt19937& rng, const TransformerGenomeBounds& bounds);
TransformerGenome crossover(const TransformerGenome& a,
    const TransformerGenome& b,
    std::mt19937& rng,
    const TransformerGenomeBounds& bounds);
void mutate(
    TransformerGenome& g, std::mt19937& rng, const TransformerGenomeBounds& bounds, double prob);

auto to_transformer_cfg(const TransformerGenome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::transformer::TransformerAutoencoderConfig;

auto genome_key(const TransformerGenome& g) -> std::string;

auto genome_to_json(const TransformerGenome& g) -> nlohmann::json;
void genome_from_json(const nlohmann::json& j, TransformerGenome& out);

} // namespace meeting01::ga
