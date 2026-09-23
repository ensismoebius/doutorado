#pragma once

#include <random>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "nlohmann/json.hpp"

namespace meeting01::ga
{

// Bounds/legal-choice pool for random_genome/crossover/mutate. `encoding_choices` and
// `architecture_choices` are populated from the profile's already-validated
// evaluation.encodings / evaluation.snn_architectures lists (Meeting01Config.cpp
// check_evaluation whitelists both against {"direct","poisson","latency"} /
// {"dense","conv1d","recurrent"}) — the GA's categorical genes draw from exactly the
// same legal sets the grid search used, no new whitelist invented.
struct GenomeBounds
{
    int min_layers = 1;
    int max_layers = 4;
    int min_width = 4;
    int max_width = 128;
    float voltage_threshold_min = 0.1f;
    float voltage_threshold_max = 2.0f;
    float alpha_min = 0.5f;
    float alpha_max = 0.99f;
    std::vector<std::string> encoding_choices;
    std::vector<std::string> architecture_choices;
    // The bottleneck width — NEVER a gene, same rule as every other family (2026-09-22
    // decision, see Meeting01RecurrentGaGenome.hpp's identical note): fixed so every
    // family is compared at the same compression ratio. `repair_widths` forces
    // `encoder_widths.back()` to exactly this value; `min_width`/`max_width` bound only
    // the HIDDEN layers above it. Wired from cfg.model.latent_dim by the caller
    // (run_snn_ga_search) — this struct doesn't read Meeting01Config itself.
    int latent_dim = 32;
};

// One individual's genotype. Every axis the user asked to be free is a gene here:
// encoder_widths (layer count = size(), neurons/layer = each element, both free and
// under mutation/crossover), encoding, architecture (the input-transform selector),
// voltage_threshold, alpha. The decoder mirrors the encoder (reversed widths + output
// projection) rather than being independently evolved — confirmed with the user.
//
//   encoder_widths = {96, 40, 12}
//     encoder: window_size -> 96 -> 40 -> 12(latent)
//     decoder: 12   -> 40  -> 96 -> window_size(output)
struct Genome
{
    std::vector<int> encoder_widths = {64, 32}; // strictly decreasing; last = latent
    std::string encoding = "direct";
    std::string architecture = "dense";
    float voltage_threshold = 1.0f;
    float alpha = 0.9f;

    // Never call on an empty genome (random/repair guarantee at least one layer).
    [[nodiscard]] int latent() const
    {
        return encoder_widths.back();
    }
    [[nodiscard]] int depth() const
    {
        return static_cast<int>(encoder_widths.size());
    }

    bool operator==(const Genome& o) const noexcept
    {
        return encoder_widths == o.encoder_widths && encoding == o.encoding &&
               architecture == o.architecture && voltage_threshold == o.voltage_threshold &&
               alpha == o.alpha;
    }
};

// Force `encoder_widths` into a legal shape: every width clamped to
// [min_width, max_width], sorted strictly decreasing, duplicates broken by
// decrementing, truncated if strict decrease runs out of integer room. Result always
// has >= 1 layer. Ported from pga::repair_widths (GaGenome.cpp) — the width-list
// invariant is identical, only the bounds source differs.
void repair_widths(Genome& g, const GenomeBounds& bounds);

// Draw a random genome: random depth in [min_layers, max_layers], random
// strictly-decreasing widths, and a random draw from each categorical/continuous gene.
Genome random_genome(std::mt19937& rng, const GenomeBounds& bounds);

// Recombine two variable-length parents: child depth inherited from one parent (coin
// flip), each width position sampled from whichever parent has it, then repaired.
// encoding/architecture each inherited via coin flip; voltage_threshold/alpha via
// arithmetic mean of the two parents (no categorical precedent for continuous genes in
// paraconsistentGA, so this is new — a standard real-valued GA crossover).
Genome crossover(const Genome& a, const Genome& b, std::mt19937& rng, const GenomeBounds& bounds);

// Structural mutation (jitter/insert/delete a layer, ported from pga::mutate) plus
// categorical mutation (uniform reassignment among encoding_choices/
// architecture_choices) and continuous jitter (uniform redraw within
// [voltage_threshold_min,max] / [alpha_min,max]) — each gated by the same per-operator
// probability `prob`.
void mutate(Genome& g, std::mt19937& rng, const GenomeBounds& bounds, double prob);

// Build the AutoencoderConfig this genome represents: starts from
// make_snn_cfg(cfg, g.alpha, g.voltage_threshold, g.encoder_widths) — the profile's
// base config with only the genome-owned fields overridden (layer spec + R/C), so any
// field the profile sets and the genome doesn't own (loss_type, surrogate_gradient,
// branch/fusion specs, ...) is carried through untouched. Mirrors pga::to_ae_config's
// "override a copy, never rebuild from scratch" rule for the same reason: rebuilding
// from scratch previously caused ae_loss_type to be silently dropped (see GaGenome.hpp).
auto to_ae_config(const Genome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::autoencoder::AutoencoderConfig;

// Stable string key for checkpoint/cache lookups — identical genomes (including
// encoding/architecture/continuous genes) collapse to one training. Mirrors
// pga::genome_key (GaNsga2.cpp).
auto genome_key(const Genome& g) -> std::string;

// ADL hooks the generic checkpoint layer (Meeting01GaCheckpoint.hpp) calls unqualified
// on whatever concrete genome type an individual carries — `genome_to_json` resolves by
// argument type, `genome_from_json` writes into its out-param so the SAME call shape
// works for every family without the checkpoint code ever naming a genome type itself.
// These were private helpers inside Meeting01GaCheckpoint.cpp before the multi-family
// generalization; promoted here (same file as every other Genome-specific operation)
// once a second and third genome type existed and needed the identical hook.
auto genome_to_json(const Genome& g) -> nlohmann::json;
void genome_from_json(const nlohmann::json& j, Genome& out);

} // namespace meeting01::ga
