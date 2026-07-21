#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"

namespace pga
{

// Fixed flat input dimension of the Protocol (SNN-AE / ANN-AE) autoencoders.
// Mirrors thesis kAeInputFeatures (ThesisFeatureExtraction.cpp): the raw signal is
// average-pooled into this many bins before the AE. Used here only for the
// structural parameter/MAC proxies — it is NOT re-plumbed into the AE build, which
// still owns the real value.
inline constexpr int kAeInputFeatures = 256;

// Search-space bounds for the genome. Defaults mirror the phase00 AE profiles
// (tiny 16→8, small 32→16, base 64→32), extended so the GA can explore
// combinations phase00's hand-picked tiers did not, while staying on the same
// axes (see ga.md §5.1 — do not invent unrelated axes).
struct GenomeBounds
{
    std::vector<int> hidden_choices = {16, 32, 64, 128};
    std::vector<int> latent_choices = {8, 16, 32, 64};

    // SNN-only. Ignored for the ANN population (always "direct").
    std::vector<std::string> encoding_choices = {"direct", "latency", "poisson"};

    // When false (default, phase00 behavior) time_steps and voltage_threshold are
    // DERIVED from encoding, not searched. When true they become free genes drawn
    // from the ranges below — a declared expansion of the phase00 space (ga.md §5.1).
    bool evolve_temporal = false;
    std::vector<int> time_steps_choices = {1, 8, 16, 32};
    float voltage_threshold_min = 0.1f;
    float voltage_threshold_max = 1.0f;
};

// One individual's genotype: the phase00 AE architecture axes. `model` and
// `modality` are NOT genes — they are fixed per run (population-defining), so they
// live in GaConfig, not here.
struct Genome
{
    int hidden = 64;
    int latent = 32;

    // SNN population only. "direct" for the ANN population.
    std::string encoding = "direct";
    int time_steps = 1;
    float voltage_threshold = 1.0f;

    bool operator==(const Genome& o) const noexcept
    {
        return hidden == o.hidden && latent == o.latent && encoding == o.encoding &&
               time_steps == o.time_steps && voltage_threshold == o.voltage_threshold;
    }
};

// Couple time_steps + voltage_threshold to the encoding exactly as phase00 does:
//   direct           → (1,  1.0)
//   latency|poisson  → (16, 0.2)
// Applied unless bounds.evolve_temporal promotes them to free genes.
void apply_phase00_temporal_coupling(Genome& g);

// Draw a random genome. `is_snn` selects whether encoding/temporal genes are active
// (SNN) or clamped to the ANN defaults ("direct", 1, 1.0). latent < hidden is
// enforced (a bottleneck is the point of an autoencoder).
Genome random_genome(std::mt19937& rng, const GenomeBounds& bounds, bool is_snn);

// Uniform crossover of two parents, gene by gene. Repairs latent < hidden after.
Genome crossover(const Genome& a, const Genome& b, std::mt19937& rng, bool is_snn);

// Per-gene mutation with the given probability. Repairs latent < hidden after and
// re-applies temporal coupling unless evolve_temporal.
void mutate(Genome& g, std::mt19937& rng, const GenomeBounds& bounds, double prob, bool is_snn);

// Build the thesis AutoencoderConfig this genome represents. `model` is the
// population's model tag ("snn-ae" | "ann-ae"). For snn-ae the phase00-fixed
// firing-rate regularization (0.5 / 0.1 / 0.8) is applied. encoder/decoder specs are
// generated as phase00 does: ["linear:H:leaky","linear:L:identity"] /
// ["linear:H:leaky","linear:output:identity"].
thesis::ThesisConfig::AutoencoderConfig to_ae_config(const Genome& g, const std::string& model);

// ── Structural proxies (no training, no allocation) ──────────────────────────

// Trainable parameter count of the symmetric Protocol AE this genome builds
// (encoder 256→H→L, decoder L→H→256, biases included).
long estimated_params(const Genome& g);

// Encoder multiply-accumulates for ONE forward frame (256·H + H·L).
long encoder_macs_per_frame(const Genome& g);

// Total encoder inference cost proxy = encoder_macs_per_frame · time_steps. This is
// the deterministic secondary objective (minimize) and the pre-training screen input
// (ga.md §4). Monotonic in width, latent and temporal depth.
long inference_cost_proxy(const Genome& g);

} // namespace pga
