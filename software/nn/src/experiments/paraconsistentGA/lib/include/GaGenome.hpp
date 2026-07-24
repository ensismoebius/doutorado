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

// Search-space bounds for the genome.
//
// Depth and per-layer width are BOTH free genes: there are no pre-defined layer
// tiers. A genome is an arbitrary-length list of strictly decreasing encoder widths
// (the last one is the latent bottleneck); the decoder mirrors it. These bounds only
// keep the search finite and the latency proxy meaningful — they are ranges, not a
// menu of fixed shapes.
struct GenomeBounds
{
    // Number of encoder Linear stages, INCLUDING the latent stage. min_layers=1 is a
    // single input->latent projection (no hidden layer); larger values add hidden
    // layers. Because widths must strictly decrease, an achievable depth also needs
    // enough integer room: max_layers <= (max_width - min_width + 1), enforced by
    // GaConfig::validate.
    int min_layers = 1;
    int max_layers = 6;

    // Per-layer neuron count range. Every width (hidden and latent) is drawn from
    // [min_width, max_width]. min_width=1 permits a 1-neuron latent (as in the
    // "3->2->1" example); it applies to hidden layers too.
    int min_width = 1;
    int max_width = 128;

    // SNN-only. Ignored for the ANN population (always "direct").
    std::vector<std::string> encoding_choices = {"direct", "latency", "poisson"};

    // When false (default, phase00 behavior) time_steps and voltage_threshold are
    // DERIVED from encoding, not searched. When true they become free genes drawn
    // from the ranges below — a declared expansion of the phase00 space
    // (.wiki/Experiments/ParaconsistentGA-Design.md §5.1).
    bool evolve_temporal = false;
    std::vector<int> time_steps_choices = {1, 8, 16, 32};
    float voltage_threshold_min = 0.1f;
    float voltage_threshold_max = 1.0f;
};

// One individual's genotype. The architecture is fully free: `encoder_widths` is an
// arbitrary-length, strictly-decreasing list of neuron counts — layer count AND
// per-layer width both come from the DNA, with no pre-defined configuration. The last
// element is the latent (bottleneck) dimension; the decoder is the mirror image.
//
//   encoder_widths = {10, 5, 4, 2}
//     encoder: 256 -> 10 -> 5 -> 4 -> 2(latent)
//     decoder: 2   -> 4  -> 5 -> 10 -> 256(output)
//
// `model` and `modality` are NOT genes — they are fixed per run (population-defining),
// so they live in GaConfig, not here.
struct Genome
{
    std::vector<int> encoder_widths = {64, 32}; // strictly decreasing; last = latent

    // SNN population only. "direct" for the ANN population.
    std::string encoding = "direct";
    int time_steps = 1;
    float voltage_threshold = 1.0f;

    // Convenience accessors — never call on an empty genome (random/repair guarantee
    // at least one layer).
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
               time_steps == o.time_steps && voltage_threshold == o.voltage_threshold;
    }
};

// Couple time_steps + voltage_threshold to the encoding exactly as phase00 does:
//   direct           → (1,  1.0)
//   latency|poisson  → (16, 0.2)
// Applied unless bounds.evolve_temporal promotes them to free genes.
void apply_phase00_temporal_coupling(Genome& g);

// Force `encoder_widths` into a legal shape: every width clamped to
// [min_width, max_width], sorted strictly decreasing (each layer compresses toward the
// bottleneck — the defining property of an autoencoder), duplicates broken by
// decrementing, and the list truncated if strict decrease runs out of integer room.
// The result always has >= 1 layer. This is the ONLY structural invariant imposed on
// an otherwise free architecture; it is not a pre-defined configuration.
void repair_widths(Genome& g, const GenomeBounds& bounds);

// Draw a random genome: a random depth in [min_layers, max_layers] and random
// strictly-decreasing widths. `is_snn` selects whether encoding/temporal genes are
// active (SNN) or clamped to the ANN defaults ("direct", 1, 1.0).
Genome random_genome(std::mt19937& rng, const GenomeBounds& bounds, bool is_snn);

// Recombine two variable-length parents: the child's depth is inherited from one
// parent (coin flip), each width position sampled from whichever parent has it, then
// repaired to a legal strictly-decreasing shape.
Genome crossover(
    const Genome& a, const Genome& b, std::mt19937& rng, const GenomeBounds& bounds, bool is_snn);

// Structural mutation with the given per-operator probability: jitter a layer's width,
// insert a layer, or delete a layer (never below min_layers). Repaired afterward and
// temporal coupling re-applied unless evolve_temporal.
void mutate(Genome& g, std::mt19937& rng, const GenomeBounds& bounds, double prob, bool is_snn);

// Build the thesis AutoencoderConfig this genome represents.
//
// Starts from `base` — the PROFILE's autoencoder config — and overrides ONLY the fields
// the genome owns (layer specs, and for snn-ae the encoding/temporal genes). Everything
// else the profile set (`model`, `ae_loss_type`, firing-rate band, ...) is carried
// through untouched.
//
// This direction matters: an earlier version built a fresh config from scratch and so
// silently dropped `ae_loss_type`, making every run train under the default MSE no
// matter what the profile asked for — an mse and an mae run produced bit-identical
// results. Overriding a copy of the profile makes that class of silent drop impossible
// for any field added later.
thesis::ThesisConfig::AutoencoderConfig to_ae_config(
    const Genome& g, const thesis::ThesisConfig::AutoencoderConfig& base);

// ── Structural proxies (no training, no allocation) ──────────────────────────

// Trainable parameter count of the symmetric Protocol AE this genome builds
// (encoder 256 -> w0 -> ... -> latent, decoder the mirror, biases included).
long estimated_params(const Genome& g);

// Encoder multiply-accumulates for ONE forward frame: 256*w0 + sum_i w[i-1]*w[i].
long encoder_macs_per_frame(const Genome& g);

// Total encoder inference cost proxy = encoder_macs_per_frame · time_steps. This is
// the deterministic secondary objective (minimize) and the pre-training screen input
// (.wiki/Experiments/ParaconsistentGA-Design.md §4). Monotonic in width, depth and temporal length.
long inference_cost_proxy(const Genome& g);

} // namespace pga
