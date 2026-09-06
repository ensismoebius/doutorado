/**
 * @file src/experiments/paraconsistentGA/lib/include/GenomeBounds.hpp
 * @brief GenomeBounds struct (extracted from GaGenome.hpp).
 */

#pragma once

#include <string>
#include <vector>

namespace pga
{

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

} // namespace pga
