/**
 * @file src/experiments/paraconsistentGA/lib/include/GenomeBounds.hpp
 * @brief GenomeBounds struct (extracted from GaGenome.hpp).
 */

#pragma once

#include <string>
#include <vector>

#include "Required.hpp"

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
    thesis::Required<int> min_layers{"ga.bounds.min_layers"};
    thesis::Required<int> max_layers{"ga.bounds.max_layers"};

    // Per-layer neuron count range. Every width (hidden and latent) is drawn from
    // [min_width, max_width]. min_width=1 permits a 1-neuron latent (as in the
    // "3->2->1" example); it applies to hidden layers too.
    thesis::Required<int> min_width{"ga.bounds.min_width"};
    thesis::Required<int> max_width{"ga.bounds.max_width"};

    // SNN population only (the ANN is always "direct"): the encodings a genome may carry.
    thesis::Required<std::vector<std::string>> encoding_choices{"ga.bounds.encoding_choices"};

    // SNN population only. false = time_steps and voltage_threshold are DERIVED from the
    // encoding (apply_phase00_temporal_coupling), not searched. true = they become free
    // genes drawn from the ranges below — a declared expansion of the phase00 space
    // (.wiki/Experiments/ParaconsistentGA-Design.md §5.1).
    thesis::Required<bool> evolve_temporal{"ga.bounds.evolve_temporal"};
    // SNN population with evolve_temporal = true only:
    thesis::Required<std::vector<int>> time_steps_choices{"ga.bounds.time_steps_choices"};
    thesis::Required<float> voltage_threshold_min{"ga.bounds.voltage_threshold_min"};
    thesis::Required<float> voltage_threshold_max{"ga.bounds.voltage_threshold_max"};
};

} // namespace pga
