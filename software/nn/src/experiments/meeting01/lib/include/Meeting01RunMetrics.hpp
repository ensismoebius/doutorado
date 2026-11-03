#pragma once

#include <cstddef>
#include <limits>

namespace meeting01
{

struct RunMetrics
{
    float mse = 0.0f;
    float mae = 0.0f;
    float r2 = 0.0f;

    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;

    // Fraction of (hidden LIF neuron, simulation step) pairs that emitted a spike over the
    // evaluated windows (count_snn_ops, Meeting01Metrics.hpp). NaN for a model with no
    // spiking neuron: "not applicable", which 0 ("never fires") would misstate. Until
    // results_format 5 this was the fraction of OUTPUT values above zero, which is ~0.5 for
    // any model and says nothing about spikes.
    float spike_rate = std::numeric_limits<float>::quiet_NaN();
    // Spike-driven synaptic accumulates per window: a Linear layer whose input is binary
    // does one add per (nonzero input, output) pair instead of a multiply-accumulate.
    // Mean over the evaluated windows (it depends on how many spikes the window drew);
    // 0 for a model with no spiking layer, which is a true count, not a placeholder.
    double acs = 0.0;
    // Arithmetic-only energy estimate per window, nJ: kMacEnergyPj * macs + kAcEnergyPj *
    // acs (Horowitz 2014, 45 nm fp32 -- Meeting01Metrics.hpp). Not a device measurement:
    // memory traffic, which dominates real energy, is not in it. Replaces `energy`
    // (results_format <= 4), which added a spike count to 10 x MACs (two unitless terms).
    float arith_energy_nj = 0.0f;

    float train_ms = 0.0f;
    float infer_ms = 0.0f;

    std::size_t parameter_count = 0;
    // Dense multiply-accumulates per window. Analytic for the non-spiking families; for the
    // SNN, counted from the forward passes of the evaluated windows (count_snn_ops).
    std::size_t macs = 0;
};

} // namespace meeting01
