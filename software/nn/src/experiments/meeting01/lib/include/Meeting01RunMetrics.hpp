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
    // Memory-aware energy per window, nJ, the paper's primary energy number (results_format 7):
    // operations + address arithmetic + every SRAM access, Lemaire et al. 2022's model at
    // Horowitz 2014's prices (Meeting01Energy.hpp). energy_nj = energy_ops_nj + energy_mem_nj;
    // mem_accesses is their access count, and energy_nj_sram{8k,32k,1m} reprice all of them at
    // one memory size (the sensitivity analysis). NaN until set_energy() fills them, so a row
    // that never got an estimate is loud downstream instead of reading as free.
    double energy_nj = std::numeric_limits<double>::quiet_NaN();
    double energy_ops_nj = std::numeric_limits<double>::quiet_NaN();
    double energy_mem_nj = std::numeric_limits<double>::quiet_NaN();
    double mem_accesses = std::numeric_limits<double>::quiet_NaN();
    double energy_nj_sram8k = std::numeric_limits<double>::quiet_NaN();
    double energy_nj_sram32k = std::numeric_limits<double>::quiet_NaN();
    double energy_nj_sram1m = std::numeric_limits<double>::quiet_NaN();
    // results_format 8: energy_nj (and macs, acs, arith_energy_nj) price an input repeated at
    // every step -- the SNN-AE's direct code through the dense or conv1d transform -- once
    // per window (Meeting01Energy.hpp). This is the memory-aware total with every step's rows
    // priced, as the code runs it: the check. Equal to energy_nj for every other model.
    double energy_nj_as_executed = std::numeric_limits<double>::quiet_NaN();
    // results_format 11: the part of energy_nj a sensor pays when only the code leaves it: for
    // the SNN-AE the input encoding, the input transform and the encoder network; for the
    // ANN-AE its encoder network. NaN for the LSTM/GRU/Transformer-AE, whose ledgers are not
    // split; the paper's energy claims do not use them.
    double energy_nj_encoder = std::numeric_limits<double>::quiet_NaN();

    // results_format 11: what the network inside looks like, so a gap can be split into "it did
    // not fit" and "it fits but does not generalize", and a bounded unit can be seen pinned.
    // NaN means "not measured or not applicable": a non-spiking model has no rate or threshold,
    // and train_mse is filled only on a final fit's test row (not on GA evaluations).
    //
    // hidden_silent_share / hidden_always_share: shares of the (hidden LIF neuron, window) pairs
    // whose firing rate over T steps is <= 0.01 / >= 0.99, kept apart because the two ends mean
    // different things (no information vs. a constant). Over the evaluated windows.
    float hidden_silent_share = std::numeric_limits<float>::quiet_NaN();
    float hidden_always_share = std::numeric_limits<float>::quiet_NaN();
    // Trained LIF parameters over the spiking layers: V_th has no bound in the code, so its
    // minimum is what shows a threshold reaching 0.
    float lif_v_th_mean = std::numeric_limits<float>::quiet_NaN();
    float lif_v_th_min = std::numeric_limits<float>::quiet_NaN();
    float lif_beta_mean = std::numeric_limits<float>::quiet_NaN();
    // Mean per-window MSE of the final network on the windows it was fitted on.
    float train_mse = std::numeric_limits<float>::quiet_NaN();

    float train_ms = 0.0f;
    float infer_ms = 0.0f;

    std::size_t parameter_count = 0;
    // Dense multiply-accumulates per window. Analytic for the non-spiking families; for the
    // SNN, counted from the forward passes of the evaluated windows (count_snn_ops).
    std::size_t macs = 0;
};

} // namespace meeting01
