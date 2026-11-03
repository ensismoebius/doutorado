#pragma once
// Meeting01Energy.hpp -- the inference energy the paper reports (results_format 7).
//
// WHY NOT JUST COUNT OPERATIONS. The usual SNN estimate prices arithmetic only:
// 4.6 pJ x MACs + 0.9 pJ x ACs (arith_energy_nj, Meeting01Metrics.hpp), and the SNN wins
// because a spike turns a multiply-accumulate into an add. But fetching an operand costs
// more than using it -- in the same table a 32-bit float add is 0.9 pJ and one read of a
// 32 KB SRAM 20 pJ -- and an SNN reads and writes every neuron's membrane potential at every
// time step, traffic an ANN does not have. Estimates that count memory (Lemaire et al. 2022;
// Dampfhoffer et al., IEEE TETCI 7(3) 2023) find the advantage shrinks or reverses, and an
// operation count alone is what reviewers reject. So the number this file computes, the
// paper's primary one, is memory-aware.
//
// THE MODEL is Lemaire et al., "An Analytical Estimation of Spiking Neural Networks Energy
// Efficiency", ICONIP 2022, LNCS 13623, pp. 574-587 (doi 10.1007/978-3-031-30105-6_48), its
// fully-connected-layer equations applied unchanged to every family:
//
//     E = E_ops + E_addressing + E_memory                                     (their Eq. 18)
//
//   one layer n_in -> n_out, R rows       analog input (Eqs. 2,4,6,12,17)  spike input, theta
//   spikes (2,5,8,10,13,15,17)
//   ------------------------------------------------------------------------------------------------
//   multiply-accumulates (fp32)           R * n_in * n_out                 -
//   additions (fp32)                      R * n_out          (biases)      theta * n_out + R *
//   n_out address additions (int32)             R * (n_in + n_out)               theta * n_out
//   reads of inputs                       R * n_in                         theta
//   reads of weights and biases           R * (n_in + 1) * n_out           theta * n_out + R *
//   n_out reads, and writes, of potentials      -                                theta * n_out + R
//   * n_out writes of outputs                     R * n_out                        the LIF's spikes
//   (below)
//
//   R = the rows the layer processes: the T steps of one window (SNN-AE), 1 (ANN-AE), the S
//   positions (LSTM/GRU/Transformer-AE). LIF neuron, per (neuron, step): the leak, one
//   multiply-accumulate (Lemaire); per output spike: the reset, one add (Lemaire), and one
//   write of the spike event (Eq. 13).
//
// THE PRICES are Horowitz, "Computing's energy problem (and what we can do about it)",
// ISSCC 2014, Fig. 1.1.9 ("Rough energy costs for various operations in 45nm 0.9V"), read off
// the figure: fp32 add 0.9 pJ, fp32 multiply 3.7 pJ, int32 add 0.1 pJ, int32 multiply 3.1
// pJ; one 64-bit cache (SRAM) access 10 pJ at 8 KB, 20 pJ at 32 KB, 100 pJ at 1 MB. As in
// Lemaire et al., every layer has its own local SRAMs -- weights, potentials, input and
// output buffers -- and each is priced at its own size on the straight lines through those
// three points; one access moves one 32-bit word (Lemaire: all data, spike messages included,
// have one width) at the 64-bit figure.
//
// WHAT THE MODEL LEAVES TO US, each choice taken against the claim it could flatter:
//   - a memory below 8 KB costs 10 pJ, Horowitz's smallest point (no line toward zero): it
//     overprices small memories, the SNN's potentials above all;
//   - nonlinearities, which Lemaire do not price: ReLU = 1 comparison (an add); leaky ReLU,
//     tanh and sigmoid = 1 add + 1 multiply (a piecewise-linear unit). An exact exp costs more,
//     so this favours the sequence baselines whose removal the estimate supports;
//   - the LIF's threshold comparison, 1 add per (neuron, step) (Hoefflin & Wassner, 2025,
//     count it as E_CMP; Lemaire do not);
//   - the spike encoder, for every family that reads an encoded window (encoding_energy):
//     the SNN-AE pays for turning the analog window into spikes, the ANN-AE does not;
//   - the window is not re-read from DRAM: every family starts from the same buffer, so the
//     omission shifts all of them equally.
// SENSITIVITY: the same counts with every access priced at ONE memory size (8 KB, 32 KB,
// 1 MB) -- EnergyLedger::total_nj_at; the paper says whether a ranking survives all three.

#include <array>
#include <cstddef>
#include <string>

#include "Meeting01RunMetrics.hpp"
#include "models/autoencoder/ProtocolAutoencoder.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "models/gru/GRUAutoencoderConfig.hpp"
#include "models/lstm/LSTMAutoencoderConfig.hpp"
#include "models/transformer/TransformerAutoencoderConfig.hpp"
#include "tensor/Tensor.hpp"

namespace meeting01
{

using Tensor = nn::Tensor;

namespace energy
{
// Horowitz 2014, Fig. 1.1.9, pJ.
inline constexpr double kFpAddPj = 0.9;
inline constexpr double kFpMulPj = 3.7;
inline constexpr double kIntAddPj = 0.1;
inline constexpr double kIntMulPj = 3.1;
inline constexpr double kWordBytes = 4.0; // fp32 / int32 / one spike message

struct SramPoint
{
    double bytes;
    double pj; // one 64-bit access
};
inline constexpr std::array<SramPoint, 3> kSramPoints{
    {{8.0 * 1024.0, 10.0}, {32.0 * 1024.0, 20.0}, {1024.0 * 1024.0, 100.0}}};
// The 32 KB -> 1 MB line is extended to here and no further: past it a memory is not a
// local SRAM any more, and Horowitz's next figure is DRAM (1.3-2.6 nJ per access).
inline constexpr double kMaxSramBytes = 2.0 * 1024.0 * 1024.0;
} // namespace energy

/// pJ of one access to a local SRAM of `bytes` bytes: 10 pJ up to 8 KB, then the straight lines
/// through Horowitz's (8 KB, 10 pJ), (32 KB, 20 pJ), (1 MB, 100 pJ), the last one extended to
/// 2 MB. Throws std::invalid_argument for a size that is not a positive finite number and
/// std::out_of_range above 2 MB (a layer that big needs a DRAM term this model does not have).
auto sram_access_pj(double bytes) -> double;

/// What one inference costs, counted by kind. Each access is priced when it is added, at the
/// size of the memory it hits, into mem_pj; the counts are kept so total_nj_at() can reprice
/// them all at one size.
struct EnergyLedger
{
    double fp_macs = 0.0;  // fp32 multiply-accumulates
    double fp_adds = 0.0;  // fp32 additions and comparisons
    double fp_muls = 0.0;  // fp32 multiplications outside a MAC
    double int_adds = 0.0; // int32 additions (address arithmetic, encoder counters)
    double int_muls = 0.0; // int32 multiplications (the poisson encoder's random numbers)

    double rd_in = 0.0;    // reads of layer inputs (activations, spike events)
    double rd_param = 0.0; // reads of weights, biases and other fixed operands
    double rd_pot = 0.0;   // reads of state: membrane potentials, LSTM/GRU cells
    double wr_out = 0.0;   // writes of layer outputs (activations, spike events)
    double wr_pot = 0.0;   // writes of state

    double mem_pj = 0.0; // every access above at its own memory's price

    /// Adds `n` accesses to `counter` (one of the fields above) on a memory of `words` words.
    void access(double& counter, double n, double words);

    [[nodiscard]] auto accesses() const -> double;
    [[nodiscard]] auto ops_pj() const -> double;
    [[nodiscard]] auto total_nj() const -> double;
    /// The same counts with every access priced as one to an SRAM of `sram_bytes`.
    [[nodiscard]] auto total_nj_at(double sram_bytes) const -> double;

    auto operator+=(const EnergyLedger& other) -> EnergyLedger&;
    [[nodiscard]] auto scaled(double factor) const -> EnergyLedger;
};

/// A model's per-window cost for the frame-consuming families (the analytic MAC count the GA
/// already uses, and the ledger).
struct ModelCost
{
    std::size_t macs = 0;
    EnergyLedger energy;
};

/// Fills RunMetrics' energy fields from one window's ledger (a mean over windows is a ledger
/// too: EnergyLedger::scaled).
void set_energy(RunMetrics& m, const EnergyLedger& per_window);

// --- building blocks (Lemaire et al. 2022, Sec. III, fully-connected layers) ---------------

/// A product of `rows` input vectors of n_in values with an n_in x n_out operand held in a
/// memory of `operand_words` words, no bias (attention's Q K^T and A V; the core of a layer).
void add_matmul(EnergyLedger& e,
    double rows,
    double n_in,
    double n_out,
    double operand_words,
    bool writes_output);
/// A Linear layer on analog input (Lemaire's FNN FC): add_matmul over the weights, plus a bias
/// per output. `writes_output` false when the sum goes straight into a LIF's potential.
void add_dense_fc(EnergyLedger& e, double rows, double n_in, double n_out, bool writes_output);
/// A Linear layer on spikes (Lemaire's SNN FC): `spikes_in` events over `rows` steps, each
/// accumulated into the n_out potentials it reaches, plus a bias per neuron and step.
/// `writes_output` true when no LIF follows (the accumulated values then leave as outputs).
void add_spiking_fc(
    EnergyLedger& e, double spikes_in, double rows, double n_in, double n_out, bool writes_output);
/// A LifBPTT of `n` neurons over `rows` steps. `dense_input`: its current came from an analog
/// Linear (then the potential is read and written once per step here; after a spiking Linear
/// that traffic is already in add_spiking_fc). A readout neuron emits its potential instead of
/// spikes.
void add_lif(
    EnergyLedger& e, double rows, double n, double spikes_out, bool readout, bool dense_input);
enum class Nonlinearity
{
    Comparison,            // ReLU
    ComparisonAndMultiply, // leaky ReLU, tanh, sigmoid (piecewise-linear unit)
};
void add_nonlinearity(EnergyLedger& e, double values, Nonlinearity kind);

// --- whole models ---------------------------------------------------------------------------

/// The SNN-AE's last forward pass, walked layer by layer (call right after model.forward on
/// `input`, like count_snn_ops). Whether a Linear's input is spikes is decided by provenance,
/// as count_snn_ops does. Throws std::logic_error for a layer it has no model for, a LifBPTT
/// not fed by a Linear, the dual-branch model, or cached outputs that do not match the layers.
auto snn_ae_energy(const nn::models::autoencoder::ProtocolSpikingAutoencoder& model,
    const Tensor& input,
    bool input_is_spikes) -> EnergyLedger;
/// The ANN-AE: one forward pass of a zero window of `window_size` values, then the same walk
/// (its cost does not depend on the data). Throws like snn_ae_energy.
auto ann_ae_energy(nn::models::autoencoder::ProtocolAutoencoder& model, int window_size)
    -> EnergyLedger;
auto lstm_ae_energy(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> EnergyLedger;
auto gru_ae_energy(const nn::models::gru::GRUAutoencoderConfig& cfg) -> EnergyLedger;
auto transformer_ae_energy(const nn::models::transformer::TransformerAutoencoderConfig& cfg)
    -> EnergyLedger;

/// Turning a window of `values` analog samples into `time_steps` steps of the given encoding:
///   direct    nothing -- the first layer reads the analog window itself, every step;
///   poisson   per (value, step): read the value, one 32-bit LCG step for the random number
///             (an int multiply + an int add), one comparison;
///   latency   per value: read it, compute its spike step (a multiply + an add); per (value,
///             step): one int comparison of the step counter with it.
/// Writes: `event_driven` (the SNN-AE, which reads spike events) one per spike, `spikes`;
/// otherwise the whole encoded tensor, values x time_steps (a dense consumer reads all of it).
/// Throws on an unknown encoding.
auto encoding_energy(
    const std::string& encoding, double values, int time_steps, double spikes, bool event_driven)
    -> EnergyLedger;
/// The SNN-AE's input transform (apply_snn_architecture_transform) over `values` values per
/// step and `time_steps` steps: "dense" nothing; "conv1d" a 3-tap filter (3 MACs, 3 input and
/// 3 kernel reads, 1 write per value and step); "recurrent" a LIF per value (add_lif with an
/// analog input current read per value and step), emitting `spikes_out` spikes. Throws on an
/// unknown token.
auto architecture_transform_energy(
    const std::string& architecture, double values, int time_steps, double spikes_out)
    -> EnergyLedger;

} // namespace meeting01
