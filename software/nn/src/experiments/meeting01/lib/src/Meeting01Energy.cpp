#include "../include/Meeting01Energy.hpp"

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

#include "layers/Layers.hpp"

namespace meeting01
{

auto sram_access_pj(double bytes) -> double
{
    using energy::kSramPoints;
    if (!(bytes > 0.0) || !std::isfinite(bytes))
        throw std::invalid_argument("sram_access_pj: a memory of " + std::to_string(bytes) +
                                    " bytes; a size must be a positive finite number");
    if (bytes > energy::kMaxSramBytes)
        throw std::out_of_range(
            "sram_access_pj: a memory of " + std::to_string(bytes) +
            " bytes is above the 2 MB this model prices as a local SRAM (Horowitz's largest SRAM "
            "point is 1 MB, his next figure DRAM at 1.3-2.6 nJ per access). Shrink the layer, or "
            "add a DRAM term to the model before pricing it.");
    if (bytes <= kSramPoints[0].bytes) return kSramPoints[0].pj;
    // The segment containing `bytes`; the last one is extended past 1 MB up to kMaxSramBytes.
    std::size_t s = 1;
    while (s + 1 < kSramPoints.size() && bytes > kSramPoints[s].bytes) ++s;
    const auto& a = kSramPoints[s - 1];
    const auto& b = kSramPoints[s];
    return a.pj + (b.pj - a.pj) * (bytes - a.bytes) / (b.bytes - a.bytes);
}

void EnergyLedger::access(double& counter, double n, double words)
{
    if (n == 0.0) return;
    counter += n;
    mem_pj += n * sram_access_pj(words * energy::kWordBytes);
}

auto EnergyLedger::accesses() const -> double
{
    return rd_in + rd_param + rd_pot + wr_out + wr_pot;
}

auto EnergyLedger::ops_pj() const -> double
{
    using namespace energy;
    return fp_macs * (kFpMulPj + kFpAddPj) + fp_adds * kFpAddPj + fp_muls * kFpMulPj +
           int_adds * kIntAddPj + int_muls * kIntMulPj;
}

auto EnergyLedger::total_nj() const -> double
{
    return (ops_pj() + mem_pj) / 1000.0;
}

auto EnergyLedger::total_nj_at(double sram_bytes) const -> double
{
    return (ops_pj() + accesses() * sram_access_pj(sram_bytes)) / 1000.0;
}

auto EnergyLedger::operator+=(const EnergyLedger& o) -> EnergyLedger&
{
    fp_macs += o.fp_macs;
    fp_adds += o.fp_adds;
    fp_muls += o.fp_muls;
    int_adds += o.int_adds;
    int_muls += o.int_muls;
    rd_in += o.rd_in;
    rd_param += o.rd_param;
    rd_pot += o.rd_pot;
    wr_out += o.wr_out;
    wr_pot += o.wr_pot;
    mem_pj += o.mem_pj;
    return *this;
}

auto EnergyLedger::scaled(double f) const -> EnergyLedger
{
    EnergyLedger r = *this;
    for (double* v : {&r.fp_macs,
             &r.fp_adds,
             &r.fp_muls,
             &r.int_adds,
             &r.int_muls,
             &r.rd_in,
             &r.rd_param,
             &r.rd_pot,
             &r.wr_out,
             &r.wr_pot,
             &r.mem_pj})
        *v *= f;
    return r;
}

void set_energy(RunMetrics& m, const EnergyLedger& w)
{
    m.energy_nj = w.total_nj();
    m.energy_ops_nj = w.ops_pj() / 1000.0;
    m.energy_mem_nj = w.mem_pj / 1000.0;
    m.mem_accesses = w.accesses();
    m.energy_nj_sram8k = w.total_nj_at(energy::kSramPoints[0].bytes);
    m.energy_nj_sram32k = w.total_nj_at(energy::kSramPoints[1].bytes);
    m.energy_nj_sram1m = w.total_nj_at(energy::kSramPoints[2].bytes);
    m.energy_nj_as_executed = w.total_nj();
}

void set_energy_as_executed(RunMetrics& m, const EnergyLedger& w)
{
    m.energy_nj_as_executed = w.total_nj();
}

// --- building blocks -----------------------------------------------------------------------

void add_matmul(EnergyLedger& e,
    double rows,
    double n_in,
    double n_out,
    double operand_words,
    bool writes_output)
{
    e.fp_macs += rows * n_in * n_out;
    e.int_adds += rows * (n_in + n_out);                        // Lemaire Eq. 17
    e.access(e.rd_in, rows * n_in, n_in);                       // Eq. 4
    e.access(e.rd_param, rows * n_in * n_out, operand_words);   // Eq. 6, weights
    if (writes_output) e.access(e.wr_out, rows * n_out, n_out); // Eq. 12
}

void add_dense_fc(EnergyLedger& e, double rows, double n_in, double n_out, bool writes_output)
{
    const double weight_words = (n_in + 1.0) * n_out;
    add_matmul(e, rows, n_in, n_out, weight_words, writes_output);
    e.fp_adds += rows * n_out;                        // Eq. 2, the bias
    e.access(e.rd_param, rows * n_out, weight_words); // Eq. 6, the bias reads
}

void add_spiking_fc(
    EnergyLedger& e, double spikes_in, double rows, double n_in, double n_out, bool writes_output)
{
    const double events = spikes_in * n_out; // one synaptic update per (spike, target neuron)
    const double weight_words = (n_in + 1.0) * n_out;
    e.fp_adds += events + rows * n_out;                        // Eq. 2: synapses + bias/step
    e.int_adds += events;                                      // Eq. 17
    e.access(e.rd_in, spikes_in, n_in);                        // Eq. 5, the spike FIFO
    e.access(e.rd_param, events + rows * n_out, weight_words); // Eq. 8
    e.access(e.rd_pot, events + rows * n_out, n_out);          // Eq. 10
    e.access(e.wr_pot, events + rows * n_out, n_out);          // Eq. 15
    if (writes_output) e.access(e.wr_out, rows * n_out, n_out);
}

void add_lif(
    EnergyLedger& e, double rows, double n, double spikes_out, bool readout, bool dense_input)
{
    e.fp_macs += rows * n; // the leak, v = beta * v + I (Lemaire: one MAC per neuron and step)
    if (dense_input)
    {
        e.access(e.rd_pot, rows * n, n);
        e.access(e.wr_pot, rows * n, n);
    }
    if (readout)
    {
        e.access(e.wr_out, rows * n, n); // the potential itself is the output
        return;
    }
    e.fp_adds += rows * n + spikes_out; // threshold comparison per step + reset per spike
    e.access(e.wr_out, spikes_out, n);  // Eq. 13, one event per spike
}

void add_nonlinearity(EnergyLedger& e, double values, Nonlinearity kind)
{
    e.fp_adds += values;
    if (kind == Nonlinearity::ComparisonAndMultiply) e.fp_muls += values;
}

namespace
{

// Walks one Sequential's last forward pass into `e`; returns whether its output is spikes.
// repeated_steps > 1: the input's rows repeat that many times (snn_ae_energy), so its first
// layer -- a Linear into a LifBPTT, on analog input -- is priced once per sample.
auto walk_sequential(const nn::Sequential& seq,
    const Tensor& input,
    bool input_is_spikes,
    EnergyLedger& e,
    const char* who,
    int repeated_steps) -> bool
{
    if (seq.outputs.size() != seq.layers.size())
        throw std::logic_error(std::string(who) + ": a Sequential cached " +
                               std::to_string(seq.outputs.size()) + " outputs for " +
                               std::to_string(seq.layers.size()) +
                               " layers. Call it right after model.forward() on the window "
                               "whose energy it should count.");
    if (repeated_steps < 1)
        throw std::invalid_argument(std::string(who) + ": repeated_steps must be >= 1 (got " +
                                    std::to_string(repeated_steps) +
                                    "); 1 prices every row, T a repeated input once");
    if (repeated_steps > 1)
    {
        const bool first_is_linear_into_lif =
            seq.layers.size() >= 2 && dynamic_cast<const nn::Linear*>(seq.layers[0].get()) &&
            dynamic_cast<const nn::LifBPTT*>(seq.layers[1].get());
        if (input_is_spikes || !first_is_linear_into_lif ||
            input.rows() % static_cast<nn::Index>(repeated_steps) != 0)
            throw std::logic_error(
                std::string(who) + ": a repeated input is priced once only when it is analog, " +
                "its " + std::to_string(input.rows()) + " rows split into " +
                std::to_string(repeated_steps) +
                " equal steps, and it feeds a Linear followed by a LifBPTT; this one does not. "
                "Pass repeated_steps = 1, or extend the model for this topology.");
    }

    const Tensor* in = &input;
    bool in_is_spikes = input_is_spikes;
    bool after_linear = false;       // the previous layer was a Linear
    bool after_dense_linear = false; // ... on analog input
    for (std::size_t i = 0; i < seq.layers.size(); ++i)
    {
        const auto* layer = seq.layers[i].get();
        const Tensor& out = seq.outputs[i];
        const bool next_is_lif =
            i + 1 < seq.layers.size() &&
            dynamic_cast<const nn::LifBPTT*>(seq.layers[i + 1].get()) != nullptr;
        bool out_is_spikes = false;
        bool is_linear = false;

        if (const auto* lin = dynamic_cast<const nn::Linear*>(layer); lin != nullptr)
        {
            const auto rows = static_cast<double>(in->rows());
            const auto n_in = static_cast<double>(lin->in_features);
            const auto n_out = static_cast<double>(lin->out_features);
            if (in_is_spikes)
                add_spiking_fc(e, static_cast<double>(in->sum()), rows, n_in, n_out, !next_is_lif);
            else if (i == 0 && repeated_steps > 1)
            {
                // The same current at every step: computed and written once per sample, read
                // back by the LIF at each of the `rows` steps (checked above).
                add_dense_fc(e, rows / repeated_steps, n_in, n_out, /*writes_output=*/true);
                e.access(e.rd_in, rows * n_out, n_out);
            }
            else
                add_dense_fc(e, rows, n_in, n_out, !next_is_lif);
            is_linear = true;
            after_dense_linear = !in_is_spikes;
        }
        else if (const auto* lif = dynamic_cast<const nn::LifBPTT*>(layer); lif != nullptr)
        {
            if (!after_linear)
                throw std::logic_error(std::string(who) + ": layer " + std::to_string(i) +
                                       " is a LifBPTT not fed by a Linear; its input current is "
                                       "priced as that Linear's output. Extend the model for "
                                       "this topology before counting it.");
            const double spikes = lif->readout_mode ? 0.0 : static_cast<double>(out.sum());
            add_lif(e,
                static_cast<double>(out.rows()),
                static_cast<double>(out.cols()),
                spikes,
                lif->readout_mode,
                after_dense_linear);
            out_is_spikes = !lif->readout_mode;
        }
        else if (dynamic_cast<const nn::TimeMeanPool*>(layer) != nullptr)
        {
            // An event-driven counter adds one per spike; an analog input adds every value.
            const double summed =
                in_is_spikes ? static_cast<double>(in->sum()) : static_cast<double>(in->size());
            const auto features = static_cast<double>(out.cols());
            e.fp_adds += summed;
            e.fp_macs += static_cast<double>(out.size()); // the 1/T scaling
            e.access(e.rd_in, summed, features);
            e.access(e.wr_out, static_cast<double>(out.size()), features);
        }
        else if (dynamic_cast<const nn::ReLU*>(layer) != nullptr)
        {
            add_nonlinearity(e, static_cast<double>(out.size()), Nonlinearity::Comparison);
        }
        else if (dynamic_cast<const nn::LeakyReLU*>(layer) != nullptr ||
                 dynamic_cast<const nn::Tanh*>(layer) != nullptr)
        {
            add_nonlinearity(
                e, static_cast<double>(out.size()), Nonlinearity::ComparisonAndMultiply);
        }
        else
        {
            throw std::logic_error(std::string(who) + ": layer " + std::to_string(i) + " (" +
                                   typeid(*layer).name() +
                                   ") has no energy model. Add one to Meeting01Energy.cpp "
                                   "(and to the paper's energy section) before counting it.");
        }
        after_linear = is_linear;
        in = &out;
        in_is_spikes = out_is_spikes;
    }
    return in_is_spikes;
}

// Recurrent autoencoder of LSTM (cell = true, 4 gates) or GRU (3 gates) layers -- the
// topology of recurrent_ae_macs (Meeting01Metrics.cpp), per window of S positions:
//   encoder   L layers, layer 0 reads [frame (D); h (H)], the others [h below (H); h (H)]
//   latent    enc_proj H -> Z (tanh) on the last state, dec_expand Z -> H, once each
//   decoder   L layers reading [expanded latent or h below (H); h (H)]
//   head      out_proj H -> D at every position
// Per (position, unit) after the gate product:
//   LSTM  c = f*c + i*g, h = o*tanh(c): 5 nonlinearities (3 sigmoid, 2 tanh), 3 multiplies,
//         1 add; reads the 4 gate values and c, writes c and h
//   GRU   n = tanh(Wn x + r*(Un h)), h = (1-z)*n + z*h: 3 nonlinearities, 3 multiplies,
//         3 adds; reads the 3 gate values and h, writes h (state) and h (output)
auto recurrent_ae_energy(const char* who,
    bool lstm,
    int seq_len,
    int input_size,
    int hidden_size,
    int latent_size,
    int num_layers) -> EnergyLedger
{
    for (const auto& [field, v] : {std::pair<const char*, int>{"seq_len", seq_len},
             {"input_size", input_size},
             {"hidden_size", hidden_size},
             {"latent_size", latent_size},
             {"num_layers", num_layers}})
        if (v < 1)
            throw std::invalid_argument(std::string(who) + ": " + field + " must be >= 1 (got " +
                                        std::to_string(v) + "); not a buildable network");

    const auto S = static_cast<double>(seq_len);
    const auto D = static_cast<double>(input_size);
    const auto H = static_cast<double>(hidden_size);
    const auto Z = static_cast<double>(latent_size);
    const double gates = lstm ? 4.0 : 3.0;

    EnergyLedger e;
    const auto layer = [&](double in)
    {
        add_dense_fc(e, S, in + H, gates * H, /*writes_output=*/true);
        add_nonlinearity(e, S * (lstm ? 5.0 : 3.0) * H, Nonlinearity::ComparisonAndMultiply);
        e.fp_muls += S * 3.0 * H;
        e.fp_adds += S * (lstm ? 1.0 : 3.0) * H;
        e.access(e.rd_in, S * gates * H, gates * H);
        e.access(e.rd_pot, S * H, H);
        e.access(e.wr_pot, S * H, H);
        e.access(e.wr_out, S * H, H);
    };
    for (int l = 0; l < num_layers; ++l) layer(l == 0 ? D : H);  // encoder
    add_dense_fc(e, 1.0, H, Z, true);                            // enc_proj
    add_nonlinearity(e, Z, Nonlinearity::ComparisonAndMultiply); // its tanh
    add_dense_fc(e, 1.0, Z, H, true);                            // dec_expand
    for (int l = 0; l < num_layers; ++l) layer(H);               // decoder
    add_dense_fc(e, S, H, D, true);                              // out_proj
    return e;
}

} // namespace

auto snn_ae_energy(const nn::models::autoencoder::ProtocolSpikingAutoencoder& model,
    const Tensor& input,
    bool input_is_spikes,
    int repeated_steps) -> EnergyLedger
{
    if (model.use_dual_branch_)
        throw std::logic_error(
            "snn_ae_energy: the dual-branch ProtocolSpikingAutoencoder is not "
            "counted (meeting01 never builds it).");
    EnergyLedger e;
    const bool latent_is_spikes =
        walk_sequential(model.encoder_, input, input_is_spikes, e, "snn_ae_energy", repeated_steps);
    if (model.encoder_.outputs.empty())
        throw std::logic_error("snn_ae_energy: the encoder cached no outputs; run forward first.");
    // The decoder reads the latent, which changes from step to step (or is pooled once).
    walk_sequential(
        model.decoder_, model.encoder_.outputs.back(), latent_is_spikes, e, "snn_ae_energy", 1);
    return e;
}

auto ann_ae_energy(nn::models::autoencoder::ProtocolAutoencoder& model, int window_size)
    -> EnergyLedger
{
    if (model.use_dual_branch_)
        throw std::logic_error(
            "ann_ae_energy: the dual-branch ProtocolAutoencoder is not "
            "counted (meeting01 never builds it).");
    if (window_size < 1)
        throw std::invalid_argument(
            "ann_ae_energy: window_size must be >= 1 (got " + std::to_string(window_size) + ")");
    const Tensor zeros = Tensor::zeros(1, window_size);
    (void) model.forward(zeros, false);
    EnergyLedger e;
    walk_sequential(model.encoder_, zeros, false, e, "ann_ae_energy", 1);
    if (model.encoder_.outputs.empty())
        throw std::logic_error("ann_ae_energy: the encoder cached no outputs after forward.");
    walk_sequential(model.decoder_, model.encoder_.outputs.back(), false, e, "ann_ae_energy", 1);
    return e;
}

auto lstm_ae_energy(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> EnergyLedger
{
    return recurrent_ae_energy("lstm_ae_energy",
        true,
        cfg.seq_len,
        cfg.input_size,
        cfg.hidden_size,
        cfg.latent_size,
        cfg.num_layers);
}

auto gru_ae_energy(const nn::models::gru::GRUAutoencoderConfig& cfg) -> EnergyLedger
{
    return recurrent_ae_energy("gru_ae_energy",
        false,
        cfg.seq_len,
        cfg.input_size,
        cfg.hidden_size,
        cfg.latent_size,
        cfg.num_layers);
}

// The TransformerAutoencoder (models/transformer/TransformerAutoencoder.hpp), per window of S
// positions, M = d_model, F = d_ff, h heads of M/h, L blocks in the encoder and L in the
// decoder (self-attention only, post-norm: TransformerEncoderBlock.hpp):
//   embed D -> M and the positional table, at every position (encoder and decoder)
//   block     Q,K,V (M -> 3M); scores Q K^T (each query against the S keys of its head) and
//             their 1/sqrt(M/h) scaling; softmax (exp + sum + divide per score); A V; O (M -> M);
//             residual add; LayerNorm; FFN M -> F (ReLU) -> M; residual add; LayerNorm
//   tail      mean over the S positions, M -> Z (tanh); decoder head Z -> M, replicated; output
//             projection M -> D at every position
// LayerNorm per position: mean and variance (2 passes over M) and the affine normalization:
// 4M adds, 3M + 2 multiplies, one square root (priced as a nonlinearity); reads 2M inputs and
// the 2M gamma/beta, writes M.
auto transformer_ae_energy(const nn::models::transformer::TransformerAutoencoderConfig& cfg)
    -> EnergyLedger
{
    for (const auto& [field, v] : {std::pair<const char*, int>{"seq_len", cfg.seq_len},
             {"input_size", cfg.input_size},
             {"d_model", cfg.d_model},
             {"n_heads", cfg.n_heads},
             {"n_layers", cfg.n_layers},
             {"d_ff", cfg.d_ff},
             {"latent_size", cfg.latent_size}})
        if (v < 1)
            throw std::invalid_argument(std::string("transformer_ae_energy: ") + field +
                                        " must be >= 1 (got " + std::to_string(v) +
                                        "); not a buildable network");
    if (cfg.d_model % cfg.n_heads != 0)
        throw std::invalid_argument("transformer_ae_energy: n_heads (" +
                                    std::to_string(cfg.n_heads) + ") must divide d_model (" +
                                    std::to_string(cfg.d_model) + ")");

    const auto S = static_cast<double>(cfg.seq_len);
    const auto D = static_cast<double>(cfg.input_size);
    const auto M = static_cast<double>(cfg.d_model);
    const auto F = static_cast<double>(cfg.d_ff);
    const auto h = static_cast<double>(cfg.n_heads);
    const auto Z = static_cast<double>(cfg.latent_size);
    const double dk = M / h;

    EnergyLedger e;
    const auto positional = [&]()
    {
        e.fp_adds += S * M;
        e.access(e.rd_in, S * M, M);
        e.access(e.rd_param, S * M, S * M);
        e.access(e.wr_out, S * M, M);
    };
    const auto residual = [&]()
    {
        e.fp_adds += S * M;
        e.access(e.rd_in, S * M, M);
        e.access(e.wr_out, S * M, M);
    };
    const auto layer_norm = [&]()
    {
        e.fp_adds += S * 4.0 * M;
        e.fp_muls += S * (3.0 * M + 2.0);
        add_nonlinearity(e, S, Nonlinearity::ComparisonAndMultiply); // the square root
        e.access(e.rd_in, S * 2.0 * M, M);
        e.access(e.rd_param, S * 2.0 * M, 2.0 * M);
        e.access(e.wr_out, S * M, M);
    };
    const auto block = [&]()
    {
        add_dense_fc(e, S, M, 3.0 * M, true); // Q, K, V
        // Scores: S*h query vectors of dk against the S keys of their head (S*dk words).
        add_matmul(e, S * h, dk, S, S * dk, true);
        e.fp_muls += S * S * h; // 1/sqrt(dk)
        // Softmax over each query's S scores.
        add_nonlinearity(e, S * S * h, Nonlinearity::ComparisonAndMultiply); // exp
        e.fp_adds += S * S * h;                                              // the sum
        e.fp_muls += S * S * h;                                              // the division
        e.access(e.rd_in, S * S * h, S);
        e.access(e.wr_out, S * S * h, S);
        // A V: each query's S weights against the S values of its head.
        add_matmul(e, S * h, S, dk, S * dk, true);
        add_dense_fc(e, S, M, M, true); // O
        residual();
        layer_norm();
        add_dense_fc(e, S, M, F, true);
        add_nonlinearity(e, S * F, Nonlinearity::Comparison); // ReLU
        add_dense_fc(e, S, F, M, true);
        residual();
        layer_norm();
    };

    add_dense_fc(e, S, D, M, true); // embed
    positional();
    for (int l = 0; l < cfg.n_layers; ++l) block();
    // A pre-norm block has the same LayerNorms and residuals, in another order; only the
    // final LayerNorm that closes a pre-norm stack is extra (TransformerAutoencoder).
    if (cfg.norm_first) layer_norm();
    e.fp_adds += S * M; // mean over positions
    e.fp_macs += M;
    e.access(e.rd_in, S * M, M);
    e.access(e.wr_out, M, M);
    add_dense_fc(e, 1.0, M, Z, true);                            // to_latent
    add_nonlinearity(e, Z, Nonlinearity::ComparisonAndMultiply); // tanh
    add_dense_fc(e, 1.0, Z, M, true);                            // from_latent
    positional();                                                // replicate + positional
    for (int l = 0; l < cfg.n_layers; ++l) block();
    if (cfg.norm_first) layer_norm();
    add_dense_fc(e, S, M, D, true); // out_proj
    return e;
}

auto encoding_energy(
    const std::string& encoding, double values, int time_steps, double spikes, bool event_driven)
    -> EnergyLedger
{
    if (time_steps < 1)
        throw std::invalid_argument(
            "encoding_energy: time_steps must be >= 1 (got " + std::to_string(time_steps) + ")");
    const auto T = static_cast<double>(time_steps);
    EnergyLedger e;
    if (encoding == "direct") return e;
    const double writes = event_driven ? spikes : values * T;
    const double out_words = event_driven ? values : values * T;
    if (encoding == "poisson")
    {
        e.access(e.rd_in, values * T, values);
        e.int_muls += values * T; // x = a * x + c, one 32-bit LCG step per random number
        e.int_adds += values * T;
        e.fp_adds += values * T; // u < rate
        e.access(e.wr_out, writes, out_words);
        return e;
    }
    if (encoding == "latency")
    {
        e.access(e.rd_in, values, values);
        e.fp_muls += values;      // x * T
        e.fp_adds += values;      // T - floor(.)
        e.int_adds += values * T; // step counter == spike step
        e.access(e.wr_out, writes, out_words);
        return e;
    }
    throw std::invalid_argument("encoding_energy: unknown encoding '" + encoding +
                                "' (expected direct, poisson or latency)");
}

auto architecture_transform_energy(
    const std::string& architecture, double values, int time_steps, double spikes_out)
    -> EnergyLedger
{
    const auto T = static_cast<double>(time_steps);
    EnergyLedger e;
    if (architecture == "dense") return e;
    if (architecture == "conv1d")
    {
        e.fp_macs += 3.0 * values * T;
        e.access(e.rd_in, 3.0 * values * T, values);
        e.access(e.rd_param, 3.0 * values * T, 3.0);
        e.access(e.wr_out, values * T, values);
        return e;
    }
    if (architecture == "recurrent")
    {
        e.access(e.rd_in, values * T, values); // the encoded input, as the current
        add_lif(e, T, values, spikes_out, /*readout=*/false, /*dense_input=*/true);
        return e;
    }
    throw std::invalid_argument("architecture_transform_energy: unknown SNN architecture '" +
                                architecture + "' (expected dense, conv1d or recurrent)");
}

} // namespace meeting01
