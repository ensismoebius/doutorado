#include "../include/Meeting01Metrics.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace meeting01
{

namespace
{

// MACs of one forward pass over one window of a LSTMAutoencoder / GRUAutoencoder, which have
// the same topology (`gates` is 4 for the LSTM, 3 for the GRU). Dense multiply-accumulates
// only: gate nonlinearities and biases are not MACs, the convention
// estimate_transformer_macs() uses too.
//
//   encoder   layer 0 reads the D-wide frame, layers 1..L-1 the H-wide state below them
//   decoder   L layers, each H -> H, run over the same S positions
//   head      out_proj H -> D on every position
//   latent    enc_proj H -> Z on the final state, dec_expand Z -> H on the latent: once each
//
// A recurrent layer costs gates * H * (in + H) per position: the input and the previous
// state go into every gate. Until 2026-10-07 the estimate counted the encoder stack alone,
// at width D for every layer, and the head once: about 2.8x too low at H = 64, L = 1 (and
// 3.2x at the production profile), while the Transformer's estimate always counted both
// stacks. parameter_count_gtest ties this to the network the constructor really builds.
auto recurrent_ae_macs(const char* who,
    std::size_t gates,
    int seq_len,
    int input_size,
    int hidden_size,
    int latent_size,
    int num_layers) -> std::size_t
{
    const auto require_positive = [who](const char* field, int value)
    {
        if (value < 1)
        {
            throw std::invalid_argument(std::string(who) + ": " + field + " must be >= 1 (got " +
                                        std::to_string(value) +
                                        "): that is not a buildable network, so it has no "
                                        "cost. Check the profile's model.* fields (a latent "
                                        "width of -1 means model.latent_dim is unset and the "
                                        "layer specs gave none) or the genome that produced it");
        }
    };
    require_positive("seq_len", seq_len);
    require_positive("input_size", input_size);
    require_positive("hidden_size", hidden_size);
    require_positive("latent_size", latent_size);
    require_positive("num_layers", num_layers);

    const auto S = static_cast<std::size_t>(seq_len);
    const auto D = static_cast<std::size_t>(input_size);
    const auto H = static_cast<std::size_t>(hidden_size);
    const auto Z = static_cast<std::size_t>(latent_size);
    const auto L = static_cast<std::size_t>(num_layers);

    const auto layer = [gates, H](std::size_t in) { return gates * H * (in + H); };

    std::size_t per_position = layer(D) + (L - 1) * layer(H); // encoder
    per_position += L * layer(H);                             // decoder
    per_position += H * D;                                    // out_proj
    return S * per_position + 2 * H * Z;                      // + enc_proj, dec_expand
}

} // namespace

auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t
{
    return recurrent_ae_macs("estimate_lstm_macs",
        4,
        cfg.seq_len,
        cfg.input_size,
        cfg.hidden_size,
        cfg.latent_size,
        cfg.num_layers);
}

auto estimate_snn_macs(std::size_t input_features,
    const std::vector<int>& encoder_widths,
    int encoder_steps,
    int decoder_steps) -> std::size_t
{
    // Sums the REAL per-layer projections instead of assuming every hidden layer is as
    // wide as the first. The (first_width, depth) approximation gave {128, 8} and
    // {128, 120} an identical cost, so the GA's second objective could not tell a cheap
    // genome from an expensive one. Encoder is mirrored by the decoder.
    if (encoder_widths.empty()) return 0;

    std::size_t encoder = input_features * static_cast<std::size_t>(encoder_widths.front());
    for (std::size_t i = 0; i + 1 < encoder_widths.size(); ++i)
        encoder += static_cast<std::size_t>(encoder_widths[i]) *
                   static_cast<std::size_t>(encoder_widths[i + 1]);

    // Decoder mirrors the encoder (reversed widths, then back out to the window).
    std::size_t decoder = 0;
    for (std::size_t i = encoder_widths.size() - 1; i > 0; --i)
        decoder += static_cast<std::size_t>(encoder_widths[i]) *
                   static_cast<std::size_t>(encoder_widths[i - 1]);
    decoder += static_cast<std::size_t>(encoder_widths.front()) * input_features;

    return encoder * static_cast<std::size_t>(std::max(1, encoder_steps)) +
           decoder * static_cast<std::size_t>(std::max(1, decoder_steps));
}

auto estimate_snn_macs(
    std::size_t input_features, const std::vector<int>& encoder_widths, int time_steps)
    -> std::size_t
{
    return estimate_snn_macs(input_features, encoder_widths, time_steps, time_steps);
}

auto arith_energy_nj(double macs, double acs) -> float
{
    return static_cast<float>((kMacEnergyPj * macs + kAcEnergyPj * acs) / 1000.0);
}

namespace
{

// Counts one Sequential's layers into `ops`; `input` is what its first layer received and
// `input_is_spikes` whether it is a spike train. Returns whether the last layer's output is one.
// repeated_steps > 1: the input's rows repeat that many times, so its first Linear is counted
// once per sample (the same rule, and the same checks, as walk_sequential in Meeting01Energy).
auto count_sequential_ops(const nn::Sequential& seq,
    const Tensor& input,
    bool input_is_spikes,
    SnnOpCounts& ops,
    int repeated_steps) -> bool
{
    if (seq.outputs.size() != seq.layers.size())
        throw std::logic_error("count_snn_ops: a Sequential cached " +
                               std::to_string(seq.outputs.size()) + " outputs for " +
                               std::to_string(seq.layers.size()) +
                               " layers. Call it right after model.forward() on the window "
                               "whose operations it should count.");
    if (repeated_steps < 1)
        throw std::invalid_argument("count_snn_ops: repeated_steps must be >= 1 (got " +
                                    std::to_string(repeated_steps) + ")");
    if (repeated_steps > 1)
    {
        const bool first_is_linear_into_lif =
            seq.layers.size() >= 2 && dynamic_cast<const nn::Linear*>(seq.layers[0].get()) &&
            dynamic_cast<const nn::LifBPTT*>(seq.layers[1].get());
        if (input_is_spikes || !first_is_linear_into_lif ||
            input.rows() % static_cast<nn::Index>(repeated_steps) != 0)
            throw std::logic_error(
                "count_snn_ops: a repeated input is counted once only when it is analog, its " +
                std::to_string(input.rows()) + " rows split into " +
                std::to_string(repeated_steps) +
                " equal steps, and it feeds a Linear followed by a LifBPTT; this one does not. "
                "Pass repeated_steps = 1, or extend the count for this topology.");
    }

    const Tensor* in = &input;
    bool in_is_spikes = input_is_spikes;
    for (std::size_t i = 0; i < seq.layers.size(); ++i)
    {
        const auto* layer = seq.layers[i].get();
        const Tensor& out = seq.outputs[i];
        const auto in_values = static_cast<double>(in->size());
        const auto out_values = static_cast<double>(out.size());
        bool out_is_spikes = false;

        if (dynamic_cast<const nn::Linear*>(layer) != nullptr)
        {
            const auto out_features = static_cast<double>(out.cols());
            if (in_is_spikes)
            {
                double spikes = 0.0;
                for (nn::Index k = 0; k < in->size(); ++k)
                {
                    const float v = in->at(k);
                    if (v != 0.0F && v != 1.0F)
                        throw std::logic_error(
                            "count_snn_ops: layer " + std::to_string(i) +
                            " (Linear) was taken to read spikes but its input holds " +
                            std::to_string(v) +
                            ". Check snn_input_is_spikes for this encoding/architecture.");
                    spikes += v;
                }
                ops.acs += spikes * out_features;
            }
            else if (i == 0 && repeated_steps > 1)
            {
                ops.macs += in_values / repeated_steps * out_features; // once per sample
            }
            else
            {
                ops.macs += in_values * out_features; // rows x in x out
            }
        }
        else if (const auto* lif = dynamic_cast<const nn::LifBPTT*>(layer); lif != nullptr)
        {
            ops.macs += out_values;
            if (!lif->readout_mode)
            {
                ops.neuron_steps += out_values;
                ops.spikes += static_cast<double>(out.sum());
                out_is_spikes = true;
            }
        }
        else if (dynamic_cast<const nn::TimeMeanPool*>(layer) != nullptr)
        {
            ops.acs += in_values;
            ops.macs += out_values;
        }
        in = &out;
        in_is_spikes = out_is_spikes;
    }
    return in_is_spikes;
}

} // namespace

auto count_snn_ops(const nn::models::autoencoder::ProtocolSpikingAutoencoder& model,
    const Tensor& input,
    bool input_is_spikes,
    int repeated_steps) -> SnnOpCounts
{
    if (model.use_dual_branch_)
        throw std::logic_error(
            "count_snn_ops: the dual-branch ProtocolSpikingAutoencoder is not counted (meeting01 "
            "never builds it). Extend count_snn_ops over its six Sequentials before using it.");
    SnnOpCounts ops;
    const bool latent_is_spikes =
        count_sequential_ops(model.encoder_, input, input_is_spikes, ops, repeated_steps);
    if (model.encoder_.outputs.empty())
        throw std::logic_error("count_snn_ops: the encoder cached no outputs; run forward first.");
    count_sequential_ops(model.decoder_, model.encoder_.outputs.back(), latent_is_spikes, ops, 1);
    return ops;
}

auto snn_input_repeats_every_step(const std::string& encoding, const std::string& architecture)
    -> bool
{
    (void) snn_input_is_spikes(encoding, architecture); // validates both tokens
    return encoding == "direct" && (architecture == "dense" || architecture == "conv1d");
}

auto snn_input_is_spikes(const std::string& encoding, const std::string& architecture) -> bool
{
    if (encoding != "direct" && encoding != "poisson" && encoding != "latency")
        throw std::invalid_argument("snn_input_is_spikes: unknown encoding '" + encoding +
                                    "' (expected direct, poisson or latency)");
    if (architecture == "recurrent") return true;
    if (architecture == "conv1d") return false;
    if (architecture == "dense") return encoding != "direct";
    throw std::invalid_argument("snn_input_is_spikes: unknown SNN architecture '" + architecture +
                                "' (expected dense, conv1d or recurrent)");
}

auto count_architecture_transform_macs(const std::string& architecture, std::size_t values)
    -> double
{
    if (architecture == "dense") return 0.0;
    if (architecture == "conv1d") return 3.0 * static_cast<double>(values);
    if (architecture == "recurrent") return static_cast<double>(values);
    throw std::invalid_argument("count_architecture_transform_macs: unknown SNN architecture '" +
                                architecture +
                                "' (expected dense, conv1d or recurrent, the tokens "
                                "apply_snn_architecture_transform implements)");
}

auto estimate_gru_macs(const nn::models::gru::GRUAutoencoderConfig& cfg) -> std::size_t
{
    // Same topology as the LSTM autoencoder with 3 gates (r, z, n) where the LSTM has 4, so
    // at matched dimensions every recurrent layer costs 3/4 of the LSTM's.
    return recurrent_ae_macs("estimate_gru_macs",
        3,
        cfg.seq_len,
        cfg.input_size,
        cfg.hidden_size,
        cfg.latent_size,
        cfg.num_layers);
}

auto estimate_transformer_macs(const nn::models::transformer::TransformerAutoencoderConfig& cfg)
    -> std::size_t
{
    const std::size_t T = static_cast<std::size_t>(cfg.seq_len);
    const std::size_t D = static_cast<std::size_t>(cfg.input_size);
    const std::size_t M = static_cast<std::size_t>(cfg.d_model);
    const std::size_t F = static_cast<std::size_t>(cfg.d_ff);
    const std::size_t L = static_cast<std::size_t>(std::max(1, cfg.n_layers));
    const std::size_t Z = static_cast<std::size_t>(cfg.latent_size);

    // Per encoder block: Q/K/V/O projections (4 * T * M * M), the two
    // O(T^2 * M) attention products (scores Q K^T and A V), and the
    // position-wise FFN (2 * T * M * F).
    const std::size_t proj_macs = 4 * T * M * M;
    const std::size_t attn_macs = 2 * T * T * M;
    const std::size_t ffn_macs = 2 * T * M * F;
    const std::size_t per_block = proj_macs + attn_macs + ffn_macs;

    // Encoder and decoder each run L such blocks.
    const std::size_t blocks = 2 * L * per_block;
    // Input embed (T*D*M), latent down/up (M*Z + Z*M), output projection (T*M*D).
    const std::size_t io = T * D * M + M * Z + Z * M + T * M * D;
    return blocks + io;
}

} // namespace meeting01
