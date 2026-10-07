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

auto estimate_snn_macs(std::size_t input_features, int hidden_size, int layers) -> std::size_t
{
    const std::size_t H = static_cast<std::size_t>(hidden_size);
    const std::size_t L = static_cast<std::size_t>(std::max(1, layers));
    const std::size_t in_proj = input_features * H;
    const std::size_t hidden_proj = (L > 1) ? (L - 1) * H * H : 0;
    const std::size_t out_proj = H * input_features;
    return in_proj + hidden_proj + out_proj;
}

auto estimate_snn_macs(
    std::size_t input_features, const std::vector<int>& encoder_widths, int time_steps)
    -> std::size_t
{
    // Sums the REAL per-layer projections instead of assuming every hidden layer is as
    // wide as the first. The (first_width, depth) approximation gave {128, 8} and
    // {128, 120} an identical cost, so the GA's second objective could not tell a cheap
    // genome from an expensive one. Encoder is mirrored by the decoder, and the whole
    // stack is evaluated once per simulation step.
    if (encoder_widths.empty()) return 0;

    std::size_t per_step = input_features * static_cast<std::size_t>(encoder_widths.front());
    for (std::size_t i = 0; i + 1 < encoder_widths.size(); ++i)
        per_step += static_cast<std::size_t>(encoder_widths[i]) *
                    static_cast<std::size_t>(encoder_widths[i + 1]);

    // Decoder mirrors the encoder (reversed widths, then back out to the window).
    std::size_t decoder = 0;
    for (std::size_t i = encoder_widths.size() - 1; i > 0; --i)
        decoder += static_cast<std::size_t>(encoder_widths[i]) *
                   static_cast<std::size_t>(encoder_widths[i - 1]);
    decoder += static_cast<std::size_t>(encoder_widths.front()) * input_features;

    return (per_step + decoder) * static_cast<std::size_t>(std::max(1, time_steps));
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
