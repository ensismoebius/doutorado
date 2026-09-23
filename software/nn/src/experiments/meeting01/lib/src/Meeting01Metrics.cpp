#include "../include/Meeting01Metrics.hpp"

#include <algorithm>

namespace meeting01
{

auto estimate_lstm_macs(const nn::models::lstm::LSTMAutoencoderConfig& cfg) -> std::size_t
{
    const std::size_t T = static_cast<std::size_t>(cfg.seq_len);
    const std::size_t I = static_cast<std::size_t>(cfg.input_size);
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t L = static_cast<std::size_t>(cfg.num_layers);

    const std::size_t per_gate = H * (I + H);
    const std::size_t per_step = 4 * per_gate;
    const std::size_t per_stack = per_step * L;
    const std::size_t proj = H * static_cast<std::size_t>(cfg.latent_size) +
                             static_cast<std::size_t>(cfg.latent_size) * H + H * I;
    return T * per_stack + proj;
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
    const std::size_t T = static_cast<std::size_t>(cfg.seq_len);
    const std::size_t I = static_cast<std::size_t>(cfg.input_size);
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t L = static_cast<std::size_t>(std::max(1, cfg.num_layers));
    const std::size_t Z = static_cast<std::size_t>(cfg.latent_size);

    // Mirrors estimate_lstm_macs() exactly so the two are directly comparable:
    // one stack of L layers unrolled T steps, plus the projection heads. GRU has
    // 3 gates (r, z, n) where the LSTM has 4, so at matched dimensions the GRU
    // estimate is strictly 3/4 of the LSTM's recurrent term.
    const std::size_t per_gate = H * (I + H);
    const std::size_t per_step = 3 * per_gate;
    const std::size_t per_stack = per_step * L;
    const std::size_t proj = H * Z + Z * H + H * I;
    return T * per_stack + proj;
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
