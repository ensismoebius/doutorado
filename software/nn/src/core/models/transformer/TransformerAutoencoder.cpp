/**
 * @file src/core/models/transformer/TransformerAutoencoder.cpp
 * @brief Implementation of the bottlenecked Transformer autoencoder.
 */

#include "models/transformer/TransformerAutoencoder.hpp"

#include <random>
#include <stdexcept>
#include <string>

#include "layers/attention/PositionalEncoding.hpp"

namespace nn::models::transformer
{

using nn::layers::sinusoidal_positional_encoding;

namespace
{
void init_linear(LinearImpl<nn::Backend>& lin, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (nn::Index k = 0; k < static_cast<nn::Index>(lin.weight.size()); ++k)
        lin.weight.at(k) = dist(rng);
    lin.bias.set_zero();
}
} // namespace

TransformerAutoencoder::TransformerAutoencoder(const TransformerAutoencoderConfig& cfg) : cfg_(cfg)
{
    const int D = cfg_.input_size;
    const int M = cfg_.d_model;
    const int Z = cfg_.latent_size;

    if (D <= 0 || M <= 0 || Z <= 0 || cfg_.seq_len <= 0 || cfg_.n_layers <= 0)
        throw std::invalid_argument("TransformerAutoencoder: all dimensions must be > 0");

    embed_ = std::make_unique<Linear>(D, M);
    init_linear(*embed_, 4100u);
    to_latent_ = std::make_unique<Linear>(M, Z);
    init_linear(*to_latent_, 4101u);
    from_latent_ = std::make_unique<Linear>(Z, M);
    init_linear(*from_latent_, 4102u);
    out_proj_ = std::make_unique<Linear>(M, D);
    init_linear(*out_proj_, 4103u);

    for (int l = 0; l < cfg_.n_layers; ++l)
    {
        enc_blocks_.push_back(
            std::make_unique<Block>(M, cfg_.n_heads, cfg_.d_ff, static_cast<unsigned>(200 + l)));
        dec_blocks_.push_back(
            std::make_unique<Block>(M, cfg_.n_heads, cfg_.d_ff, static_cast<unsigned>(300 + l)));
    }

    pe_ = sinusoidal_positional_encoding<nn::Backend>(cfg_.seq_len, M);

    build_param_ptrs();
}

void TransformerAutoencoder::build_param_ptrs()
{
    param_ptrs_.clear();
    auto take = [&](std::span<Tensor*> ps)
    {
        for (Tensor* p : ps) param_ptrs_.push_back(p);
    };
    take(embed_->params());
    for (auto& b : enc_blocks_) take(b->params());
    take(to_latent_->params());
    take(from_latent_->params());
    for (auto& b : dec_blocks_) take(b->params());
    take(out_proj_->params());
}

auto TransformerAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    if (input.get_shape().size() != 2 || static_cast<int>(input.cols()) != cfg_.input_size)
        throw std::invalid_argument("TransformerAutoencoder::encode: expected (T, input_size)");

    last_T_ = static_cast<int>(input.rows());
    const Tensor pe =
        pe_.block(0, 0, static_cast<nn::Index>(last_T_), static_cast<nn::Index>(cfg_.d_model));

    Tensor h = embed_->forward(input, requires_grad).add(pe); // (T, d_model)
    for (auto& b : enc_blocks_) h = b->forward(h, requires_grad);

    // Mean-pool over time → (1, d_model).
    const Tensor pooled = Tensor::ones(1, static_cast<nn::Index>(last_T_)).matmul(h) *
                          (1.0f / static_cast<float>(last_T_));

    const Tensor z_pre = to_latent_->forward(pooled, requires_grad); // (1, Z)
    const Tensor ones = Tensor::ones(z_pre.rows(), z_pre.cols());
    const Tensor two = ones + ones;
    const Tensor z = ones.divide(ones + (z_pre * -2.0f).exp()) * two - ones; // tanh
    latent_cache_ = z;
    if (requires_grad) forward_cached_ = true;
    return z;
}

auto TransformerAutoencoder::decode(const Tensor& latent, int seq_len, bool requires_grad) -> Tensor
{
    const Tensor pe =
        pe_.block(0, 0, static_cast<nn::Index>(seq_len), static_cast<nn::Index>(cfg_.d_model));

    const Tensor expand = from_latent_->forward(latent, requires_grad); // (1, d_model)
    Tensor h =
        Tensor::ones(static_cast<nn::Index>(seq_len), 1).matmul(expand).add(pe); // (T, d_model)

    for (auto& b : dec_blocks_) h = b->forward(h, requires_grad);

    return out_proj_->forward(h, requires_grad); // (T, D)
}

auto TransformerAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    const Tensor z = encode(input, requires_grad);
    return decode(z, last_T_, requires_grad);
}

auto TransformerAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    if (!forward_cached_)
        throw std::runtime_error(
            "TransformerAutoencoder::backward called before forward(requires_grad=true)");

    const int T = last_T_;

    // ---- decoder ----
    Tensor d_h = out_proj_->backward(grad_output); // (T, d_model)
    for (int l = static_cast<int>(dec_blocks_.size()) - 1; l >= 0; --l)
        d_h = dec_blocks_[static_cast<std::size_t>(l)]->backward(d_h);
    // + pe : identity passthrough.
    // replicate(expand, T): d_expand = sum over the T rows → (1, d_model)
    const Tensor d_expand = Tensor::ones(1, static_cast<nn::Index>(T)).matmul(d_h);
    Tensor d_z = from_latent_->backward(d_expand); // (1, Z)

    // ---- latent tanh ----
    const Tensor tanh_ones = Tensor::ones(latent_cache_.rows(), latent_cache_.cols());
    const Tensor d_z_pre = d_z * (tanh_ones - (latent_cache_ * latent_cache_));

    // ---- encoder ----
    const Tensor d_pooled = to_latent_->backward(d_z_pre); // (1, d_model)
    // mean-pool backward: every time step receives d_pooled / T.
    Tensor d_enc_h = Tensor::ones(static_cast<nn::Index>(T), 1).matmul(d_pooled) *
                     (1.0f / static_cast<float>(T)); // (T, d_model)
    for (int l = static_cast<int>(enc_blocks_.size()) - 1; l >= 0; --l)
        d_enc_h = enc_blocks_[static_cast<std::size_t>(l)]->backward(d_enc_h);
    // + pe : identity passthrough.
    return embed_->backward(d_enc_h); // (T, D)
}

void TransformerAutoencoder::reset_state()
{
    for (auto& b : enc_blocks_) b->reset_state();
    for (auto& b : dec_blocks_) b->reset_state();
    forward_cached_ = false;
}

auto TransformerAutoencoder::params() -> std::span<Tensor*>
{
    return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}

auto TransformerAutoencoder::state_dict() const -> std::map<std::string, Tensor>
{
    std::map<std::string, Tensor> sd;
    auto merge = [&](const std::string& pfx, const std::map<std::string, Tensor>& src)
    {
        for (const auto& [k, v] : src) sd[pfx + k] = v;
    };
    merge("embed.", embed_->state_dict());
    for (std::size_t l = 0; l < enc_blocks_.size(); ++l)
        merge("enc" + std::to_string(l) + ".", enc_blocks_[l]->state_dict());
    merge("to_latent.", to_latent_->state_dict());
    merge("from_latent.", from_latent_->state_dict());
    for (std::size_t l = 0; l < dec_blocks_.size(); ++l)
        merge("dec" + std::to_string(l) + ".", dec_blocks_[l]->state_dict());
    merge("out_proj.", out_proj_->state_dict());
    return sd;
}

void TransformerAutoencoder::load_state_dict(const std::map<std::string, Tensor>& sd)
{
    auto sub = [&](const std::string& pfx)
    {
        std::map<std::string, Tensor> out;
        for (const auto& [k, v] : sd)
            if (k.rfind(pfx, 0) == 0) out[k.substr(pfx.size())] = v;
        return out;
    };
    embed_->load_state_dict(sub("embed."));
    for (std::size_t l = 0; l < enc_blocks_.size(); ++l)
        enc_blocks_[l]->load_state_dict(sub("enc" + std::to_string(l) + "."));
    to_latent_->load_state_dict(sub("to_latent."));
    from_latent_->load_state_dict(sub("from_latent."));
    for (std::size_t l = 0; l < dec_blocks_.size(); ++l)
        dec_blocks_[l]->load_state_dict(sub("dec" + std::to_string(l) + "."));
    out_proj_->load_state_dict(sub("out_proj."));
}

} // namespace nn::models::transformer
