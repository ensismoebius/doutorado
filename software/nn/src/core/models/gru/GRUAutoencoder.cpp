/**
 * @file src/core/models/gru/GRUAutoencoder.cpp
 * @brief Implementation of GRUAutoencoder — a port of LSTMAutoencoder with the
 *        recurrent layer swapped for GRULayer.
 */

#include "models/gru/GRUAutoencoder.hpp"

#include <random>
#include <string>
#include <utility>

#include "layers/gru/GRULayer.hpp"

namespace nn::models::gru
{

GRUAutoencoder::GRUAutoencoder(const GRUAutoencoderConfig& cfg) : cfg_(cfg)
{
    const int D = cfg_.input_size;
    const int H = cfg_.hidden_size;
    const int Z = cfg_.latent_size;
    const int L = cfg_.num_layers;

    for (int l = 0; l < L; ++l)
    {
        const int in_dim = (l == 0) ? D : H;
        enc_grus_.push_back(std::make_unique<GRULayer>(in_dim, H));
    }

    auto normal_fill = [](Tensor& t, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.05f);
        for (nn::Index k = 0; k < static_cast<nn::Index>(t.size()); ++k) t.at(k) = dist(rng);
    };

    enc_proj_ = std::make_unique<Linear>(H, Z);
    normal_fill(enc_proj_->weight, 100u);
    enc_proj_->bias.set_zero();

    dec_expand_ = std::make_unique<Linear>(Z, H);
    normal_fill(dec_expand_->weight, 101u);
    dec_expand_->bias.set_zero();

    for (int l = 0; l < L; ++l) dec_grus_.push_back(std::make_unique<GRULayer>(H, H));

    out_proj_ = std::make_unique<Linear>(H, D);
    normal_fill(out_proj_->weight, 102u);
    out_proj_->bias.set_zero();

    build_param_ptrs();
}

void GRUAutoencoder::build_param_ptrs()
{
    param_ptrs_.clear();
    for (auto& gru : enc_grus_)
        for (Tensor* p : gru->params()) param_ptrs_.push_back(p);
    for (Tensor* p : enc_proj_->params()) param_ptrs_.push_back(p);
    for (Tensor* p : dec_expand_->params()) param_ptrs_.push_back(p);
    for (auto& gru : dec_grus_)
        for (Tensor* p : gru->params()) param_ptrs_.push_back(p);
    for (Tensor* p : out_proj_->params()) param_ptrs_.push_back(p);
}

auto GRUAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    const auto in_shape = input.get_shape();
    last_batched_ = (in_shape.size() == 3);
    last_B_ = last_batched_ ? static_cast<int>(in_shape[0]) : 1;
    last_T_ = static_cast<int>(last_batched_ ? in_shape[1] : in_shape[0]);

    Tensor h = input;
    for (auto& gru : enc_grus_)
    {
        gru->reset_state();
        h = gru->forward(h, requires_grad);
    }
    enc_output_cache_ = h;

    Tensor h_last = last_batched_ ? h.slice_time(static_cast<nn::Index>(last_T_ - 1))
                                  : h.row(static_cast<nn::Index>(last_T_ - 1));

    Tensor z_pre = enc_proj_->forward(h_last, requires_grad);
    latent_pre_cache_ = z_pre;
    const Tensor ones = Tensor::ones(z_pre.rows(), z_pre.cols());
    const Tensor two = ones + ones;
    Tensor z = ones.divide(ones + (z_pre * -2.0f).exp()) * two - ones; // tanh(z_pre)
    latent_cache_ = z;
    return z;
}

auto GRUAutoencoder::decode(const Tensor& latent, int seq_len, bool requires_grad) -> Tensor
{
    const int H = cfg_.hidden_size;
    const int D = cfg_.input_size;

    Tensor h_expand = dec_expand_->forward(latent, requires_grad);

    Tensor dec_in;
    if (last_batched_)
    {
        dec_in = Tensor::zeros(static_cast<nn::Index>(last_B_),
            static_cast<nn::Index>(seq_len),
            static_cast<nn::Index>(H));
        for (int t = 0; t < seq_len; ++t)
            dec_in.set_time_slice(static_cast<nn::Index>(t), h_expand);
    }
    else
    {
        dec_in = Tensor::ones(static_cast<nn::Index>(seq_len), 1).matmul(h_expand);
    }
    dec_input_cache_ = dec_in;

    Tensor dec_h = dec_in;
    for (auto& gru : dec_grus_)
    {
        gru->reset_state();
        dec_h = gru->forward(dec_h, requires_grad);
    }
    dec_output_cache_ = dec_h;

    if (last_batched_)
    {
        Tensor dec_h_2d = std::as_const(dec_h).reshape(
            {static_cast<nn::Index>(last_B_ * seq_len), static_cast<nn::Index>(H)});
        Tensor recon_2d = out_proj_->forward(dec_h_2d, requires_grad);
        Tensor recon = std::as_const(recon_2d).reshape({static_cast<nn::Index>(last_B_),
            static_cast<nn::Index>(seq_len),
            static_cast<nn::Index>(D)});
        recon_cache_ = recon;
        return recon;
    }

    Tensor recon = out_proj_->forward(dec_h, requires_grad);
    recon_cache_ = recon;
    return recon;
}

auto GRUAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    requires_grad_ = requires_grad;
    Tensor z = encode(input, requires_grad);
    return decode(z, last_T_, requires_grad);
}

auto GRUAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    const int H = cfg_.hidden_size;
    const int D = cfg_.input_size;
    const int T = last_T_;
    const int B = last_B_;

    Tensor d_dec_h;
    if (last_batched_)
    {
        Tensor g_2d =
            grad_output.reshape({static_cast<nn::Index>(B * T), static_cast<nn::Index>(D)});
        Tensor d_2d = out_proj_->backward(g_2d);
        d_dec_h = std::as_const(d_2d).reshape(
            {static_cast<nn::Index>(B), static_cast<nn::Index>(T), static_cast<nn::Index>(H)});
    }
    else
    {
        d_dec_h = out_proj_->backward(grad_output);
    }

    Tensor d_dec_in = d_dec_h;
    for (int l = static_cast<int>(dec_grus_.size()) - 1; l >= 0; --l)
        d_dec_in = dec_grus_[static_cast<std::size_t>(l)]->backward(d_dec_in);

    Tensor d_h_expand;
    if (last_batched_)
    {
        d_h_expand = Tensor::zeros(static_cast<nn::Index>(B), static_cast<nn::Index>(H));
        for (int t = 0; t < T; ++t)
            d_h_expand = d_h_expand.add(d_dec_in.slice_time(static_cast<nn::Index>(t)));
    }
    else
    {
        d_h_expand = Tensor::ones(1, d_dec_in.rows()).matmul(d_dec_in);
    }
    Tensor d_z = dec_expand_->backward(d_h_expand);

    const Tensor tanh_ones = Tensor::ones(latent_cache_.rows(), latent_cache_.cols());
    Tensor d_z_pre = d_z * (tanh_ones - (latent_cache_ * latent_cache_));
    Tensor d_h_last = enc_proj_->backward(d_z_pre);

    Tensor d_enc_h;
    if (last_batched_)
    {
        d_enc_h = Tensor::zeros(
            static_cast<nn::Index>(B), static_cast<nn::Index>(T), static_cast<nn::Index>(H));
        d_enc_h.set_time_slice(static_cast<nn::Index>(T - 1), d_h_last);
    }
    else
    {
        d_enc_h = Tensor::zeros(static_cast<nn::Index>(T), static_cast<nn::Index>(H));
        d_enc_h.setBlock(static_cast<nn::Index>(T - 1), 0, d_h_last);
    }

    Tensor d_input = d_enc_h;
    for (int l = static_cast<int>(enc_grus_.size()) - 1; l >= 0; --l)
        d_input = enc_grus_[static_cast<std::size_t>(l)]->backward(d_input);

    return d_input;
}

void GRUAutoencoder::reset_state()
{
    for (auto& gru : enc_grus_) gru->reset_state();
    for (auto& gru : dec_grus_) gru->reset_state();
}

auto GRUAutoencoder::params() -> std::span<Tensor*>
{
    return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}

auto GRUAutoencoder::state_dict() const -> std::map<std::string, Tensor>
{
    std::map<std::string, Tensor> sd;
    auto prefix_merge = [&](const std::map<std::string, Tensor>& src, const std::string& pfx)
    {
        for (const auto& [k, v] : src) sd[pfx + k] = v;
    };

    for (std::size_t l = 0; l < enc_grus_.size(); ++l)
        prefix_merge(enc_grus_[l]->state_dict(), "enc_gru" + std::to_string(l) + ".");
    prefix_merge(enc_proj_->state_dict(), "enc_proj.");
    prefix_merge(dec_expand_->state_dict(), "dec_expand.");
    for (std::size_t l = 0; l < dec_grus_.size(); ++l)
        prefix_merge(dec_grus_[l]->state_dict(), "dec_gru" + std::to_string(l) + ".");
    prefix_merge(out_proj_->state_dict(), "out_proj.");
    return sd;
}

void GRUAutoencoder::load_state_dict(const std::map<std::string, Tensor>& sd)
{
    auto extract_prefix = [&](const std::string& pfx) -> std::map<std::string, Tensor>
    {
        std::map<std::string, Tensor> sub;
        for (const auto& [k, v] : sd)
            if (k.substr(0, pfx.size()) == pfx) sub[k.substr(pfx.size())] = v;
        return sub;
    };

    for (std::size_t l = 0; l < enc_grus_.size(); ++l)
        enc_grus_[l]->load_state_dict(extract_prefix("enc_gru" + std::to_string(l) + "."));
    enc_proj_->load_state_dict(extract_prefix("enc_proj."));
    dec_expand_->load_state_dict(extract_prefix("dec_expand."));
    for (std::size_t l = 0; l < dec_grus_.size(); ++l)
        dec_grus_[l]->load_state_dict(extract_prefix("dec_gru" + std::to_string(l) + "."));
    out_proj_->load_state_dict(extract_prefix("out_proj."));
}

} // namespace nn::models::gru
