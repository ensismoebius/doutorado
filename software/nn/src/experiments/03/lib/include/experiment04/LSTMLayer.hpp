#pragma once

/**
 * @file LSTMLayer.hpp
 * @brief Single-layer LSTM cell with full BPTT (Backpropagation Through Time).
 *
 * Gate equations (per time step t):
 *   i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)
 *   f_t = sigma(W_f x_t + U_f h_{t-1} + b_f)
 *   o_t = sigma(W_o x_t + U_o h_{t-1} + b_o)
 *   g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)
 */

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

namespace lstm_autoencoder_experiment
{

inline auto sigmoid(const nn::Tensor& x) -> nn::Tensor
{
    nn::Tensor result(x.rows(), x.cols());
    for (nn::Index i = 0; i < static_cast<nn::Index>(x.size()); ++i)
    {
        result.at(i) = 1.0f / (1.0f + std::exp(-x.at(i)));
    }
    return result;
}

inline auto tanh_tensor(const nn::Tensor& x) -> nn::Tensor
{
    nn::Tensor result(x.rows(), x.cols());
    for (nn::Index i = 0; i < static_cast<nn::Index>(x.size()); ++i)
    {
        result.at(i) = std::tanh(x.at(i));
    }
    return result;
}

inline auto sigmoid_grad(const nn::Tensor& sigmoid_out) -> nn::Tensor
{
    nn::Tensor ones = nn::Tensor::ones(sigmoid_out.rows(), sigmoid_out.cols());
    nn::Tensor one_minus = ones - sigmoid_out;
    return sigmoid_out * one_minus;
}

inline auto tanh_grad(const nn::Tensor& tanh_out) -> nn::Tensor
{
    nn::Tensor sq = tanh_out * tanh_out;
    nn::Tensor ones = nn::Tensor::ones(sq.rows(), sq.cols());
    return ones - sq;
}

struct LSTMStepCache
{
    nn::Tensor x;
    nn::Tensor h_prev;
    nn::Tensor c_prev;
    nn::Tensor i;
    nn::Tensor f;
    nn::Tensor o;
    nn::Tensor g;
    nn::Tensor c;
    nn::Tensor tanh_c;
    nn::Tensor h;
};

class LSTMLayer : public Module<nn::EigenTensorBackend>
{
   public:
    using Tensor = nn::Tensor;

    int input_size_;
    int hidden_size_;
    nn::Tensor W_;
    nn::Tensor U_;
    nn::Tensor b_;
    nn::Tensor dW_;
    nn::Tensor dU_;
    nn::Tensor db_;
    std::vector<nn::Tensor*> param_ptrs_;
    nn::Tensor h0_;
    nn::Tensor c0_;
    std::vector<LSTMStepCache> cache_;
    bool requires_grad_ = false;
    int last_batch_ = 0;

    explicit LSTMLayer(int input_size, int hidden_size)
        : input_size_(input_size),
          hidden_size_(hidden_size),
          W_(4 * hidden_size, input_size),
          U_(4 * hidden_size, hidden_size),
          b_(4 * hidden_size, 1),
          dW_(4 * hidden_size, input_size),
          dU_(4 * hidden_size, hidden_size),
          db_(4 * hidden_size, 1),
          h0_(1, hidden_size),
          c0_(1, hidden_size)
    {
        auto xavier_fill = [](nn::Tensor& t, int fan_in, int fan_out, unsigned seed_offset)
        {
            const float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
            std::mt19937 rng(42u + seed_offset);
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (nn::Index k = 0; k < static_cast<nn::Index>(t.size()); ++k)
            {
                t.at(k) = dist(rng);
            }
        };

        xavier_fill(W_, input_size, hidden_size, 0u);
        xavier_fill(U_, hidden_size, hidden_size, 1u);
        b_.set_zero();
        h0_.set_zero();
        c0_.set_zero();
        dW_.set_zero();
        dU_.set_zero();
        db_.set_zero();

        for (int r = hidden_size_; r < 2 * hidden_size_; ++r)
        {
            b_.at(static_cast<nn::Index>(r), 0) = 1.0f;
        }

        param_ptrs_ = {&W_, &U_, &b_};
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        const int T = static_cast<int>(input.rows());
        const int D = static_cast<int>(input.cols());

        if (D != input_size_)
        {
            throw std::invalid_argument(
                "LSTMLayer::forward: input cols=" + std::to_string(D) +
                " does not match input_size=" + std::to_string(input_size_));
        }

        if (requires_grad)
        {
            cache_.clear();
            cache_.reserve(static_cast<size_t>(T));
        }

        nn::Tensor h = h0_;
        nn::Tensor c = c0_;
        nn::Tensor all_h(static_cast<nn::Index>(T), static_cast<nn::Index>(hidden_size_));

        for (int t = 0; t < T; ++t)
        {
            Tensor x_t = input.row(static_cast<nn::Index>(t));
            Tensor pre =
                x_t.matmul(W_.transpose()).add(h.matmul(U_.transpose())).add(b_.transpose());

            Tensor pre_i = pre.block(0, 0, 1, hidden_size_);
            Tensor pre_f = pre.block(0, 1 * hidden_size_, 1, hidden_size_);
            Tensor pre_o = pre.block(0, 2 * hidden_size_, 1, hidden_size_);
            Tensor pre_g = pre.block(0, 3 * hidden_size_, 1, hidden_size_);

            Tensor i_gate = sigmoid(pre_i);
            Tensor f_gate = sigmoid(pre_f);
            Tensor o_gate = sigmoid(pre_o);
            Tensor g_gate = tanh_tensor(pre_g);

            Tensor c_new = (f_gate * c).add(i_gate * g_gate);
            Tensor tanh_c = tanh_tensor(c_new);
            Tensor h_new = o_gate * tanh_c;

            if (requires_grad)
            {
                LSTMStepCache step{x_t, h, c, i_gate, f_gate, o_gate, g_gate, c_new, tanh_c, h_new};
                cache_.push_back(std::move(step));
            }

            for (nn::Index col = 0; col < static_cast<nn::Index>(hidden_size_); ++col)
            {
                all_h.at(static_cast<nn::Index>(t), col) = h_new.at(0, col);
            }

            h = h_new;
            c = c_new;
        }

        h0_ = h;
        c0_ = c;
        return all_h;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const int T = static_cast<int>(cache_.size());
        if (T == 0)
        {
            throw std::runtime_error(
                "LSTMLayer::backward called before forward with requires_grad");
        }

        dW_.set_zero();
        dU_.set_zero();
        db_.set_zero();

        nn::Tensor dx_all(static_cast<nn::Index>(T), static_cast<nn::Index>(input_size_));
        dx_all.set_zero();

        nn::Tensor dh_next = nn::Tensor::zeros(1, hidden_size_);
        nn::Tensor dc_next = nn::Tensor::zeros(1, hidden_size_);

        for (int t = T - 1; t >= 0; --t)
        {
            const auto& step = cache_[static_cast<size_t>(t)];
            nn::Tensor dh = grad_output.row(static_cast<nn::Index>(t)).add(dh_next);
            nn::Tensor do_gate = dh * step.tanh_c;
            nn::Tensor dtanh_c = dh * step.o;
            nn::Tensor dc = (dtanh_c * tanh_grad(step.tanh_c)).add(dc_next);
            nn::Tensor di_gate = dc * step.g;
            nn::Tensor df_gate = dc * step.c_prev;
            nn::Tensor dg_gate = dc * step.i;
            dc_next = dc * step.f;

            nn::Tensor dpre_i = di_gate * sigmoid_grad(step.i);
            nn::Tensor dpre_f = df_gate * sigmoid_grad(step.f);
            nn::Tensor dpre_o = do_gate * sigmoid_grad(step.o);
            nn::Tensor dpre_g = dg_gate * tanh_grad(step.g);

            nn::Tensor dpre(1, 4 * hidden_size_);
            for (nn::Index col = 0; col < static_cast<nn::Index>(hidden_size_); ++col)
            {
                dpre.at(0, 0 * hidden_size_ + col) = dpre_i.at(0, col);
                dpre.at(0, 1 * hidden_size_ + col) = dpre_f.at(0, col);
                dpre.at(0, 2 * hidden_size_ + col) = dpre_o.at(0, col);
                dpre.at(0, 3 * hidden_size_ + col) = dpre_g.at(0, col);
            }

            nn::Tensor dW_t = dpre.transpose().matmul(step.x);
            dW_.add_inplace(dW_t);
            nn::Tensor dU_t = dpre.transpose().matmul(step.h_prev);
            dU_.add_inplace(dU_t);
            nn::Tensor db_t = dpre.transpose();
            db_.add_inplace(db_t);

            nn::Tensor dx_t = dpre.matmul(W_);
            for (nn::Index col = 0; col < static_cast<nn::Index>(input_size_); ++col)
            {
                dx_all.at(static_cast<nn::Index>(t), col) = dx_t.at(0, col);
            }

            dh_next = dpre.matmul(U_);
        }

        W_.set_grad(dW_);
        U_.set_grad(dU_);
        b_.set_grad(db_);

        return dx_all;
    }

    void reset_state() override
    {
        h0_.set_zero();
        c0_.set_zero();
        cache_.clear();
    }

    auto params() -> std::span<nn::Tensor*> override
    {
        return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, nn::Tensor> override
    {
        return {{"W", W_}, {"U", U_}, {"b", b_}};
    }

    void load_state_dict(const std::map<std::string, nn::Tensor>& sd) override
    {
        if (auto it = sd.find("W"); it != sd.end()) W_ = it->second;
        if (auto it = sd.find("U"); it != sd.end()) U_ = it->second;
        if (auto it = sd.find("b"); it != sd.end()) b_ = it->second;
    }
};

} // namespace lstm_autoencoder_experiment