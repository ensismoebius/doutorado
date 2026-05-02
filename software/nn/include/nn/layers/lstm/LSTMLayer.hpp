#ifndef NN_LAYERS_LSTM_LSTMLAYER_HPP
#define NN_LAYERS_LSTM_LSTMLAYER_HPP

/**
 * @file include/nn/layers/lstm/LSTMLayer.hpp
 * @brief Single-layer LSTM cell with full BPTT and batch gradient accumulation.
 *
 * Gate equations per time step t — verified against [1, 2]:
 *   pre_t  = x_t W^T + H_{t-1} U^T + b^T          // (B, 4H)
 *   i_t    = sigma(pre_t[0:H])                      // input gate
 *   f_t    = sigma(pre_t[H:2H])                     // forget gate (bias init=1 [3])
 *   o_t    = sigma(pre_t[2H:3H])                    // output gate
 *   g_t    = tanh(pre_t[3H:4H])                     // cell candidate
 *   C_t    = f_t ⊙ C_{t-1} + i_t ⊙ g_t            // cell state
 *   H_t    = o_t ⊙ tanh(C_t)                       // hidden state
 *
 * Gate ordering [i|f|o|g] follows Greff et al. (2015) [2] convention.
 * Weights stored as stacked matrices in the same order.
 *
 * References:
 *   [1] Hochreiter & Schmidhuber, Neural Computation 9(8), 1997.
 *   [2] Greff et al., IEEE TNNLS 28(10), 2017. arXiv:1503.04069
 *   [3] Jozefowicz et al., ICML 2015 (forget-gate bias initialisation).
 *
 * Shape contract:
 *   forward(Tensor{T, D})    → Tensor{T, H}    — single sequence, 2D
 *   forward(Tensor{B, T, D}) → Tensor{B, T, H} — batched, 3D
 *   backward(Tensor{T, H})   → Tensor{T, D}    — single
 *   backward(Tensor{B, T, H})→ Tensor{B, T, D} — batch; grads W/U/b accumulated over B
 */

#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "nn/layers/activations/Sigmoid.hpp"
#include "nn/layers/activations/Tanh.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

template <typename Backend>
struct LSTMStepCacheT
{
    using Tensor = nn::TensorImpl<Backend>;
    Tensor x;      // x_t (B, D)
    Tensor h_prev; // H_{t-1} (B, H)
    Tensor c_prev; // C_{t-1} (B, H)
    Tensor i;      // σ(pre_i) (B, H)
    Tensor f;      // σ(pre_f) (B, H)
    Tensor o;      // σ(pre_o) (B, H)
    Tensor g;      // tanh(pre_g) (B, H)
    Tensor tanh_c; // tanh(C_t) (B, H)
};

template <typename Backend>
class LSTMLayerImpl : public Module<Backend>
{
   public:
    using Tensor = nn::TensorImpl<Backend>;

    int input_size_;
    int hidden_size_;

    Tensor W_;
    Tensor U_;
    Tensor b_;

    Tensor dW_;
    Tensor dU_;
    Tensor db_;

    std::vector<Tensor*> param_ptrs_;

    Tensor h0_;
    Tensor c0_;

    std::vector<LSTMStepCacheT<Backend>> cache_;
    bool requires_grad_ = false;

    explicit LSTMLayerImpl(int input_size, int hidden_size)
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
        auto normal_fill = [](Tensor& t, unsigned seed_offset)
        {
            std::mt19937 rng(42u + seed_offset);
            std::normal_distribution<float> dist(0.0f, 0.05f);
            for (nn::Index k = 0; k < static_cast<nn::Index>(t.size()); ++k) t.at(k) = dist(rng);
        };

        normal_fill(W_, 0u);
        normal_fill(U_, 1u);
        b_.set_zero();
        h0_.set_zero();
        c0_.set_zero();
        dW_.set_zero();
        dU_.set_zero();
        db_.set_zero();

        for (int r = hidden_size_; r < 2 * hidden_size_; ++r)
            b_.at(static_cast<nn::Index>(r), 0) = 1.0f;

        param_ptrs_ = {&W_, &U_, &b_};
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        const auto& shape = input.get_shape();

        int B = 1;
        Tensor seq_3d;
        if (shape.size() == 2)
        {
            // (T, D) → (1, T, D)
            seq_3d = input.reshape({1, shape[0], shape[1]});
        }
        else
        {
            B = static_cast<int>(shape[0]);
            seq_3d = input;
        }

        const int T_seq = static_cast<int>(seq_3d.get_shape()[1]);
        const int D_in = static_cast<int>(seq_3d.get_shape()[2]);
        if (D_in != input_size_)
            throw std::invalid_argument("LSTMLayerImpl::forward: input D=" + std::to_string(D_in) +
                                        " != input_size=" + std::to_string(input_size_));

        if (requires_grad)
        {
            cache_.clear();
            cache_.reserve(T_seq);
        }

        // Initial states (B, H)
        Tensor h = Tensor::zeros(static_cast<nn::Index>(B), static_cast<nn::Index>(hidden_size_));
        Tensor c = Tensor::zeros(static_cast<nn::Index>(B), static_cast<nn::Index>(hidden_size_));

        if (shape.size() == 2)
        {
            h = h0_;
            c = c0_;
        }

        Tensor b_T = b_.transpose(); // (1, 4H) — computed once outside loop

        auto sigmoid_tensor = [](const Tensor& x) -> Tensor
        {
            const Tensor ones = Tensor::ones(x.rows(), x.cols());
            return ones.divide(ones + (x * -1.0f).exp());
        };
        auto tanh_tensor = [&](const Tensor& x) -> Tensor
        {
            const Tensor ones = Tensor::ones(x.rows(), x.cols());
            const Tensor two = ones + ones;
            return sigmoid_tensor(x * 2.0f) * two - ones;
        };

        // Opt: for single-sequence (B=1), write directly to (T,H) output; skip 3D alloc+copy.
        const bool is_2d = (shape.size() == 2);
        Tensor all_out = is_2d ? Tensor::zeros(static_cast<nn::Index>(T_seq),
                                     static_cast<nn::Index>(hidden_size_))
                               : Tensor::zeros(static_cast<nn::Index>(B),
                                     static_cast<nn::Index>(T_seq),
                                     static_cast<nn::Index>(hidden_size_));

        for (int t = 0; t < T_seq; ++t)
        {
            // Opt: vectorized slice instead of scalar copy loop.
            Tensor x_t = seq_3d.slice_time(static_cast<nn::Index>(t)); // (B, D)

            Tensor pre =
                x_t.matmul_transposed(W_).add(h.matmul_transposed(U_)).add_row_broadcast(b_T);

            Tensor i_g = sigmoid_tensor(pre.block(0, 0, B, hidden_size_));
            Tensor f_g = sigmoid_tensor(pre.block(0, 1 * hidden_size_, B, hidden_size_));
            Tensor o_g = sigmoid_tensor(pre.block(0, 2 * hidden_size_, B, hidden_size_));
            Tensor g_g = tanh_tensor(pre.block(0, 3 * hidden_size_, B, hidden_size_));

            Tensor c_new = (f_g * c).add(i_g * g_g);
            Tensor tc = tanh_tensor(c_new);
            Tensor h_new = o_g * tc;

            if (requires_grad) cache_.push_back({x_t, h, c, i_g, f_g, o_g, g_g, tc});

            // Opt: vectorized write instead of scalar copy loop.
            if (is_2d)
                all_out.setBlock(static_cast<nn::Index>(t), 0, h_new); // h_new is (1,H) when B=1
            else
                all_out.set_time_slice(static_cast<nn::Index>(t), h_new); // h_new is (B,H)

            h = h_new;
            c = c_new;
        }

        if (is_2d)
        {
            h0_ = h;
            c0_ = c;
        }
        return all_out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (cache_.empty())
        {
            throw std::runtime_error(
                "LSTMLayerImpl::backward called before forward(requires_grad=true)");
        }

        const auto& shape = grad_output.get_shape();
        int B = (shape.size() == 3) ? static_cast<int>(shape[0]) : 1;

        Tensor go_3d;
        if (shape.size() == 2)
        {
            // (T, H) → (1, T, H)
            go_3d = grad_output.reshape({1, shape[0], shape[1]});
        }
        else
        {
            go_3d = grad_output;
        }

        auto [dW, dU, db, dx_3d] = _bptt_pure(cache_, go_3d, B);

        W_.set_grad(dW);
        U_.set_grad(dU);
        b_.set_grad(db);
        dW_ = dW;
        dU_ = dU;
        db_ = db;

        if (shape.size() == 2)
        {
            // Return (T, D) for single sequence
            Tensor dx2d = Tensor::zeros(
                static_cast<nn::Index>(shape[0]), static_cast<nn::Index>(input_size_));
            for (int t = 0; t < static_cast<int>(shape[0]); ++t)
                for (int d = 0; d < input_size_; ++d) dx2d.at(t, d) = dx_3d.at(0, t, d);
            return dx2d;
        }
        return dx_3d;
    }

    void reset_state() override
    {
        h0_.set_zero();
        c0_.set_zero();
        cache_.clear();
    }

    auto params() -> std::span<Tensor*> override
    {
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        return {{"W", W_}, {"U", U_}, {"b", b_}};
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        if (auto it = sd.find("W"); it != sd.end()) W_ = it->second;
        if (auto it = sd.find("U"); it != sd.end()) U_ = it->second;
        if (auto it = sd.find("b"); it != sd.end()) b_ = it->second;
    }

   private:
    auto _bptt_pure(
        const std::vector<LSTMStepCacheT<Backend>>& step_cache, const Tensor& grad_output, int B)
        -> std::tuple<Tensor, Tensor, Tensor, Tensor>
    {
        const int T = static_cast<int>(step_cache.size());
        const int H = hidden_size_;

        Tensor dW = Tensor::zeros(dW_.rows(), dW_.cols());
        Tensor dU = Tensor::zeros(dU_.rows(), dU_.cols());
        Tensor db = Tensor::zeros(db_.rows(), db_.cols());
        Tensor dx_all = Tensor::zeros(static_cast<nn::Index>(B),
            static_cast<nn::Index>(T),
            static_cast<nn::Index>(input_size_));

        Tensor dh_next = Tensor::zeros(static_cast<nn::Index>(B), static_cast<nn::Index>(H));
        Tensor dc_next = Tensor::zeros(static_cast<nn::Index>(B), static_cast<nn::Index>(H));

        auto sigmoid_grad_from_output = [](const Tensor& y) -> Tensor
        {
            const Tensor ones = Tensor::ones(y.rows(), y.cols());
            return y * (ones - y);
        };
        auto tanh_grad_from_output = [](const Tensor& y) -> Tensor
        {
            const Tensor ones = Tensor::ones(y.rows(), y.cols());
            return ones - (y * y);
        };

        // Opt: pre-allocate dpre once; overwrite each timestep via setBlock.
        Tensor dpre(static_cast<nn::Index>(B), 4 * static_cast<nn::Index>(H));

        for (int t = T - 1; t >= 0; --t)
        {
            const auto& step = step_cache[static_cast<std::size_t>(t)];

            // Opt: vectorized slice instead of scalar loop over (B, H).
            Tensor dh = grad_output.slice_time(static_cast<nn::Index>(t)); // (B, H)
            dh = dh.add(dh_next);

            Tensor do_gate = dh * step.tanh_c;
            Tensor dtanh_c = dh * step.o;
            Tensor dc = (dtanh_c * tanh_grad_from_output(step.tanh_c)).add(dc_next);

            Tensor di_gate = dc * step.g;
            Tensor df_gate = dc * step.c_prev;
            Tensor dg_gate = dc * step.i;
            dc_next = dc * step.f;

            Tensor dpre_i = di_gate * sigmoid_grad_from_output(step.i);
            Tensor dpre_f = df_gate * sigmoid_grad_from_output(step.f);
            Tensor dpre_o = do_gate * sigmoid_grad_from_output(step.o);
            Tensor dpre_g = dg_gate * tanh_grad_from_output(step.g);

            // Opt: block writes instead of scalar scatter.
            dpre.setBlock(0, 0 * H, dpre_i);
            dpre.setBlock(0, 1 * H, dpre_f);
            dpre.setBlock(0, 2 * H, dpre_o);
            dpre.setBlock(0, 3 * H, dpre_g);

            Tensor dpre_T = dpre.transpose(); // (4H, B)
            dW.add_inplace(dpre_T.matmul(step.x));
            dU.add_inplace(dpre_T.matmul(step.h_prev));
            // Opt: sum over batch before accumulating to (4H, 1).
            db.add_inplace(dpre_T.rowwise_sum());

            // Opt: vectorized write instead of scalar loop.
            dx_all.set_time_slice(static_cast<nn::Index>(t), dpre.matmul(W_));
            dh_next = dpre.matmul(U_);
        }

        return {dW, dU, db, dx_all};
    }
};

namespace nn::models::lstm
{
using LSTMLayer = LSTMLayerImpl<nn::Backend>;
} // namespace nn::models::lstm

#endif // NN_LAYERS_LSTM_LSTMLAYER_HPP
