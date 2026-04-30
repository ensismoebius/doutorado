#ifndef NN_LAYERS_LSTM_LSTMLAYER_HPP
#define NN_LAYERS_LSTM_LSTMLAYER_HPP

/**
 * @file include/nn/layers/lstm/LSTMLayer.hpp
 * @brief Single-layer LSTM cell with full BPTT and batch gradient accumulation.
 *
 * Gate equations (per time step t):
 *   i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)   // input gate
 *   f_t = sigma(W_f x_t + U_f h_{t-1} + b_f)   // forget gate
 *   o_t = sigma(W_o x_t + U_o h_{t-1} + b_o)   // output gate
 *   g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)    // cell gate
 *
 * Parameters are stored as stacked matrices [i|f|o|g].
 * Hidden and cell state persist across forward() calls; call reset_state()
 * between independent sequences.
 *
 * Shape contract:
 *   forward(Tensor{T, D})    → Tensor{T, H}    — single sequence, 2D
 *   forward(Tensor{B, T, D}) → Tensor{B, T, H} — batched, 3D
 *   backward(Tensor{T, H})   → Tensor{T, D}    — single
 *   backward(Tensor{B, T, H})→ Tensor{B, T, D} — batch; grads W/U/b accumulated over B
 *
 * Backend pattern (follows LinearImpl<Backend>):
 *   - LSTMLayerImpl is in global namespace, matching all other *Impl layer classes.
 *   - Trainable params (W_, U_, b_) always CPU-resident nn::Tensor — optimizer compat.
 *   - forward()/backward() interface uses Tensor = TensorImpl<Backend>.
 *   - Internal LSTM computation is CPU-xtensor regardless of Backend.
 *   - nn::models::lstm::LSTMLayer is a backward-compat alias.
 */

#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "nn/layers/activations/Sigmoid.hpp"
#include "nn/layers/activations/Tanh.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

// ---------------------------------------------------------------------------
// Per-timestep cache for BPTT (CPU-resident nn::Tensor throughout)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// LSTMLayerImpl<Backend>  — global namespace, same convention as LinearImpl etc.
// ---------------------------------------------------------------------------

/**
 * @class LSTMLayerImpl
 * @brief One stacked LSTM layer.  Weights:
 *   W_ : (4H, D) — input-to-hidden, gates stacked [i|f|o|g]
 *   U_ : (4H, H) — hidden-to-hidden
 *   b_ : (4H, 1) — bias (forget-gate bias initialised to 1)
 *
 * Trainable params are always CPU-resident nn::Tensor so existing CPU optimizers
 * work without modification (same convention as LinearImpl<Backend>).
 */
template <typename Backend>
class LSTMLayerImpl : public Module<Backend>
{
   public:
    /// Tensor type for the active compute backend (forward/backward interface).
    using Tensor = nn::TensorImpl<Backend>;

    int input_size_;
    int hidden_size_;

    // Trainable params — always CPU-resident for optimizer compat.
    nn::Tensor W_;
    nn::Tensor U_;
    nn::Tensor b_;

    nn::Tensor dW_;
    nn::Tensor dU_;
    nn::Tensor db_;

    std::vector<nn::Tensor*> param_ptrs_;

    nn::Tensor h0_;
    nn::Tensor c0_;

    std::vector<LSTMStepCache>              cache_;        // single-sample BPTT cache
    std::vector<std::vector<LSTMStepCache>> batch_caches_; // multi-sample caches
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
        auto normal_fill = [](nn::Tensor& t, unsigned seed_offset)
        {
            std::mt19937 rng(42u + seed_offset);
            std::normal_distribution<float> dist(0.0f, 0.05f);
                for (nn::Index k = 0; k < static_cast<nn::Index>(t.size()); ++k)
                    t.at(k) = dist(rng);

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

    // --- single-sample forward ---

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        const auto& shape = input.get_shape();

        if (shape.size() == 3)
            return forward_3d(input, requires_grad);
        else
            return forward_2d(input, requires_grad);
    }

    // --- single-sample backward ---

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (grad_output.get_shape().size() == 3)
            return backward_3d(grad_output);
        else
            return Tensor(_bptt_apply(cache_, nn::Tensor(grad_output), false));
    }

    // --- state management ---

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

    private:

    // Core LSTM gate loop over one CPU-resident (T, D) sequence.
    // Returns {all_h (T,H), final_h (1,H), final_c (1,H)}.
    // Appends step caches to *cache_out when requires_grad && cache_out != nullptr.
    auto _run_sequence(const nn::Tensor& seq,
                       nn::Tensor h, nn::Tensor c,
                       bool requires_grad,
                       std::vector<LSTMStepCache>* cache_out)
        -> std::tuple<nn::Tensor, nn::Tensor, nn::Tensor>
    {
        const int T_seq = static_cast<int>(seq.rows());
        nn::Tensor all_h(static_cast<nn::Index>(T_seq),
                         static_cast<nn::Index>(hidden_size_));

        for (int t = 0; t < T_seq; ++t)
        {
            nn::Tensor x_t = seq.row(static_cast<nn::Index>(t));
            nn::Tensor pre = x_t.matmul_transposed(W_)
                                 .add(h.matmul_transposed(U_))
                                 .add(b_.transpose());


            nn::Tensor i_g = nn::activation::sigmoid(pre.block(0, 0,              1, hidden_size_));
            nn::Tensor f_g = nn::activation::sigmoid(pre.block(0, 1*hidden_size_, 1, hidden_size_));
            nn::Tensor o_g = nn::activation::sigmoid(pre.block(0, 2*hidden_size_, 1, hidden_size_));
            nn::Tensor g_g = nn::activation::tanh(  pre.block(0, 3*hidden_size_, 1, hidden_size_));

            nn::Tensor c_new = (f_g * c).add(i_g * g_g);
            nn::Tensor tc    = nn::activation::tanh(c_new);
            nn::Tensor h_new = o_g * tc;

            if (requires_grad && cache_out)
                cache_out->push_back({x_t, h, c, i_g, f_g, o_g, g_g, c_new, tc, h_new});

            all_h.setBlock(static_cast<nn::Index>(t), 0, h_new);
            h = h_new;
            c = c_new;
        }
        return {all_h, h, c};
    }

    // 2D single-sample forward: input (T, D) → output (T, H). Persists h0_/c0_.
    auto forward_2d(const Tensor& input, bool requires_grad) -> Tensor
    {
        const int D_in = static_cast<int>(input.cols());
        if (D_in != input_size_)
            throw std::invalid_argument("LSTMLayerImpl::forward: input cols=" +
                                        std::to_string(D_in) + " != input_size=" +
                                        std::to_string(input_size_));

        const int T_seq = static_cast<int>(input.rows());
        if (requires_grad) { cache_.clear(); cache_.reserve(T_seq); }

        nn::Tensor seq(input);
        auto [all_h, h_f, c_f] = _run_sequence(seq, h0_, c0_, requires_grad,
                                                requires_grad ? &cache_ : nullptr);
        h0_ = h_f;
        c0_ = c_f;
        return Tensor(all_h);
    }

    // 3D batch forward: input (B, T, D) → output (B, T, H). Each sample starts h/c=0.
    auto forward_3d(const Tensor& input, bool requires_grad) -> Tensor
    {
        const auto& shape = input.get_shape();
        const int B     = static_cast<int>(shape[0]);
        const int T_seq = static_cast<int>(shape[1]);
        const int D_in  = static_cast<int>(shape[2]);

        if (D_in != input_size_)
            throw std::invalid_argument("LSTMLayerImpl::forward_3d: input D=" +
                                        std::to_string(D_in) + " != input_size=" +
                                        std::to_string(input_size_));

        if (requires_grad)
        {
            batch_caches_.clear();
            batch_caches_.resize(B);
            for (auto& bc : batch_caches_) bc.reserve(T_seq);
        }

        nn::Tensor all_out = nn::Tensor::zeros(
            static_cast<nn::Index>(B),
            static_cast<nn::Index>(T_seq),
            static_cast<nn::Index>(hidden_size_));

        for (int b = 0; b < B; ++b)
        {
            nn::Tensor sample(static_cast<nn::Index>(T_seq),
                              static_cast<nn::Index>(D_in));
            for (int t = 0; t < T_seq; ++t)
                for (int d = 0; d < D_in; ++d)
                    sample.at(static_cast<nn::Index>(t), static_cast<nn::Index>(d)) =
                        input.at(static_cast<nn::Index>(b),
                                  static_cast<nn::Index>(t),
                                  static_cast<nn::Index>(d));


            nn::Tensor h0 = nn::Tensor::zeros(1, hidden_size_);
            nn::Tensor c0 = nn::Tensor::zeros(1, hidden_size_);
            auto [h_out, h_f, c_f] = _run_sequence(sample, h0, c0, requires_grad,
                                                    requires_grad ? &batch_caches_[b] : nullptr);
            (void)h_f; (void)c_f;

            for (int t = 0; t < T_seq; ++t)
                for (int hh = 0; hh < hidden_size_; ++hh)
                    all_out.at(static_cast<nn::Index>(b),
                                static_cast<nn::Index>(t),
                                static_cast<nn::Index>(hh)) = h_out.at(t, hh);

        }
        return Tensor(all_out);
    }
 
    // 3D backward: grad_output shape (B, T, H), returns grad_input shape (B, T, D).
    auto backward_3d(const Tensor& grad_output) -> Tensor
    {
        const auto& shape = grad_output.get_shape();
        const int B = static_cast<int>(shape[0]);
        const int T = static_cast<int>(shape[1]);
 
        if (static_cast<int>(batch_caches_.size()) != B)
            throw std::runtime_error("backward_3d: batch size mismatch with forward cache");
 
        nn::Tensor dW_accum = nn::Tensor::zeros(dW_.rows(), dW_.cols());
        nn::Tensor dU_accum = nn::Tensor::zeros(dU_.rows(), dU_.cols());
        nn::Tensor db_accum = nn::Tensor::zeros(db_.rows(), db_.cols());
 
        nn::Tensor dx_all = nn::Tensor::zeros(
            static_cast<nn::Index>(B),
            static_cast<nn::Index>(T),
            static_cast<nn::Index>(input_size_));
 
        for (int b = 0; b < B; ++b)
        {
            // Extract grad slice for sample b: (T, H)
            nn::Tensor grad_b(static_cast<nn::Index>(T),
                              static_cast<nn::Index>(hidden_size_));
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < hidden_size_; ++h)
                    grad_b.at(static_cast<nn::Index>(t), static_cast<nn::Index>(h)) =
                        grad_output.at(static_cast<nn::Index>(b),
                                        static_cast<nn::Index>(t),
                                        static_cast<nn::Index>(h));

 
            auto [dW_b, dU_b, db_b, dx_b] = _bptt_pure(batch_caches_[b], grad_b);
 
            for (nn::Index k = 0; k < static_cast<nn::Index>(dW_accum.size()); ++k)
                dW_accum.at(k) += dW_b.at(k);
            for (nn::Index k = 0; k < static_cast<nn::Index>(dU_accum.size()); ++k)
                dU_accum.at(k) += dU_b.at(k);
            for (nn::Index k = 0; k < static_cast<nn::Index>(db_accum.size()); ++k)
                db_accum.at(k) += db_b.at(k);
 
            // Write dx_b into dx_all[b, :, :]
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < input_size_; ++d)
                    dx_all.at(static_cast<nn::Index>(b),
                             static_cast<nn::Index>(t),
                             static_cast<nn::Index>(d)) =
                        dx_b.at(static_cast<nn::Index>(t), static_cast<nn::Index>(d));

        }
 
        W_.set_grad(dW_accum); U_.set_grad(dU_accum); b_.set_grad(db_accum);
        dW_ = dW_accum; dU_ = dU_accum; db_ = db_accum;
 
        return Tensor(dx_all);
    }
 
    private:
    // Pure BPTT: does not touch member dW_/dU_/db_. Returns (dW, dU, db, dx).
    auto _bptt_pure(const std::vector<LSTMStepCache>& step_cache,
                    const nn::Tensor& grad_output)
        -> std::tuple<nn::Tensor, nn::Tensor, nn::Tensor, nn::Tensor>
    {
        const int T = static_cast<int>(step_cache.size());

        nn::Tensor dW = nn::Tensor::zeros(dW_.rows(), dW_.cols());
        nn::Tensor dU = nn::Tensor::zeros(dU_.rows(), dU_.cols());
        nn::Tensor db = nn::Tensor::zeros(db_.rows(), db_.cols());
        nn::Tensor dx_all(static_cast<nn::Index>(T), static_cast<nn::Index>(input_size_));
        dx_all.set_zero();

        nn::Tensor dh_next = nn::Tensor::zeros(1, hidden_size_);
        nn::Tensor dc_next = nn::Tensor::zeros(1, hidden_size_);

        for (int t = T - 1; t >= 0; --t)
        {
            const auto& step = step_cache[static_cast<std::size_t>(t)];
            nn::Tensor dh = grad_output.row(static_cast<nn::Index>(t)).add(dh_next);

            nn::Tensor do_gate = dh * step.tanh_c;
            nn::Tensor dtanh_c = dh * step.o;
            nn::Tensor dc      = (dtanh_c * nn::activation::tanh_grad(step.tanh_c)).add(dc_next);

            nn::Tensor di_gate = dc * step.g;
            nn::Tensor df_gate = dc * step.c_prev;
            nn::Tensor dg_gate = dc * step.i;
            dc_next            = dc * step.f;

            nn::Tensor dpre_i = di_gate * nn::activation::sigmoid_grad(step.i);
            nn::Tensor dpre_f = df_gate * nn::activation::sigmoid_grad(step.f);
            nn::Tensor dpre_o = do_gate * nn::activation::sigmoid_grad(step.o);
            nn::Tensor dpre_g = dg_gate * nn::activation::tanh_grad(step.g);

            nn::Tensor dpre(1, 4 * hidden_size_);
            dpre.setBlock(0, 0,                   dpre_i);
            dpre.setBlock(0, hidden_size_,         dpre_f);
            dpre.setBlock(0, 2 * hidden_size_,     dpre_o);
            dpre.setBlock(0, 3 * hidden_size_,     dpre_g);

            nn::Tensor dW_step = dpre.transpose().matmul(step.x);
            nn::Tensor dU_step = dpre.transpose().matmul(step.h_prev);
            nn::Tensor db_step = dpre.transpose();

            for (nn::Index k = 0; k < static_cast<nn::Index>(dW.size()); ++k)
                dW.at(k) += dW_step.at(k);
            for (nn::Index k = 0; k < static_cast<nn::Index>(dU.size()); ++k)
                dU.at(k) += dU_step.at(k);
            for (nn::Index k = 0; k < static_cast<nn::Index>(db.size()); ++k)
                db.at(k) += db_step.at(k);

            dx_all.setBlock(static_cast<nn::Index>(t), 0, dpre.matmul(W_));
            dh_next = dpre.matmul(U_);
        }

        return {dW, dU, db, dx_all};
    }

    // Apply BPTT and write results to member grad storage.
    auto _bptt_apply(const std::vector<LSTMStepCache>& step_cache,
                     const nn::Tensor& grad_output,
                     bool /*accumulate*/) -> nn::Tensor
    {
        if (step_cache.empty())
            throw std::runtime_error(
                "LSTMLayerImpl::backward called before forward with requires_grad");

        auto [dW, dU, db, dx] = _bptt_pure(step_cache, grad_output);
        dW_ = dW;
        dU_ = dU;
        db_ = db;
        W_.set_grad(dW_);
        U_.set_grad(dU_);
        b_.set_grad(db_);
        return dx;
    }
};

// Backward-compat alias in the original namespace.
// Canonical alias: nn::LSTMLayer (from generated nn/layers/Layers.hpp).
namespace nn::models::lstm
{
using LSTMLayer = LSTMLayerImpl<nn::XTensorBackend>;
} // namespace nn::models::lstm

#endif // NN_LAYERS_LSTM_LSTMLAYER_HPP
