#ifndef NN_LAYERS_LSTM_LSTMLAYER_HPP
#define NN_LAYERS_LSTM_LSTMLAYER_HPP

/**
 * @file include/nn/layers/lstm/LSTMLayer.hpp
 * @brief Single-layer LSTM cell with full BPTT and batch gradient accumulation.
 *
 * Gate equations per time step t — verified against [1, 2]:
 *   pre_t  = x_t W^T + H_{t-1} U^T + b^T          // (1, 4H)
 *   i_t    = sigma(pre_t[0:H])                      // input gate
 *   f_t    = sigma(pre_t[H:2H])                     // forget gate (bias init=1 [3])
 *   o_t    = sigma(pre_t[2H:3H])                    // output gate
 *   g_t    = tanh(pre_t[3H:4H])                     // cell candidate
 *   C_t    = f_t ⊙ C_{t-1} + i_t ⊙ g_t            // cell state
 *   H_t    = o_t ⊙ tanh(C_t)                       // hidden state
 *
 * Gate ordering [i|f|o|g] follows Greff et al. (2015) [2] convention.
 * Weights stored as stacked matrices in the same order.
 * Hidden and cell state persist across forward() calls; call reset_state()
 * between independent sequences.
 *
 * References:
 *   [1] Hochreiter & Schmidhuber, Neural Computation 9(8), 1997.
 *       https://doi.org/10.1162/neco.1997.9.8.1735
 *   [2] Greff et al., IEEE TNNLS 28(10), 2017. arXiv:1503.04069
 *   [3] Jozefowicz et al., ICML 2015 (forget-gate bias initialisation).
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
 *   - All forward/backward computation uses Tensor = TensorImpl<Backend>.
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
// Per-timestep cache for BPTT.
//
// Fields chosen to be exactly what _bptt_pure needs — no extras.
// Verified against Hochreiter & Schmidhuber (1997) and Greff et al. (2015):
//   dL/do   needs tanh_c  (= tanh(C_t))
//   dL/dC_t needs o, tanh_c
//   dL/df   needs c_prev  (= C_{t-1})
//   dW,dU   need x, h_prev
//   dx,dh   need W, U (member tensors, not cached here)
// C_t and H_t are NOT needed in the backward pass and are intentionally omitted.
// Uses Tensor = TensorImpl<Backend> so all ops stay on the active backend.
// ---------------------------------------------------------------------------

template <typename Backend>
struct LSTMStepCacheT
{
    using Tensor = nn::TensorImpl<Backend>;
    Tensor x;       // x_t
    Tensor h_prev;  // H_{t-1}
    Tensor c_prev;  // C_{t-1}
    Tensor i;       // σ(pre_i)
    Tensor f;       // σ(pre_f)
    Tensor o;       // σ(pre_o)
    Tensor g;       // tanh(pre_g)
    Tensor tanh_c;  // tanh(C_t) — needed for dL/do and dL/dC
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
 * All intermediate forward/backward computations use Tensor = TensorImpl<Backend>.
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

    // State: uses Tensor (backend) — not optimizer params.
    Tensor h0_;
    Tensor c0_;

    std::vector<LSTMStepCacheT<Backend>>              cache_;
    std::vector<std::vector<LSTMStepCacheT<Backend>>> batch_caches_;
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

        // Forget-gate bias = 1 for training stability [3].
        for (int r = hidden_size_; r < 2 * hidden_size_; ++r)
            b_.at(static_cast<nn::Index>(r), 0) = 1.0f;

        param_ptrs_ = {&W_, &U_, &b_};
    }

    // --- forward: dispatches on input rank ---

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        if (input.get_shape().size() == 3)
            return forward_3d(input, requires_grad);
        return forward_2d(input, requires_grad);
    }

    // --- backward: dispatches on grad rank ---

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (grad_output.get_shape().size() == 3)
            return backward_3d(grad_output);
        return _bptt_apply(cache_, grad_output);
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

    // -----------------------------------------------------------------------
    // Core LSTM gate loop over one (T, D) sequence — all ops on Tensor (backend).
    // Returns {all_h (T,H), final_h (1,H), final_c (1,H)}.
    // Appends to *cache_out when requires_grad && cache_out != nullptr.
    // -----------------------------------------------------------------------
    auto _run_sequence(const Tensor& seq,
                       Tensor h, Tensor c,
                       bool requires_grad,
                       std::vector<LSTMStepCacheT<Backend>>* cache_out)
        -> std::tuple<Tensor, Tensor, Tensor>
    {
        const int T_seq = static_cast<int>(seq.rows());
        Tensor all_h(static_cast<nn::Index>(T_seq),
                     static_cast<nn::Index>(hidden_size_));

        for (int t = 0; t < T_seq; ++t)
        {
            Tensor x_t = seq.row(static_cast<nn::Index>(t));
            Tensor pre = x_t.matmul_transposed(W_)
                            .add(h.matmul_transposed(U_))
                            .add(b_.transpose());

            Tensor i_g = nn::activation::sigmoid(pre.block(0, 0,              1, hidden_size_));
            Tensor f_g = nn::activation::sigmoid(pre.block(0, 1*hidden_size_, 1, hidden_size_));
            Tensor o_g = nn::activation::sigmoid(pre.block(0, 2*hidden_size_, 1, hidden_size_));
            Tensor g_g = nn::activation::tanh(   pre.block(0, 3*hidden_size_, 1, hidden_size_));

            Tensor c_new = (f_g * c).add(i_g * g_g);
            Tensor tc    = nn::activation::tanh(c_new);
            Tensor h_new = o_g * tc;

            if (requires_grad && cache_out)
                cache_out->push_back({x_t, h, c, i_g, f_g, o_g, g_g, tc});

            all_h.setBlock(static_cast<nn::Index>(t), 0, h_new);
            h = h_new;
            c = c_new;
        }
        return {all_h, h, c};
    }

    // 2D forward: input (T, D) → output (T, H). Persists h0_/c0_.
    // Input passed directly — no copy to nn::Tensor.
    auto forward_2d(const Tensor& input, bool requires_grad) -> Tensor
    {
        const int D_in = static_cast<int>(input.cols());
        if (D_in != input_size_)
            throw std::invalid_argument("LSTMLayerImpl::forward: input cols=" +
                                        std::to_string(D_in) + " != input_size=" +
                                        std::to_string(input_size_));

        const int T_seq = static_cast<int>(input.rows());
        if (requires_grad) { cache_.clear(); cache_.reserve(T_seq); }

        auto [all_h, h_f, c_f] = _run_sequence(input, h0_, c0_, requires_grad,
                                                requires_grad ? &cache_ : nullptr);
        h0_ = h_f;
        c0_ = c_f;
        return all_h;
    }

    // 3D batch forward: input (B, T, D) → output (B, T, H).
    // Each sample starts h/c=0. Uses backend slice_batch / set_batch_slice.
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

        Tensor all_out = Tensor::zeros(static_cast<nn::Index>(B),
                                       static_cast<nn::Index>(T_seq),
                                       static_cast<nn::Index>(hidden_size_));

        for (int b = 0; b < B; ++b)
        {
            // Backend slice — avoids element-by-element copy.
            Tensor sample = input.slice_batch(static_cast<nn::Index>(b));

            Tensor h0 = Tensor::zeros(1, static_cast<nn::Index>(hidden_size_));
            Tensor c0 = Tensor::zeros(1, static_cast<nn::Index>(hidden_size_));
            auto [h_out, h_f, c_f] = _run_sequence(sample, h0, c0, requires_grad,
                                                    requires_grad ? &batch_caches_[b] : nullptr);
            (void)h_f; (void)c_f;

            // Backend slice assign — avoids element-by-element write.
            all_out.set_batch_slice(static_cast<nn::Index>(b), h_out);
        }
        return all_out;
    }

    // 3D backward: grad_output (B, T, H) → grad_input (B, T, D).
    // Uses backend slice ops and add_inplace for accumulation.
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

        Tensor dx_all = Tensor::zeros(static_cast<nn::Index>(B),
                                      static_cast<nn::Index>(T),
                                      static_cast<nn::Index>(input_size_));

        for (int b = 0; b < B; ++b)
        {
            // Backend slice — avoids element-by-element copy.
            Tensor grad_b = grad_output.slice_batch(static_cast<nn::Index>(b));

            auto [dW_b, dU_b, db_b, dx_b] = _bptt_pure(batch_caches_[b], grad_b);

            // add_inplace — uses backend vectorised add.
            dW_accum.add_inplace(dW_b);
            dU_accum.add_inplace(dU_b);
            db_accum.add_inplace(db_b);

            // Backend slice assign.
            dx_all.set_batch_slice(static_cast<nn::Index>(b), dx_b);
        }

        W_.set_grad(dW_accum); U_.set_grad(dU_accum); b_.set_grad(db_accum);
        dW_ = dW_accum; dU_ = dU_accum; db_ = db_accum;

        return dx_all;
    }

    // -----------------------------------------------------------------------
    // Pure BPTT: does not touch member dW_/dU_/db_.
    // Returns (dW, dU, db, dx) — all Tensor (backend).
    // Weight grads remain nn::Tensor-compatible since Tensor = TensorImpl<Backend>
    // and W_/U_/b_ are nn::Tensor (same underlying type for XTensorBackend).
    // -----------------------------------------------------------------------
    auto _bptt_pure(const std::vector<LSTMStepCacheT<Backend>>& step_cache,
                    const Tensor& grad_output)
        -> std::tuple<nn::Tensor, nn::Tensor, nn::Tensor, Tensor>
    {
        const int T = static_cast<int>(step_cache.size());

        nn::Tensor dW = nn::Tensor::zeros(dW_.rows(), dW_.cols());
        nn::Tensor dU = nn::Tensor::zeros(dU_.rows(), dU_.cols());
        nn::Tensor db = nn::Tensor::zeros(db_.rows(), db_.cols());
        Tensor dx_all = Tensor::zeros(static_cast<nn::Index>(T),
                                      static_cast<nn::Index>(input_size_));

        Tensor dh_next = Tensor::zeros(1, static_cast<nn::Index>(hidden_size_));
        Tensor dc_next = Tensor::zeros(1, static_cast<nn::Index>(hidden_size_));

        for (int t = T - 1; t >= 0; --t)
        {
            const auto& step = step_cache[static_cast<std::size_t>(t)];
            Tensor dh = grad_output.row(static_cast<nn::Index>(t)).add(dh_next);

            Tensor do_gate = dh * step.tanh_c;
            Tensor dtanh_c = dh * step.o;
            Tensor dc      = (dtanh_c * nn::activation::tanh_grad(step.tanh_c)).add(dc_next);

            Tensor di_gate = dc * step.g;
            Tensor df_gate = dc * step.c_prev;
            Tensor dg_gate = dc * step.i;
            dc_next        = dc * step.f;

            Tensor dpre_i = di_gate * nn::activation::sigmoid_grad(step.i);
            Tensor dpre_f = df_gate * nn::activation::sigmoid_grad(step.f);
            Tensor dpre_o = do_gate * nn::activation::sigmoid_grad(step.o);
            Tensor dpre_g = dg_gate * nn::activation::tanh_grad(step.g);

            Tensor dpre(1, 4 * static_cast<nn::Index>(hidden_size_));
            dpre.setBlock(0, 0,                   dpre_i);
            dpre.setBlock(0, hidden_size_,         dpre_f);
            dpre.setBlock(0, 2 * hidden_size_,     dpre_o);
            dpre.setBlock(0, 3 * hidden_size_,     dpre_g);

            // Weight grads: use add_inplace — backend vectorised.
            dW.add_inplace(dpre.transpose().matmul(step.x));
            dU.add_inplace(dpre.transpose().matmul(step.h_prev));
            db.add_inplace(dpre.transpose());

            dx_all.setBlock(static_cast<nn::Index>(t), 0, dpre.matmul(W_));
            dh_next = dpre.matmul(U_);
        }

        return {dW, dU, db, dx_all};
    }

    // Apply BPTT and write results to member grad storage.
    auto _bptt_apply(const std::vector<LSTMStepCacheT<Backend>>& step_cache,
                     const Tensor& grad_output) -> Tensor
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
