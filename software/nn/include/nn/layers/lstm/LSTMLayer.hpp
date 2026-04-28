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
 * Batch support:
 *   forward_batch({s1, s2, ...}) — runs per-sample forward, stores per-sample
 *   caches in batch_caches_. Follow with backward_batch(grad_outputs) to
 *   accumulate gradients across the batch.
 *   BatchEquivalence: backward_batch({g, g}) == 2× backward(g) for same sample.
 */

#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::models::lstm
{

// ---------------------------------------------------------------------------
// Activation helpers (inline — used by LSTMLayer and LSTMAutoencoder)
// ---------------------------------------------------------------------------

inline auto sigmoid(const nn::Tensor& x) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(x.rows(), x.cols());
    return ones.divide(ones + (x * -1.0f).exp());
}

inline auto tanh_tensor(const nn::Tensor& x) -> nn::Tensor
{
    return (sigmoid(x * 2.0f) * 2.0f) - 1.0f;
}

inline auto sigmoid_grad(const nn::Tensor& sigmoid_out) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(sigmoid_out.rows(), sigmoid_out.cols());
    return sigmoid_out * (ones - sigmoid_out);
}

inline auto tanh_grad(const nn::Tensor& tanh_out) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(tanh_out.rows(), tanh_out.cols());
    return ones - (tanh_out * tanh_out);
}

// ---------------------------------------------------------------------------
// Per-timestep cache for BPTT
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
// LSTMLayer
// ---------------------------------------------------------------------------

/**
 * @class LSTMLayer
 * @brief One stacked LSTM layer.  Weights:
 *   W_ : (4H, D) — input-to-hidden, gates stacked [i|f|o|g]
 *   U_ : (4H, H) — hidden-to-hidden
 *   b_ : (4H, 1) — bias (forget-gate bias initialised to 1)
 */
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

    std::vector<LSTMStepCache>              cache_;       // single-sample BPTT cache
    std::vector<std::vector<LSTMStepCache>> batch_caches_; // multi-sample caches
    bool requires_grad_ = false;

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
        const int T = static_cast<int>(input.rows());
        const int D = static_cast<int>(input.cols());

        if (D != input_size_)
            throw std::invalid_argument("LSTMLayer::forward: input cols=" +
                                        std::to_string(D) +
                                        " != input_size=" + std::to_string(input_size_));

        if (requires_grad)
        {
            cache_.clear();
            cache_.reserve(static_cast<std::size_t>(T));
        }

        nn::Tensor h = h0_;
        nn::Tensor c = c0_;
        nn::Tensor all_h(static_cast<nn::Index>(T), static_cast<nn::Index>(hidden_size_));

        for (int t = 0; t < T; ++t)
        {
            Tensor x_t = input.row(static_cast<nn::Index>(t));
            Tensor pre  = x_t.matmul(W_.transpose())
                              .add(h.matmul(U_.transpose()))
                              .add(b_.transpose());

            Tensor i_gate = sigmoid(pre.block(0, 0,                   1, hidden_size_));
            Tensor f_gate = sigmoid(pre.block(0, 1 * hidden_size_,    1, hidden_size_));
            Tensor o_gate = sigmoid(pre.block(0, 2 * hidden_size_,    1, hidden_size_));
            Tensor g_gate = tanh_tensor(pre.block(0, 3 * hidden_size_, 1, hidden_size_));

            Tensor c_new  = (f_gate * c).add(i_gate * g_gate);
            Tensor tanh_c = tanh_tensor(c_new);
            Tensor h_new  = o_gate * tanh_c;

            if (requires_grad)
                cache_.push_back({x_t, h, c, i_gate, f_gate, o_gate, g_gate, c_new, tanh_c, h_new});

            all_h.setBlock(static_cast<nn::Index>(t), 0, h_new);
            h = h_new;
            c = c_new;
        }

        h0_ = h;
        c0_ = c;
        return all_h;
    }

    // --- single-sample backward ---

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        return _bptt_apply(cache_, grad_output, /*accumulate=*/false);
    }

    // --- batch forward: run B independent sequences, store per-sample caches ---

    auto forward_batch(const std::vector<Tensor>& batch_inputs,
                       bool requires_grad = true) -> std::vector<Tensor>
    {
        batch_caches_.clear();
        batch_caches_.reserve(batch_inputs.size());

        std::vector<Tensor> outputs;
        outputs.reserve(batch_inputs.size());

        for (const auto& sample : batch_inputs)
        {
            reset_state();
            Tensor out = forward(sample, requires_grad);
            outputs.push_back(out);
            if (requires_grad)
                batch_caches_.push_back(cache_);
        }

        return outputs;
    }

    // --- batch backward: accumulate gradients across all samples in the batch ---

    auto backward_batch(const std::vector<Tensor>& grad_outputs) -> std::vector<Tensor>
    {
        if (grad_outputs.size() != batch_caches_.size())
            throw std::runtime_error(
                "backward_batch: grad_outputs.size() != batch_caches_.size()");

        // Zero accumulator tensors
        nn::Tensor dW_accum = nn::Tensor::zeros(dW_.rows(), dW_.cols());
        nn::Tensor dU_accum = nn::Tensor::zeros(dU_.rows(), dU_.cols());
        nn::Tensor db_accum = nn::Tensor::zeros(db_.rows(), db_.cols());

        std::vector<Tensor> dx_all_outputs;
        dx_all_outputs.reserve(grad_outputs.size());

        for (std::size_t b = 0; b < grad_outputs.size(); ++b)
        {
            // Run BPTT for this sample without touching member grad storage
            auto [dW_b, dU_b, db_b, dx_b] =
                _bptt_pure(batch_caches_[b], grad_outputs[b]);

            // Accumulate
            for (nn::Index k = 0; k < static_cast<nn::Index>(dW_accum.size()); ++k)
                dW_accum.at(k) += dW_b.at(k);
            for (nn::Index k = 0; k < static_cast<nn::Index>(dU_accum.size()); ++k)
                dU_accum.at(k) += dU_b.at(k);
            for (nn::Index k = 0; k < static_cast<nn::Index>(db_accum.size()); ++k)
                db_accum.at(k) += db_b.at(k);

            dx_all_outputs.push_back(dx_b);
        }

        W_.set_grad(dW_accum);
        U_.set_grad(dU_accum);
        b_.set_grad(db_accum);

        dW_ = dW_accum;
        dU_ = dU_accum;
        db_ = db_accum;

        return dx_all_outputs;
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
    // Pure BPTT: does not touch member dW_/dU_/db_. Returns (dW, dU, db, dx).
    auto _bptt_pure(const std::vector<LSTMStepCache>& step_cache,
                    const Tensor& grad_output)
        -> std::tuple<Tensor, Tensor, Tensor, Tensor>
    {
        const int T = static_cast<int>(step_cache.size());

        Tensor dW = Tensor::zeros(dW_.rows(), dW_.cols());
        Tensor dU = Tensor::zeros(dU_.rows(), dU_.cols());
        Tensor db = Tensor::zeros(db_.rows(), db_.cols());
        Tensor dx_all(static_cast<nn::Index>(T), static_cast<nn::Index>(input_size_));
        dx_all.set_zero();

        Tensor dh_next = Tensor::zeros(1, hidden_size_);
        Tensor dc_next = Tensor::zeros(1, hidden_size_);

        for (int t = T - 1; t >= 0; --t)
        {
            const auto& step = step_cache[static_cast<std::size_t>(t)];
            Tensor dh = grad_output.row(static_cast<nn::Index>(t)).add(dh_next);

            Tensor do_gate = dh * step.tanh_c;
            Tensor dtanh_c = dh * step.o;
            Tensor dc      = (dtanh_c * tanh_grad(step.tanh_c)).add(dc_next);

            Tensor di_gate = dc * step.g;
            Tensor df_gate = dc * step.c_prev;
            Tensor dg_gate = dc * step.i;
            dc_next        = dc * step.f;

            Tensor dpre_i = di_gate * sigmoid_grad(step.i);
            Tensor dpre_f = df_gate * sigmoid_grad(step.f);
            Tensor dpre_o = do_gate * sigmoid_grad(step.o);
            Tensor dpre_g = dg_gate * tanh_grad(step.g);

            Tensor dpre(1, 4 * hidden_size_);
            dpre.setBlock(0, 0,                   dpre_i);
            dpre.setBlock(0, hidden_size_,         dpre_f);
            dpre.setBlock(0, 2 * hidden_size_,     dpre_o);
            dpre.setBlock(0, 3 * hidden_size_,     dpre_g);

            Tensor dW_step = dpre.transpose().matmul(step.x);
            Tensor dU_step = dpre.transpose().matmul(step.h_prev);
            Tensor db_step = dpre.transpose();

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
                     const Tensor& grad_output,
                     bool /*accumulate*/) -> Tensor
    {
        if (step_cache.empty())
            throw std::runtime_error(
                "LSTMLayer::backward called before forward with requires_grad");

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

} // namespace nn::models::lstm

#endif // NN_LAYERS_LSTM_LSTMLAYER_HPP
