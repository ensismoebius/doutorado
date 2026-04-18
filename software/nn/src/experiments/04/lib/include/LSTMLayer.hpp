#pragma once

/**
 * @file LSTMLayer.hpp
 * @brief Single-layer LSTM cell with full BPTT (Backpropagation Through Time).
 *
 * Gate equations (per time step t):
 *   i_t = σ(W_i · x_t + U_i · h_{t-1} + b_i)   -- input gate
 *   f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)   -- forget gate
 *   o_t = σ(W_o · x_t + U_o · h_{t-1} + b_o)   -- output gate
 *   g_t = tanh(W_g · x_t + U_g · h_{t-1} + b_g) -- candidate cell
 *
 * State updates:
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
 *   h_t = o_t ⊙ tanh(c_t)
 *
 * Tensor shapes (all 2-D, row = batch, col = feature):
 *   x_t  : [batch × input_size]
 *   h_t  : [batch × hidden_size]
 *   c_t  : [batch × hidden_size]
 *   W_*  : [hidden_size × input_size]    (input weight)
 *   U_*  : [hidden_size × hidden_size]   (recurrent weight)
 *   b_*  : [hidden_size × 1]             (bias, broadcast over batch)
 *
 * Memory layout optimisation:
 *   All four input weight matrices are stacked into one big W [4·H × D]
 *   and all four recurrent weight matrices into one U [4·H × H].
 *   Gate slices are extracted via block() — avoids four separate matmuls.
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

namespace experiment04
{

// ---------------------------------------------------------------------------
// Helper: element-wise sigmoid applied to a Tensor (returns new Tensor)
// ---------------------------------------------------------------------------
inline auto sigmoid(const nn::Tensor& x) -> nn::Tensor
{
    // σ(x) = 1 / (1 + exp(-x))
    nn::Tensor result(x.rows(), x.cols());
    for (nn::Index i = 0; i < static_cast<nn::Index>(x.size()); ++i)
    {
        result.at(i) = 1.0f / (1.0f + std::exp(-x.at(i)));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Helper: element-wise tanh applied to a Tensor (returns new Tensor)
// ---------------------------------------------------------------------------
inline auto tanh_tensor(const nn::Tensor& x) -> nn::Tensor
{
    nn::Tensor result(x.rows(), x.cols());
    for (nn::Index i = 0; i < static_cast<nn::Index>(x.size()); ++i)
    {
        result.at(i) = std::tanh(x.at(i));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Helper: sigmoid derivative given sigmoid output σ  →  σ · (1 - σ)
// ---------------------------------------------------------------------------
inline auto sigmoid_grad(const nn::Tensor& sigmoid_out) -> nn::Tensor
{
    // d σ/d pre-act = σ · (1 - σ)
    nn::Tensor ones = nn::Tensor::ones(sigmoid_out.rows(), sigmoid_out.cols());
    nn::Tensor one_minus = ones - sigmoid_out;
    return sigmoid_out * one_minus;
}

// ---------------------------------------------------------------------------
// Helper: tanh derivative given tanh output  →  1 - tanh²
// ---------------------------------------------------------------------------
inline auto tanh_grad(const nn::Tensor& tanh_out) -> nn::Tensor
{
    // d tanh/d pre-act = 1 - tanh²
    nn::Tensor sq = tanh_out * tanh_out;
    nn::Tensor ones = nn::Tensor::ones(sq.rows(), sq.cols());
    return ones - sq;
}

// ---------------------------------------------------------------------------
// Cache for one time-step needed by BPTT
// ---------------------------------------------------------------------------
struct LSTMStepCache
{
    nn::Tensor x;      // [batch × D]  -- input at this step
    nn::Tensor h_prev; // [batch × H]  -- hidden state entering this step
    nn::Tensor c_prev; // [batch × H]  -- cell state entering this step
    nn::Tensor i;      // [batch × H]  -- input gate activation
    nn::Tensor f;      // [batch × H]  -- forget gate activation
    nn::Tensor o;      // [batch × H]  -- output gate activation
    nn::Tensor g;      // [batch × H]  -- candidate cell activation
    nn::Tensor c;      // [batch × H]  -- cell state at this step
    nn::Tensor tanh_c; // [batch × H]  -- tanh(c_t)
    nn::Tensor h;      // [batch × H]  -- hidden state at this step
};

// ---------------------------------------------------------------------------
// LSTMLayer: Module wrapping one LSTM layer
// ---------------------------------------------------------------------------
class LSTMLayer : public Module<nn::EigenTensorBackend>
{
   public:
    using Tensor = nn::Tensor;

    int input_size_;  ///< D: dimensionality of each input step
    int hidden_size_; ///< H: dimensionality of hidden/cell state

    // ------- Stacked weight matrices -------
    // W : [4H × D]  input weights for [i, f, o, g] gates stacked
    // U : [4H × H]  recurrent weights stacked the same way
    // b : [4H × 1]  biases stacked
    nn::Tensor W_; // [4H × D]
    nn::Tensor U_; // [4H × H]
    nn::Tensor b_; // [4H × 1]

    // ------- Gradient accumulators -------
    nn::Tensor dW_;
    nn::Tensor dU_;
    nn::Tensor db_;

    // ------- Parameter pointer list (for optimizer) -------
    std::vector<nn::Tensor*> param_ptrs_;

    // ------- Initial hidden/cell (trainable not required, zeroed) -------
    nn::Tensor h0_; // [1 × H]
    nn::Tensor c0_; // [1 × H]

    // ------- BPTT cache -------
    std::vector<LSTMStepCache> cache_; // one entry per time step
    bool requires_grad_ = false;

    // ------- Current (last-seen) batch size for zero-init of h0/c0 -------
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
        // Xavier-like initialisation for gate weights
        // limit = sqrt(6 / (fan_in + fan_out))
        auto xavier_fill = [](nn::Tensor& t, int fan_in, int fan_out, unsigned seed_offset)
        {
            float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
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

        // Set forget-gate bias to 1.0 (rows [H, 2H)) — common good default
        for (int r = hidden_size_; r < 2 * hidden_size_; ++r)
        {
            b_.at(static_cast<nn::Index>(r), 0) = 1.0f;
        }

        // Build param pointer list: W, U, b (h0/c0 kept non-trainable here)
        param_ptrs_ = {&W_, &U_, &b_};
    }

    // ------- Module interface -------

    /**
     * @brief Forward pass over the full sequence.
     *
     * @param input  Flattened sequence tensor [batch × (T * D)]
     *               The caller folds time into columns: columns 0..D-1 = t=0, etc.
     *               Or, more naturally, shape [T × D] for a single sample (batch=1).
     *
     * For experiment04 we accept shape [T × D] and treat T as sequence length,
     * D as input dimension, with implicit batch = 1.
     * Returns h_T : [1 × H] — the final hidden state (latent code for encoder,
     * or full sequence of hidden states [T × H] for decoder).
     *
     * @param requires_grad  Cache intermediate states for BPTT when true.
     */
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        const int T = static_cast<int>(input.rows()); // sequence length
        const int D = static_cast<int>(input.cols()); // input feature dim

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

        // Initialise h and c (batch=1 implicit)
        nn::Tensor h = h0_; // [1 × H]
        nn::Tensor c = c0_; // [1 × H]

        // We accumulate all hidden states into [T × H] for the decoder path
        nn::Tensor all_h(static_cast<nn::Index>(T), static_cast<nn::Index>(hidden_size_));

        for (int t = 0; t < T; ++t)
        {
            // x_t : [1 × D]
            Tensor x_t = input.row(static_cast<nn::Index>(t));

            // Pre-activations: [1 × 4H] = x_t · W^T + h · U^T + b^T
            // W_ : [4H × D]  →  W_^T : [D × 4H]  →  x_t · W_^T : [1 × 4H]
            Tensor pre =
                x_t.matmul(W_.transpose()).add(h.matmul(U_.transpose())).add(b_.transpose());

            // Slice gate pre-activations (each [1 × H])
            Tensor pre_i = pre.block(0, 0, 1, hidden_size_);
            Tensor pre_f = pre.block(0, 1 * hidden_size_, 1, hidden_size_);
            Tensor pre_o = pre.block(0, 2 * hidden_size_, 1, hidden_size_);
            Tensor pre_g = pre.block(0, 3 * hidden_size_, 1, hidden_size_);

            // Gate activations
            Tensor i_gate = sigmoid(pre_i);
            Tensor f_gate = sigmoid(pre_f);
            Tensor o_gate = sigmoid(pre_o);
            Tensor g_gate = tanh_tensor(pre_g);

            // Cell and hidden state updates
            // c_t = f ⊙ c_{t-1} + i ⊙ g
            Tensor c_new = (f_gate * c).add(i_gate * g_gate);
            Tensor tanh_c = tanh_tensor(c_new);
            // h_t = o ⊙ tanh(c_t)
            Tensor h_new = o_gate * tanh_c;

            if (requires_grad)
            {
                LSTMStepCache step{x_t, h, c, i_gate, f_gate, o_gate, g_gate, c_new, tanh_c, h_new};
                cache_.push_back(std::move(step));
            }

            // Write h_new into all_h row t
            for (nn::Index col = 0; col < static_cast<nn::Index>(hidden_size_); ++col)
            {
                all_h.at(static_cast<nn::Index>(t), col) = h_new.at(0, col);
            }

            h = h_new;
            c = c_new;
        }

        // Store final states for potential next-segment use
        h0_ = h;
        c0_ = c;

        return all_h; // [T × H]
    }

    /**
     * @brief BPTT backward pass.
     *
     * @param grad_output  Gradient of loss w.r.t. all hidden states, [T × H].
     *                     For encoder: only the last row is non-zero (dh_T).
     * @return Gradient w.r.t. input sequence [T × D].
     */
    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const int T = static_cast<int>(cache_.size());
        if (T == 0)
        {
            throw std::runtime_error(
                "LSTMLayer::backward called before forward with requires_grad");
        }

        // Zero gradient accumulators
        dW_.set_zero();
        dU_.set_zero();
        db_.set_zero();

        nn::Tensor dx_all(static_cast<nn::Index>(T), static_cast<nn::Index>(input_size_));
        dx_all.set_zero();

        // Carry-back gradients
        nn::Tensor dh_next = nn::Tensor::zeros(1, hidden_size_);
        nn::Tensor dc_next = nn::Tensor::zeros(1, hidden_size_);

        // Iterate time steps backwards (BPTT unroll)
        for (int t = T - 1; t >= 0; --t)
        {
            const auto& step = cache_[static_cast<size_t>(t)];

            // dL/dh_t = grad from loss at this step + carried gradient
            nn::Tensor dh = grad_output.row(static_cast<nn::Index>(t)).add(dh_next);

            // dL/do_t = dh ⊙ tanh(c_t)
            nn::Tensor do_gate = dh * step.tanh_c;

            // dL/d tanh(c_t) = dh ⊙ o_t
            nn::Tensor dtanh_c = dh * step.o;

            // dL/dc_t = dtanh_c ⊙ (1 - tanh²(c_t)) + dc_next
            nn::Tensor dc = (dtanh_c * tanh_grad(step.tanh_c)).add(dc_next);

            // dL/di = dc ⊙ g_t
            nn::Tensor di_gate = dc * step.g;

            // dL/df = dc ⊙ c_{t-1}
            nn::Tensor df_gate = dc * step.c_prev;

            // dL/dg = dc ⊙ i_t
            nn::Tensor dg_gate = dc * step.i;

            // dL/dc_{t-1} = dc ⊙ f_t
            dc_next = dc * step.f;

            // Pre-activation gradients (chain through gate nonlinearity)
            nn::Tensor dpre_i = di_gate * sigmoid_grad(step.i);
            nn::Tensor dpre_f = df_gate * sigmoid_grad(step.f);
            nn::Tensor dpre_o = do_gate * sigmoid_grad(step.o);
            nn::Tensor dpre_g = dg_gate * tanh_grad(step.g);

            // Assemble dpre : [1 × 4H]
            nn::Tensor dpre(1, 4 * hidden_size_);
            for (nn::Index col = 0; col < static_cast<nn::Index>(hidden_size_); ++col)
            {
                dpre.at(0, 0 * hidden_size_ + col) = dpre_i.at(0, col);
                dpre.at(0, 1 * hidden_size_ + col) = dpre_f.at(0, col);
                dpre.at(0, 2 * hidden_size_ + col) = dpre_o.at(0, col);
                dpre.at(0, 3 * hidden_size_ + col) = dpre_g.at(0, col);
            }

            // Accumulate weight gradients
            // dW += dpre^T · x_t        (shape: [4H × D])
            nn::Tensor dW_t = dpre.transpose().matmul(step.x);
            dW_.add_inplace(dW_t);

            // dU += dpre^T · h_{t-1}   (shape: [4H × H])
            nn::Tensor dU_t = dpre.transpose().matmul(step.h_prev);
            dU_.add_inplace(dU_t);

            // db += dpre^T (sum over batch, but batch=1 here)  (shape: [4H × 1])
            nn::Tensor db_t = dpre.transpose();
            db_.add_inplace(db_t);

            // dL/dx_t = dpre · W   shape: [1 × D]
            nn::Tensor dx_t = dpre.matmul(W_);
            for (nn::Index col = 0; col < static_cast<nn::Index>(input_size_); ++col)
            {
                dx_all.at(static_cast<nn::Index>(t), col) = dx_t.at(0, col);
            }

            // Carry hidden gradient backwards through h_{t-1}
            // dh_{t-1} = dpre · U
            dh_next = dpre.matmul(U_);
        }

        // Store gradients for optimizer
        W_.set_grad(dW_);
        U_.set_grad(dU_);
        b_.set_grad(db_);

        return dx_all; // [T × D]
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

} // namespace experiment04
