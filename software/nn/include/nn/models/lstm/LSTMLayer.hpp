#ifndef NN_MODELS_LSTM_LSTMLAYER_HPP
#define NN_MODELS_LSTM_LSTMLAYER_HPP

/**
 * @file include/nn/models/lstm/LSTMLayer.hpp
 * @brief Single-layer LSTM cell with full BPTT (Backpropagation Through Time).
 *
 * Gate equations (per time step t):
 *   i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)   // input gate
 *   f_t = sigma(W_f x_t + U_f h_{t-1} + b_f)   // forget gate
 *   o_t = sigma(W_o x_t + U_o h_{t-1} + b_o)   // output gate
 *   g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)    // cell gate
 *
 * Parameters are stored as stacked matrices [i|f|o|g] to reduce kernel
 * dispatch overhead.  Hidden state and cell state are persisted across
 * calls to @c forward so that sequences can be processed in chunks; call
 * @c reset_state() between independent sequences.
 */

#include <random>
#include <stdexcept>
#include <vector>

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::models::lstm
{

// ---------------------------------------------------------------------------
// Activation helpers (inline — used by LSTMLayer and LSTMAutoencoder)
// ---------------------------------------------------------------------------

/// Element-wise sigmoid: 1 / (1 + exp(-x))
inline auto sigmoid(const nn::Tensor& x) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(x.rows(), x.cols());
    return ones.divide(ones + (x * -1.0f).exp());
}

/// Element-wise tanh via sigmoid identity: tanh(x) = 2*sigmoid(2x) - 1
inline auto tanh_tensor(const nn::Tensor& x) -> nn::Tensor
{
    return (sigmoid(x * 2.0f) * 2.0f) - 1.0f;
}

/// Gradient of sigmoid given its output: s * (1 - s)
inline auto sigmoid_grad(const nn::Tensor& sigmoid_out) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(sigmoid_out.rows(), sigmoid_out.cols());
    return sigmoid_out * (ones - sigmoid_out);
}

/// Gradient of tanh given its output: 1 - tanh(x)^2
inline auto tanh_grad(const nn::Tensor& tanh_out) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(tanh_out.rows(), tanh_out.cols());
    return ones - (tanh_out * tanh_out);
}

// ---------------------------------------------------------------------------
// Per-timestep cache for BPTT
// ---------------------------------------------------------------------------

/// Cached activations for a single LSTM time step, required during backward.
struct LSTMStepCache
{
    nn::Tensor x;      ///< input at this step
    nn::Tensor h_prev; ///< hidden state from previous step
    nn::Tensor c_prev; ///< cell state from previous step
    nn::Tensor i;      ///< input-gate activation
    nn::Tensor f;      ///< forget-gate activation
    nn::Tensor o;      ///< output-gate activation
    nn::Tensor g;      ///< cell-gate (tanh) activation
    nn::Tensor c;      ///< new cell state
    nn::Tensor tanh_c; ///< tanh of new cell state
    nn::Tensor h;      ///< new hidden state
};

// ---------------------------------------------------------------------------
// LSTMLayer
// ---------------------------------------------------------------------------

/**
 * @class LSTMLayer
 * @brief One stacked LSTM layer.  All weights, gradients, and hidden state
 *        are owned by this object.
 *
 * Weight layout (row-major):
 *   W_  : (4*H, D) — input-to-hidden weights, gates stacked [i|f|o|g]
 *   U_  : (4*H, H) — hidden-to-hidden weights, same stacking
 *   b_  : (4*H, 1) — bias (forget-gate bias initialised to 1)
 */
class LSTMLayer : public Module<nn::EigenTensorBackend>
{
   public:
    using Tensor = nn::Tensor;

    int input_size_;  ///< D: input feature dimension
    int hidden_size_; ///< H: hidden state dimension

    nn::Tensor W_; ///< input-weight matrix  (4H × D)
    nn::Tensor U_; ///< hidden-weight matrix (4H × H)
    nn::Tensor b_; ///< bias                 (4H × 1)

    nn::Tensor dW_; ///< gradient accumulator for W_
    nn::Tensor dU_; ///< gradient accumulator for U_
    nn::Tensor db_; ///< gradient accumulator for b_

    std::vector<nn::Tensor*> param_ptrs_; ///< flat param list for optimizer

    nn::Tensor h0_; ///< persistent hidden state (reset via reset_state())
    nn::Tensor c0_; ///< persistent cell   state (reset via reset_state())

    std::vector<LSTMStepCache> cache_; ///< per-step cache for BPTT
    bool requires_grad_ = false;
    int last_batch_ = 0;

    /// Construct with given input and hidden dimensions; initialises weights
    /// with N(0, 0.05) and sets the forget-gate bias to 1 for training stability.
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
            {
                t.at(k) = dist(rng);
            }
        };

        normal_fill(W_, 0u);
        normal_fill(U_, 1u);
        b_.set_zero();
        h0_.set_zero();
        c0_.set_zero();
        dW_.set_zero();
        dU_.set_zero();
        db_.set_zero();

        // Initialise forget-gate bias to 1 to help learning long sequences.
        for (int r = hidden_size_; r < 2 * hidden_size_; ++r)
        {
            b_.at(static_cast<nn::Index>(r), 0) = 1.0f;
        }

        param_ptrs_ = {&W_, &U_, &b_};
    }

    /// Forward pass over a full sequence.
    /// @param input  (T × D) input tensor
    /// @param requires_grad  if true, caches activations for backward()
    /// @return (T × H) tensor of per-step hidden states
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        const int T = static_cast<int>(input.rows());
        const int D = static_cast<int>(input.cols());

        if (D != input_size_)
        {
            throw std::invalid_argument("LSTMLayer::forward: input cols=" + std::to_string(D) +
                                        " != input_size=" + std::to_string(input_size_));
        }

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
            // Combined projection: [i|f|o|g] = W x + U h + b
            Tensor pre =
                x_t.matmul(W_.transpose()).add(h.matmul(U_.transpose())).add(b_.transpose());

            Tensor i_gate = sigmoid(pre.block(0, 0, 1, hidden_size_));
            Tensor f_gate = sigmoid(pre.block(0, 1 * hidden_size_, 1, hidden_size_));
            Tensor o_gate = sigmoid(pre.block(0, 2 * hidden_size_, 1, hidden_size_));
            Tensor g_gate = tanh_tensor(pre.block(0, 3 * hidden_size_, 1, hidden_size_));

            Tensor c_new = (f_gate * c).add(i_gate * g_gate);
            Tensor tanh_c = tanh_tensor(c_new);
            Tensor h_new = o_gate * tanh_c;

            if (requires_grad)
            {
                cache_.push_back({x_t, h, c, i_gate, f_gate, o_gate, g_gate, c_new, tanh_c, h_new});
            }

            all_h.setBlock(static_cast<nn::Index>(t), 0, h_new);
            h = h_new;
            c = c_new;
        }

        // Persist state for the next call (stateful across chunks).
        h0_ = h;
        c0_ = c;
        return all_h;
    }

    /// BPTT backward pass.
    /// @param grad_output  (T × H) gradient of loss w.r.t. each hidden output
    /// @return (T × D) gradient w.r.t. input
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
            const auto& step = cache_[static_cast<std::size_t>(t)];
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

            // Re-assemble into the stacked gate gradient.
            nn::Tensor dpre(1, 4 * hidden_size_);
            dpre.setBlock(0, 0, dpre_i);
            dpre.setBlock(0, hidden_size_, dpre_f);
            dpre.setBlock(0, 2 * hidden_size_, dpre_o);
            dpre.setBlock(0, 3 * hidden_size_, dpre_g);

            dW_.add_inplace(dpre.transpose().matmul(step.x));
            dU_.add_inplace(dpre.transpose().matmul(step.h_prev));
            db_.add_inplace(dpre.transpose());

            dx_all.setBlock(static_cast<nn::Index>(t), 0, dpre.matmul(W_));
            dh_next = dpre.matmul(U_);
        }

        W_.set_grad(dW_);
        U_.set_grad(dU_);
        b_.set_grad(db_);

        return dx_all;
    }

    /// Reset hidden and cell state to zero (call between independent sequences).
    void reset_state() override
    {
        h0_.set_zero();
        c0_.set_zero();
        cache_.clear();
    }

    /// Returns {W_, U_, b_} as a flat span.
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

} // namespace nn::models::lstm

#endif // NN_MODELS_LSTM_LSTMLAYER_HPP
