// The Temporal Deep Residual SNN classifier used by ThesisClassifiers.cpp's
// "dsnn" classifier type. Split into its own header (not a .cpp) because
// every other Module<Backend> type in this codebase is header-only -- see
// include/layers/spiking/LifBPTT.hpp, ThresholdDependentBatchNorm.hpp, etc.
// -- and because ThesisClassifiersInternal.hpp's with_classifier<Fn> template
// constructs it directly, so its definition must be visible wherever that
// template is instantiated regardless.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "Backend.hpp"
#include "initializers/kaiming_snn.hpp"
#include "layers/dense/Linear.hpp"
#include "layers/spiking/LifBPTT.hpp"
#include "layers/spiking/ThresholdDependentBatchNorm.hpp"
#include "tensor/Tensor.hpp"

namespace thesis
{

// Number of LIF simulation steps the static feature vector is unrolled over.
// (Tracked as audit item M-4: use a genuine temporal SNN, not a 1-step net.)
constexpr int kSnnTimeSteps = 16;

// Firing threshold of the LIF layers (LifBPTT default). tdBN rescales the
// pre-spike current to N(0,(α·V_th)²), so it must use the same V_th.
constexpr float kSnnVth = 1.0f;

// Temporal Deep Residual Spiking Neural Network for Thesis.
//
// Unlike a single-step thresholding net, this unrolls genuine LIF dynamics over
// kSnnTimeSteps. The static feature vector is rate-encoded by constant-current
// injection: each of the T steps receives the same input row (time-major layout
// (T*B, F) expected by LifBPTT). Spikes integrate over time through LifBPTT
// neurons; the readout is the spike-count (mean firing rate over T) of the final
// linear layer, giving class logits. Gradients flow through real BPTT.
//
// Residual structure: each hidden block (Linear→(tdBN)→LIF, all hidden_dim wide)
// is wrapped in an identity skip connection h_out = block(h) + h. Skips are
// parameter-free, so they add no state to serialization. The input stage
// (fc_in: input_dim→hidden_dim) and output stage (fc_out) carry no skip because
// their widths differ from hidden_dim. This matches the deep-residual-SNN design
// of Zheng et al. (AAAI 2021) that tdBN was introduced to train.
//
// Firing-rate regularization (fr_lambda > 0): each spiking layer's mean firing
// rate is pushed into the band [fr_min, fr_max], preventing dead neurons
// (rate→0, gradient vanishes through the surrogate) and bursting neurons
// (rate→1, selectivity lost). The penalty
//   reg = λ · Σ_layers (max(0, fr_min - r)² + max(0, r - fr_max)²)
// is differentiated as d_reg/d_spike = 2λ(r - clamp(r, fr_min, fr_max)) / n and
// injected into the incoming gradient at each LIF spike output during backward,
// mirroring SpikeCountLossImpl. Inert when fr_lambda == 0.
class ThesisDsnnClassifier : public Module<nn::Backend>
{
   public:
    using Tensor = nn::Tensor;

    ThesisDsnnClassifier(int input_dim,
        int hidden_dim,
        int output_dim,
        int depth,
        std::uint32_t seed,
        float fr_lambda = 0.0f,
        float fr_min = 0.05f,
        float fr_max = 0.80f,
        bool use_tdbn = false,
        float tdbn_alpha = 1.0f)
        : fc_in_(std::make_shared<LinearImpl<nn::Backend>>(input_dim, hidden_dim)),
          lif_in_(std::make_shared<LifBPTTImpl<nn::Backend>>(kSnnTimeSteps)),
          fc_out_(std::make_shared<LinearImpl<nn::Backend>>(hidden_dim, output_dim)),
          hidden_dim_(hidden_dim),
          out_dim_(output_dim),
          fr_lambda_(fr_lambda),
          fr_min_(fr_min),
          fr_max_(fr_max),
          use_tdbn_(use_tdbn)
    {
        const int n_blocks = std::max(0, depth - 1);
        hidden_fc_.reserve(static_cast<size_t>(n_blocks));
        hidden_lif_.reserve(static_cast<size_t>(n_blocks));
        for (int i = 0; i < n_blocks; ++i)
        {
            hidden_fc_.push_back(std::make_shared<LinearImpl<nn::Backend>>(hidden_dim, hidden_dim));
            hidden_lif_.push_back(std::make_shared<LifBPTTImpl<nn::Backend>>(kSnnTimeSteps));
        }

        spikes_hidden_.resize(static_cast<size_t>(n_blocks));

        // tdBN (Zheng et al., AAAI 2021): one layer after each Linear, before the
        // LIF it feeds, sized to that Linear's output (hidden_dim) and tied to the
        // LIF threshold V_th so the pre-spike current is normalized to N(0,(α·V_th)²).
        if (use_tdbn_)
        {
            tdbn_in_ = std::make_shared<ThresholdDependentBatchNormImpl<nn::Backend>>(
                static_cast<size_t>(hidden_dim), kSnnVth, kSnnTimeSteps, tdbn_alpha);
            tdbn_hidden_.reserve(static_cast<size_t>(n_blocks));
            for (int i = 0; i < n_blocks; ++i)
                tdbn_hidden_.push_back(
                    std::make_shared<ThresholdDependentBatchNormImpl<nn::Backend>>(
                        static_cast<size_t>(hidden_dim), kSnnVth, kSnnTimeSteps, tdbn_alpha));
        }

        kaimingSNNInitializer(fc_in_, seed + 1U, "e05_dsnn");
        kaimingSNNInitializer(fc_out_, seed + 2U, "e05_dsnn");
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
            kaimingSNNInitializer(
                hidden_fc_[i], seed + 100U + static_cast<unsigned int>(i), "e05_dsnn");
    }

    // input: (B, D). Returns logits (B, output_dim) = mean spike rate over T steps.
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        reset_state();

        const int B = static_cast<int>(input.rows());
        const int D = static_cast<int>(input.cols());
        const int T = kSnnTimeSteps;

        // Rate-encode: tile the static input across T time steps (time-major).
        Tensor x_t(static_cast<nn::Index>(T * B), static_cast<nn::Index>(D));
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < B; ++b)
                for (int d = 0; d < D; ++d)
                    x_t.at(static_cast<nn::Index>(t * B + b), static_cast<nn::Index>(d)) =
                        input.at(static_cast<nn::Index>(b), static_cast<nn::Index>(d));

        // Cache spike trains on every training forward. Needed by the firing-rate
        // regularizer (fr_lambda_ > 0) AND by mean_spike_rate()/sops() so the run
        // diagnostics are populated even when regularization is off. Eval passes
        // (requires_grad == false) don't cache — the last cached train stays the
        // last training batch, matching a spike-loss's last_mean_rate() semantics.
        const bool cache_spikes = requires_grad;

        Tensor h = fc_in_->forward(x_t, requires_grad);
        if (use_tdbn_) h = tdbn_in_->forward(h, requires_grad); // normalize pre-spike current
        h = lif_in_->forward(h, requires_grad);
        if (cache_spikes) spikes_in_ = h; // (T*B, hidden) spike train of input LIF
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            // Residual block: identity skip around Linear→(tdBN)→LIF. Both the skip
            // and the block output are (T*B, hidden_dim), so the add is elementwise.
            // The spike train regularized/back-propagated is the LIF output `b`
            // (before the skip add); the skip carries only the task gradient.
            const Tensor skip = h;
            Tensor b = hidden_fc_[i]->forward(h, requires_grad);
            if (use_tdbn_) b = tdbn_hidden_[i]->forward(b, requires_grad);
            b = hidden_lif_[i]->forward(b, requires_grad);
            if (cache_spikes) spikes_hidden_[i] = b; // spike train of hidden LIF i
            h = b.add(skip);                         // residual connection
        }
        Tensor logits_t = fc_out_->forward(h, requires_grad); // (T*B, C)

        const int C = static_cast<int>(logits_t.cols());
        Tensor logits(static_cast<nn::Index>(B), static_cast<nn::Index>(C));
        const float inv_T = 1.0f / static_cast<float>(T);
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
            {
                float s = 0.0f;
                for (int t = 0; t < T; ++t)
                    s += logits_t.at(static_cast<nn::Index>(t * B + b), static_cast<nn::Index>(c));
                logits.at(static_cast<nn::Index>(b), static_cast<nn::Index>(c)) = s * inv_T;
            }
        return logits;
    }

    // grad_output: (B, C). Returns grad w.r.t. input (B, D).
    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const int B = static_cast<int>(grad_output.rows());
        const int C = static_cast<int>(grad_output.cols());
        const int T = kSnnTimeSteps;

        // Distribute the mean-over-time readout gradient back to each time step.
        Tensor grad_t(static_cast<nn::Index>(T * B), static_cast<nn::Index>(C));
        const float inv_T = 1.0f / static_cast<float>(T);
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C; ++c)
                    grad_t.at(static_cast<nn::Index>(t * B + b), static_cast<nn::Index>(c)) =
                        grad_output.at(static_cast<nn::Index>(b), static_cast<nn::Index>(c)) *
                        inv_T;

        Tensor g = fc_out_->backward(grad_t);
        for (size_t i = hidden_fc_.size(); i-- > 0;)
        {
            // Reverse of h = b + skip: dL/dh_out flows to both paths. Copy it for
            // the identity skip BEFORE any in-place mutation of g, then propagate
            // the block path and add the skip gradient back (identity, no params).
            const Tensor skip_grad = g;                 // deep copy (value-semantics backend)
            add_firing_rate_grad(g, spikes_hidden_[i]); // band penalty on hidden LIF i
            g = hidden_lif_[i]->backward(g);
            if (use_tdbn_) g = tdbn_hidden_[i]->backward(g); // through tdBN (reverse of fwd)
            g = hidden_fc_[i]->backward(g);
            g.add_inplace(skip_grad); // + residual skip path
        }
        add_firing_rate_grad(g, spikes_in_); // band penalty on input LIF
        g = lif_in_->backward(g);
        if (use_tdbn_) g = tdbn_in_->backward(g);
        g = fc_in_->backward(g); // (T*B, D)

        // Sum the per-step input gradients back onto the single static input.
        const int D = static_cast<int>(g.cols());
        Tensor grad_input(static_cast<nn::Index>(B), static_cast<nn::Index>(D));
        for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
            {
                float s = 0.0f;
                for (int t = 0; t < T; ++t)
                    s += g.at(static_cast<nn::Index>(t * B + b), static_cast<nn::Index>(d));
                grad_input.at(static_cast<nn::Index>(b), static_cast<nn::Index>(d)) = s;
            }
        return grad_input;
    }

    [[nodiscard]] auto params() -> std::span<Tensor*> override
    {
        param_ptrs_.clear();
        auto add = [this](const std::span<Tensor*>& p)
        { param_ptrs_.insert(param_ptrs_.end(), p.begin(), p.end()); };

        add(fc_in_->params());
        if (use_tdbn_) add(tdbn_in_->params()); // γ, β of the input tdBN
        add(lif_in_->params());

        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            add(hidden_fc_[i]->params());
            if (use_tdbn_) add(tdbn_hidden_[i]->params());
            add(hidden_lif_[i]->params());
        }

        auto out_p = fc_out_->params();
        param_ptrs_.insert(param_ptrs_.end(), out_p.begin(), out_p.end());
        return {param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        std::map<std::string, Tensor> out;
        auto merge = [&out](const std::string& prefix, const std::map<std::string, Tensor>& sd)
        {
            for (const auto& kv : sd) out[prefix + kv.first] = kv.second;
        };

        merge("fc_in.", fc_in_->state_dict());
        if (use_tdbn_) merge("tdbn_in.", tdbn_in_->state_dict());
        merge("lif_in.", lif_in_->state_dict());
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            merge("hidden_fc." + std::to_string(i) + ".", hidden_fc_[i]->state_dict());
            if (use_tdbn_)
                merge("tdbn_hidden." + std::to_string(i) + ".", tdbn_hidden_[i]->state_dict());
            merge("hidden_lif." + std::to_string(i) + ".", hidden_lif_[i]->state_dict());
        }
        merge("fc_out.", fc_out_->state_dict());
        return out;
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        auto load_prefix = [&sd](const std::string& prefix, auto& layer)
        {
            std::map<std::string, Tensor> sub;
            for (const auto& kv : sd)
            {
                if (kv.first.rfind(prefix, 0) == 0) sub[kv.first.substr(prefix.size())] = kv.second;
            }
            layer->load_state_dict(sub);
        };

        load_prefix("fc_in.", fc_in_);
        if (use_tdbn_) load_prefix("tdbn_in.", tdbn_in_);
        load_prefix("lif_in.", lif_in_);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            load_prefix("hidden_fc." + std::to_string(i) + ".", hidden_fc_[i]);
            if (use_tdbn_) load_prefix("tdbn_hidden." + std::to_string(i) + ".", tdbn_hidden_[i]);
            load_prefix("hidden_lif." + std::to_string(i) + ".", hidden_lif_[i]);
        }
        load_prefix("fc_out.", fc_out_);
    }

    void train(bool on) override
    {
        fc_in_->train(on);
        if (use_tdbn_) tdbn_in_->train(on); // toggle batch vs running stats
        lif_in_->train(on);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            hidden_fc_[i]->train(on);
            if (use_tdbn_) tdbn_hidden_[i]->train(on);
            hidden_lif_[i]->train(on);
        }
        fc_out_->train(on);
    }

    void reset_state() override
    {
        lif_in_->reset_state();
        for (auto& l : hidden_lif_) l->reset_state();
    }

    // ── Run diagnostics (the Trainer queries these to fill EpochResult) ──────────
    // Overall mean firing rate across all spiking layers from the last training
    // forward = (total spikes) / (total spike slots). NaN until a training forward
    // has cached spikes. Matches Guayaquil's "mean firing rate across all SNN neurons".
    [[nodiscard]] auto mean_spike_rate() const -> float
    {
        double spikes = 0.0;
        double slots = 0.0;
        auto acc = [&](const Tensor& s)
        {
            if (s.size() > 0)
            {
                spikes += static_cast<double>(s.sum());
                slots += static_cast<double>(s.size());
            }
        };
        acc(spikes_in_);
        for (const auto& s : spikes_hidden_) acc(s);
        if (slots == 0.0) return std::numeric_limits<float>::quiet_NaN();
        return static_cast<float>(spikes / slots);
    }

    // Synaptic operations per sample for the last training forward: for each
    // spiking layer, (spikes it emitted over all T steps) × (fan-out = width of the
    // next Linear it drives), summed, then divided by the batch size. 0 when no
    // spikes are cached. This is the spike-driven analog of a dense net's MACs.
    [[nodiscard]] auto sops() const -> long long
    {
        if (spikes_in_.size() == 0) return 0;
        const long long batch = static_cast<long long>(spikes_in_.rows()) / kSnnTimeSteps;
        if (batch <= 0) return 0;
        // lif_in drives hidden_fc_[0] (fan-out = hidden) or fc_out_ (fan-out = output).
        double total = static_cast<double>(spikes_in_.sum()) *
                       static_cast<double>(hidden_lif_.empty() ? out_dim_ : hidden_dim_);
        for (size_t i = 0; i < spikes_hidden_.size(); ++i)
        {
            const long long fanout = (i + 1 < spikes_hidden_.size()) ? hidden_dim_ : out_dim_;
            total += static_cast<double>(spikes_hidden_[i].sum()) * static_cast<double>(fanout);
        }
        return static_cast<long long>(total / static_cast<double>(batch));
    }

   private:
    // Add the firing-rate band-penalty gradient of one spiking layer to the
    // incoming gradient g (same shape as the cached spike train). No-op when
    // regularization is disabled or the spike cache is empty (eval pass).
    // d_reg/d_spike = 2λ(r - clamp(r, fr_min_, fr_max_)) / n, broadcast to all
    // elements (mirrors SpikeCountLossImpl). r = mean firing rate of the layer.
    void add_firing_rate_grad(Tensor& g, const Tensor& spikes) const
    {
        if (fr_lambda_ <= 0.0f || spikes.size() == 0) return;
        const float n = static_cast<float>(spikes.size());
        const float r = spikes.sum() / n;
        const float clamped = std::clamp(r, fr_min_, fr_max_);
        const float d_reg = 2.0f * fr_lambda_ * (r - clamped) / n;
        if (d_reg != 0.0f) g.add_scalar_inplace(d_reg);
    }

    std::shared_ptr<LinearImpl<nn::Backend>> fc_in_;
    std::shared_ptr<LifBPTTImpl<nn::Backend>> lif_in_;
    std::vector<std::shared_ptr<LinearImpl<nn::Backend>>> hidden_fc_;
    std::vector<std::shared_ptr<LifBPTTImpl<nn::Backend>>> hidden_lif_;
    std::shared_ptr<LinearImpl<nn::Backend>> fc_out_;
    std::vector<Tensor*> param_ptrs_;

    int hidden_dim_ = 0; // widths, used to weight SOPs by each layer's fan-out
    int out_dim_ = 0;

    // Firing-rate regularization (band [fr_min_, fr_max_], weight fr_lambda_).
    float fr_lambda_ = 0.0f;
    float fr_min_ = 0.05f;
    float fr_max_ = 0.80f;
    Tensor spikes_in_;                  // cached input-LIF spike train (training only)
    std::vector<Tensor> spikes_hidden_; // cached hidden-LIF spike trains

    // tdBN layers (only constructed when use_tdbn_): one before each LIF.
    bool use_tdbn_ = false;
    std::shared_ptr<ThresholdDependentBatchNormImpl<nn::Backend>> tdbn_in_;
    std::vector<std::shared_ptr<ThresholdDependentBatchNormImpl<nn::Backend>>> tdbn_hidden_;
};

} // namespace thesis
