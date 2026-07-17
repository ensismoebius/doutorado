#include "E05Classifiers.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "Backend.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "initializers/kaiming_snn.hpp"
#include "io/StateIO.hpp"
#include "layers/dense/Linear.hpp"
#include "layers/losses/CrossEntropyLoss.hpp"
#include "layers/residual/SimpleResNet.hpp"
#include "layers/spiking/LifBPTT.hpp"
#include "layers/spiking/ThresholdDependentBatchNorm.hpp"
#include "progress/ProgressManager.hpp"
#include "statistics/eer_scorer.hpp"
#include "statistics/kfold.hpp"
#include "tensor/Tensor.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ProgressCallback.hpp"

namespace e05
{

namespace
{

static auto intersect_indices(const std::vector<size_t>& a, const std::vector<size_t>& b)
    -> std::vector<size_t>
{
    std::vector<size_t> out;
    out.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}

static auto make_pairs_from_indices(
    const nn::Tensor& inputs, const nn::Tensor& targets, const std::vector<size_t>& indices)
    -> std::vector<std::pair<nn::Tensor, nn::Tensor>>
{
    std::vector<std::pair<nn::Tensor, nn::Tensor>> pairs;
    pairs.reserve(indices.size());
    for (size_t idx : indices)
        pairs.emplace_back(
            inputs.row(static_cast<nn::Index>(idx)), targets.row(static_cast<nn::Index>(idx)));
    return pairs;
}

struct EvalMetrics
{
    double accuracy = 0.0;
    double f1 = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double specificity = 0.0;
    double eer = 0.0;
    double auc = 0.0;
};

// Number of LIF simulation steps the static feature vector is unrolled over.
// (Tracked as audit item M-4: use a genuine temporal SNN, not a 1-step net.)
constexpr int kSnnTimeSteps = 16;

// Firing threshold of the LIF layers (LifBPTT default). tdBN rescales the
// pre-spike current to N(0,(α·V_th)²), so it must use the same V_th.
constexpr float kSnnVth = 1.0f;

// Temporal Deep Residual Spiking Neural Network for E05.
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
class E05DsnnClassifier : public Module<nn::Backend>
{
   public:
    using Tensor = nn::Tensor;

    E05DsnnClassifier(int input_dim,
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

        const bool cache_spikes = requires_grad && fr_lambda_ > 0.0f;

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

// Single batched forward on (N, D); computes all metrics.
template <typename ModelType>
EvalMetrics evaluate(ModelType& model,
    const std::vector<std::vector<double>>& inputs,
    const std::vector<int>& labels,
    int n_classes,
    const statistics::IEERScorer& eer_scorer)
{
    if (inputs.empty()) return {};

    const auto N = static_cast<nn::Index>(inputs.size());
    const auto D = static_cast<nn::Index>(inputs[0].size());
    nn::Tensor batch = nn::Tensor::zeros(N, D);
    float* dst = batch.mutable_data_ptr();
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        float* row_ptr = dst + static_cast<ptrdiff_t>(i) * D;
        for (nn::Index j = 0; j < D; ++j)
            row_ptr[j] = static_cast<float>(inputs[i][static_cast<size_t>(j)]);
    }

    nn::Tensor all_logits = model.forward(batch, false);

    // Verification-only protocol (audit C-1): folds are speaker-disjoint
    // (GroupKFold), so test speakers are never seen in training. Closed-set
    // argmax accuracy/precision/recall/F1/specificity are structurally invalid
    // for unseen speakers and are therefore NOT reported (set to NaN). The model
    // output is used purely as an embedding for genuine/impostor EER/AUC, i.e.
    // an x-vector-style verification protocol (train on background speakers,
    // enrol unseen test speakers at evaluation time).
    std::vector<std::vector<float>> embeddings(static_cast<size_t>(N));
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    {
        embeddings[i].resize(static_cast<size_t>(n_classes));
        for (int j = 0; j < n_classes; ++j)
            embeddings[i][static_cast<size_t>(j)] = all_logits.at(static_cast<nn::Index>(i), j);
    }

    const double kNaN = std::numeric_limits<double>::quiet_NaN();

    EvalMetrics m;
    m.accuracy = kNaN;
    m.precision = kNaN;
    m.recall = kNaN;
    m.specificity = kNaN;
    m.f1 = kNaN;
    m.eer = eer_scorer.compute_eer(embeddings, labels, n_classes);
    m.auc = eer_scorer.compute_auc(embeddings, labels, n_classes);
    return m;
}

// Verification-only model-selection score: higher is better. Prefers AUC; falls
// back to (1 - EER) when AUC is unavailable; -inf when neither is defined.
inline double selection_score(const EvalMetrics& m)
{
    if (!std::isnan(m.auc)) return m.auc;
    if (!std::isnan(m.eer)) return 1.0 - m.eer;
    return -std::numeric_limits<double>::infinity();
}

// Two-sided t critical value at 95% confidence (t_{0.975, df}). Small lookup
// table for df 1..30; falls back to the normal quantile 1.96 for large df.
// Used for fold-wise confidence intervals where n (folds) is small (audit M-3).
inline double t_crit_95(int df)
{
    static const double kTable[] = {12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042};
    if (df < 1) return 1.96;
    if (df <= 30) return kTable[df - 1];
    return 1.96;
}

// Aggregate one metric across folds: mean, sample SD (÷(count-1)) and 95% CI
// (t_{0.975,count-1}·SD/√count) over the non-NaN fold values. NaN metrics
// (e.g. closed-set accuracy under verification-only) yield NaN mean and 0 spread.
struct MetricAgg
{
    double mean = std::numeric_limits<double>::quiet_NaN();
    double std = 0.0;
    double ci95 = 0.0;
};

inline MetricAgg aggregate_metric(const std::vector<double>& values)
{
    double sum = 0.0;
    int count = 0;
    for (double v : values)
        if (!std::isnan(v))
        {
            sum += v;
            ++count;
        }

    MetricAgg a;
    if (count == 0) return a; // mean stays NaN, spread 0
    a.mean = sum / count;

    if (count < 2) return a; // SD/CI undefined for a single value → 0
    double var = 0.0;
    for (double v : values)
        if (!std::isnan(v))
        {
            const double d = v - a.mean;
            var += d * d;
        }
    a.std = std::sqrt(var / (count - 1));
    a.ci95 = t_crit_95(count - 1) * a.std / std::sqrt(static_cast<double>(count));
    return a;
}

// Parse layer_spec to extract hidden_dim and residual depth.
// Expected format: ["linear:H:relu", "residual:D", "linear:N_speakers:identity"]
// Falls back to defaults on parse failure.
void parse_layer_spec(const std::vector<std::string>& spec, int& hidden_dim, int& depth)
{
    hidden_dim = 128;
    depth = 2;
    for (const auto& s : spec)
    {
        if (s.rfind("linear:", 0) == 0)
        {
            // linear:H:activation — take first occurrence as hidden_dim
            auto p1 = s.find(':', 7);
            if (p1 != std::string::npos)
            {
                auto token = s.substr(7, p1 - 7);
                if (token != "N_speakers")
                {
                    try
                    {
                        hidden_dim = std::stoi(token);
                    }
                    catch (...)
                    {
                    }
                }
            }
        }
        else if (s.rfind("residual:", 0) == 0)
        {
            try
            {
                depth = std::stoi(s.substr(9));
            }
            catch (...)
            {
            }
        }
    }
}

// Updates the global progress bar at each epoch end so the bar advances
// smoothly throughout training, not only on fold completion.
class GlobalBarFractionalCallback : public nn::training::ITrainingCallback
{
   public:
    GlobalBarFractionalCallback(uint32_t bar_id, float fold_base_folds)
        : bar_id_(bar_id), fold_base_(fold_base_folds)
    {
    }

    void on_train_begin(int total_epochs) override
    {
        total_epochs_ = std::max(total_epochs, 1);
    }

    void on_epoch_end(
        const nn::training::TrainingState& state, const nn::training::EpochResult&) override
    {
        const float progress =
            fold_base_ + static_cast<float>(state.epoch) / static_cast<float>(total_epochs_);
        nn::progress::ProgressManager::instance().update_bar(bar_id_, progress);
    }

   private:
    uint32_t bar_id_;
    float fold_base_;
    int total_epochs_ = 1;
};

// ─────────────────────────────────────────────────────────────────────────────
// Cross-validation helpers
//
// run_classifier supports two CV strategies (nested and flat) and two classifier
// types (rnn = SimpleResNet, dsnn = spiking). Naively that is a 2×2 matrix of
// near-identical training code. The helpers below factor out everything the four
// cases share, so each CV loop reads as plain control flow:
//   build model → train → evaluate → record fold.
// ─────────────────────────────────────────────────────────────────────────────

// Immutable per-run data shared by both CV strategies. Bundled into one struct so
// the loop helpers take a single argument instead of a dozen. All members are
// references/values valid for the duration of run_classifier.
struct FoldContext
{
    const E05DatasetView& view;
    const std::vector<std::vector<double>>& feature_vectors;
    const std::vector<int>& labels;
    const std::vector<int>& groups; // subject_id per sample (for GroupKFold)
    const nn::Tensor& all_inputs;   // (N, feat_dim) feature matrix
    const nn::Tensor& all_targets;  // (N, n_speakers) one-hot labels
    const E05Config& cfg;
    const nn::training::TrainerConfig& trainer_cfg;
    const statistics::IEERScorer& scorer;
    const std::string& feature_label;
    int feat_dim;
    int hidden_dim;
    int depth;
    int n_speakers;
    uint32_t global_bar_id;
    int* global_completed;
};

// Construct the classifier selected by cfg.classifier.type and invoke fn(model).
// This is the single place that knows the concrete model types, so every CV loop
// stays model-agnostic and works through the generic-lambda parameter `model`.
template <typename Fn>
auto with_classifier(const FoldContext& ctx, Fn&& fn)
{
    if (ctx.cfg.classifier.type == "rnn")
    {
        // Seed the Kaiming init so the ResNet is reproducible like the dsnn path.
        SimpleResNetImpl<nn::Backend> model(
            ctx.feat_dim, ctx.hidden_dim, ctx.n_speakers, ctx.depth, ctx.cfg.experiment.seed);
        return fn(model);
    }
    E05DsnnClassifier model(ctx.feat_dim,
        ctx.hidden_dim,
        ctx.n_speakers,
        ctx.depth,
        ctx.cfg.experiment.seed,
        ctx.cfg.training.firing_rate_reg_lambda,
        ctx.cfg.training.firing_rate_min,
        ctx.cfg.training.firing_rate_max,
        ctx.cfg.training.batch_normalization == "threshold-dependent",
        ctx.cfg.training.tdbn_alpha);
    return fn(model);
}

// Build a Trainer for `model`, attach the standard callbacks, and fit.
// val_pairs may be empty (flat CV trains without validation); patience <= 0
// disables early stopping (flat CV does not early-stop).
template <typename ModelT>
void train_model(ModelT& model,
    const FoldContext& ctx,
    const std::vector<std::pair<nn::Tensor, nn::Tensor>>& train_pairs,
    const std::vector<std::pair<nn::Tensor, nn::Tensor>>& val_pairs,
    size_t fold_idx,
    int total_folds,
    int patience)
{
    using Loss = CrossEntropyLossImpl<nn::Backend>;
    nn::training::Trainer<ModelT, Loss> trainer(model, ctx.trainer_cfg, Loss{});

    const int fold_num = static_cast<int>(fold_idx) + 1;
    auto progress = std::make_shared<nn::training::ProgressCallback>(
        "Fold " + std::to_string(fold_num) + "/" + std::to_string(total_folds) + " | " +
        ctx.feature_label);
    progress->set_metadata(ctx.feature_label, fold_num, total_folds, "CrossEntropy");
    trainer.add_callback(progress);

    if (ctx.global_bar_id != 0)
        trainer.add_callback(std::make_shared<GlobalBarFractionalCallback>(
            ctx.global_bar_id, static_cast<float>(fold_idx)));

    if (patience > 0)
        trainer.add_callback(std::make_shared<nn::training::EarlyStoppingCallback>(patience));

    trainer.fit_supervised(train_pairs, val_pairs);
}

// Copy the seven evaluation metrics into a fold record.
void set_fold_metrics(FoldResult& fr, const EvalMetrics& em)
{
    fr.accuracy = em.accuracy;
    fr.f1 = em.f1;
    fr.precision = em.precision;
    fr.recall = em.recall;
    fr.specificity = em.specificity;
    fr.eer = em.eer;
    fr.auc = em.auc;
}

// Persist a trained model to results/models/<run_tag>/<feature>/fold_<i>.bin and
// record its path in fr (fr.fold must already be set).
void save_fold_model(
    const std::map<std::string, nn::Tensor>& state, const FoldContext& ctx, FoldResult& fr)
{
    const std::string model_dir = ctx.cfg.dataset.results_dir + "/models/" +
                                  ctx.cfg.experiment.run_tag + "/" + ctx.feature_label;
    std::filesystem::create_directories(model_dir);
    fr.model_path = model_dir + "/fold_" + std::to_string(fr.fold) + ".bin";
    nn::io::save_state_dict(state, fr.model_path);
}

// Advance the shared outer-fold progress bar by one completed fold.
void advance_global_bar(const FoldContext& ctx)
{
    if (ctx.global_bar_id != 0 && ctx.global_completed != nullptr)
        nn::progress::ProgressManager::instance().update_bar(
            ctx.global_bar_id, static_cast<float>(++(*ctx.global_completed)));
}

// ── Per-feature standardization (audit G1) ───────────────────────────────────
// Fit z-score statistics on a fold's TRAINING rows only, apply to train and test.
// This is the input normalization the thesis describes (CMVN-style per-feature
// z-score) and the leakage guard it requires (fit-on-train-only).
struct FeatureScaler
{
    std::vector<double> mean;
    std::vector<double> inv_std;
};

// Per-column mean and 1/std over the given rows of X. Constant columns
// (std < eps) get inv_std = 0 → mapped to a constant 0 after centering.
FeatureScaler fit_scaler(const nn::Tensor& X, const std::vector<size_t>& idx)
{
    const auto D = static_cast<size_t>(X.cols());
    FeatureScaler s{std::vector<double>(D, 0.0), std::vector<double>(D, 0.0)};
    if (idx.empty())
    {
        std::fill(s.inv_std.begin(), s.inv_std.end(), 1.0);
        return s;
    }
    for (size_t i : idx)
        for (size_t j = 0; j < D; ++j)
            s.mean[j] += X.at(static_cast<nn::Index>(i), static_cast<nn::Index>(j));
    const double n = static_cast<double>(idx.size());
    for (size_t j = 0; j < D; ++j) s.mean[j] /= n;

    std::vector<double> var(D, 0.0);
    for (size_t i : idx)
        for (size_t j = 0; j < D; ++j)
        {
            const double d = X.at(static_cast<nn::Index>(i), static_cast<nn::Index>(j)) - s.mean[j];
            var[j] += d * d;
        }
    for (size_t j = 0; j < D; ++j)
    {
        const double sd = std::sqrt(var[j] / n);
        s.inv_std[j] = (sd > 1e-8) ? 1.0 / sd : 0.0;
    }
    return s;
}

// No-op scaler: mean 0, inv_std 1 (used when standardize_features is false).
FeatureScaler identity_scaler(size_t D)
{
    return FeatureScaler{std::vector<double>(D, 0.0), std::vector<double>(D, 1.0)};
}

// Standardized deep copy of X: (x - mean) * inv_std per column.
nn::Tensor apply_scaler(const nn::Tensor& X, const FeatureScaler& s)
{
    nn::Tensor Y = X; // value semantics → deep copy
    for (nn::Index i = 0; i < X.rows(); ++i)
        for (nn::Index j = 0; j < X.cols(); ++j)
            Y.at(i, j) = static_cast<float>(
                (static_cast<double>(X.at(i, j)) - s.mean[static_cast<size_t>(j)]) *
                s.inv_std[static_cast<size_t>(j)]);
    return Y;
}

// Fit the per-fold scaler on train_idx (identity when standardization is off).
FeatureScaler make_fold_scaler(const FoldContext& ctx, const std::vector<size_t>& train_idx)
{
    return ctx.cfg.training.standardize_features
               ? fit_scaler(ctx.all_inputs, train_idx)
               : identity_scaler(static_cast<size_t>(ctx.all_inputs.cols()));
}

// (feature vector, label) subset pulled from a standardized matrix.
void gather_subset_std(const nn::Tensor& Xn,
    const std::vector<int>& all_labels,
    const std::vector<size_t>& idx,
    std::vector<std::vector<double>>& feats,
    std::vector<int>& labs)
{
    feats.clear();
    labs.clear();
    feats.reserve(idx.size());
    labs.reserve(idx.size());
    const auto D = static_cast<size_t>(Xn.cols());
    for (size_t i : idx)
    {
        std::vector<double> row(D);
        for (size_t j = 0; j < D; ++j)
            row[j] = Xn.at(static_cast<nn::Index>(i), static_cast<nn::Index>(j));
        feats.push_back(std::move(row));
        labs.push_back(all_labels[i]);
    }
}

// Nested k-fold CV: an inner loop selects the best model by validation score; the
// selected model is then scored once on the held-out outer test fold. Gives an
// unbiased performance estimate when hyperparameters are tuned on the inner folds.
void run_nested_cv(const FoldContext& ctx,
    ClassificationResult& result,
    const std::vector<size_t>& text_test_indices,
    std::size_t k)
{
    auto outer_policy =
        std::make_shared<statistics::GroupKFoldPolicy>(k, true, ctx.cfg.experiment.seed);
    auto inner_policy = std::make_shared<statistics::GroupKFoldPolicy>(
        k, true, ctx.cfg.experiment.seed ^ 0xDEADBEEFU);
    statistics::NestedKFold nkf(k, k, outer_policy, inner_policy);
    const auto nested_splits = nkf.split(ctx.view.samples.size(), ctx.groups);
    const int total_outer = static_cast<int>(nested_splits.size());

    for (size_t outer_idx = 0; outer_idx < nested_splits.size(); ++outer_idx)
    {
        const auto& outer = nested_splits[outer_idx];

        // Outer-train = every sample not held out as outer test.
        std::vector<size_t> outer_train_indices;
        outer_train_indices.reserve(ctx.view.samples.size() - outer.test_indices.size());
        for (size_t i = 0; i < ctx.view.samples.size(); ++i)
            if (!std::binary_search(outer.test_indices.begin(), outer.test_indices.end(), i))
                outer_train_indices.push_back(i);

        // Keep only test samples whose phrase belongs to the text-split test set.
        const auto outer_test_indices = intersect_indices(outer.test_indices, text_test_indices);
        if (outer_train_indices.empty() || outer_test_indices.empty()) continue;

        // Standardize features with statistics fit on the outer-train rows only,
        // so the outer test fold never leaks into the scaler (audit G1). One
        // scaler per outer fold keeps the model and its test set on the same scale.
        const FeatureScaler scaler = make_fold_scaler(ctx, outer_train_indices);
        const nn::Tensor Xn = apply_scaler(ctx.all_inputs, scaler);

        std::vector<std::vector<double>> test_feats;
        std::vector<int> test_labels;
        gather_subset_std(Xn, ctx.labels, outer_test_indices, test_feats, test_labels);

        // ── Inner loop: train one candidate per inner fold, keep the best ──
        // Inner folds are independent (each trains its own model on its own
        // subset), so they run in parallel. Every candidate's score and state
        // are collected per fold index, then reduced serially in fold order —
        // this keeps the selected model bit-identical to the old serial loop
        // (same first-wins tie-break) regardless of thread scheduling.
        struct InnerCandidate
        {
            double score = -std::numeric_limits<double>::infinity();
            std::map<std::string, nn::Tensor> state;
            bool valid = false;
        };
        std::vector<InnerCandidate> candidates(outer.inner_splits.size());
        std::exception_ptr inner_error;

#pragma omp parallel for schedule(dynamic, 1)
        for (long ii = 0; ii < static_cast<long>(outer.inner_splits.size()); ++ii)
        {
            try
            {
                const auto& inner_ref = outer.inner_splits[static_cast<size_t>(ii)];
                const auto inner_train =
                    intersect_indices(inner_ref.train_indices, outer_train_indices);
                const auto inner_val =
                    intersect_indices(inner_ref.test_indices, outer_train_indices);
                if (inner_train.empty() || inner_val.empty()) continue;

                const auto train_pairs = make_pairs_from_indices(Xn, ctx.all_targets, inner_train);
                const auto val_pairs = make_pairs_from_indices(Xn, ctx.all_targets, inner_val);
                std::vector<std::vector<double>> val_feats;
                std::vector<int> val_labels;
                gather_subset_std(Xn, ctx.labels, inner_val, val_feats, val_labels);

                with_classifier(ctx,
                    [&](auto& model)
                    {
                        train_model(model,
                            ctx,
                            train_pairs,
                            val_pairs,
                            outer_idx,
                            total_outer,
                            ctx.cfg.training.early_stop_patience);
                        const EvalMetrics m =
                            evaluate(model, val_feats, val_labels, ctx.n_speakers, ctx.scorer);
                        auto& cand = candidates[static_cast<size_t>(ii)];
                        cand.score = selection_score(m);
                        cand.state = model.state_dict();
                        cand.valid = true;
                    });
            }
            catch (...)
            {
#pragma omp critical(e05_inner_cv_error)
                if (!inner_error) inner_error = std::current_exception();
            }
        }
        if (inner_error) std::rethrow_exception(inner_error);

        double best_val_score = -std::numeric_limits<double>::infinity();
        std::map<std::string, nn::Tensor> best_state;
        for (const auto& cand : candidates)
        {
            if (!cand.valid) continue;
            if (best_state.empty() || cand.score > best_val_score)
            {
                best_val_score = cand.score;
                best_state = cand.state;
            }
        }

        if (best_state.empty()) continue;

        // ── Outer eval: reload the selected model and score the test fold once ──
        FoldResult fr;
        fr.fold = static_cast<int>(outer_idx);
        const EvalMetrics em = with_classifier(ctx,
            [&](auto& model)
            {
                model.load_state_dict(best_state);
                const EvalMetrics m =
                    evaluate(model, test_feats, test_labels, ctx.n_speakers, ctx.scorer);
                save_fold_model(model.state_dict(), ctx, fr);
                return m;
            });
        set_fold_metrics(fr, em);
        result.outer_folds.push_back(fr);
        advance_global_bar(ctx);
    }
}

// Flat k-fold CV: train on each fold's train split, score on its test split. No
// inner loop and no validation/early-stopping (used when nested_cv is disabled).
void run_flat_cv(const FoldContext& ctx,
    ClassificationResult& result,
    const std::vector<size_t>& text_train_indices,
    const std::vector<size_t>& text_test_indices,
    std::size_t k)
{
    auto outer_policy =
        std::make_shared<statistics::GroupKFoldPolicy>(k, true, ctx.cfg.experiment.seed);
    const auto flat_splits = outer_policy->make_splits(ctx.view.samples.size(), ctx.groups);
    const int total_outer = static_cast<int>(flat_splits.size());

    for (size_t outer_idx = 0; outer_idx < flat_splits.size(); ++outer_idx)
    {
        const auto& split = flat_splits[outer_idx];
        const auto train_indices = intersect_indices(split.train_indices, text_train_indices);
        const auto test_indices = intersect_indices(split.test_indices, text_test_indices);
        if (train_indices.empty() || test_indices.empty()) continue;

        // Fit the scaler on this fold's train rows only (audit G1).
        const FeatureScaler scaler = make_fold_scaler(ctx, train_indices);
        const nn::Tensor Xn = apply_scaler(ctx.all_inputs, scaler);

        std::vector<std::vector<double>> test_feats;
        std::vector<int> test_labels;
        gather_subset_std(Xn, ctx.labels, test_indices, test_feats, test_labels);
        const auto train_pairs = make_pairs_from_indices(Xn, ctx.all_targets, train_indices);

        FoldResult fr;
        fr.fold = static_cast<int>(outer_idx);
        const EvalMetrics em = with_classifier(ctx,
            [&](auto& model)
            {
                train_model(model, ctx, train_pairs, {}, outer_idx, total_outer, /*patience=*/0);
                const EvalMetrics m =
                    evaluate(model, test_feats, test_labels, ctx.n_speakers, ctx.scorer);
                save_fold_model(model.state_dict(), ctx, fr);
                return m;
            });
        set_fold_metrics(fr, em);
        result.outer_folds.push_back(fr);
        advance_global_bar(ctx);
    }
}

} // namespace

auto run_classifier(const E05DatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const E05Config& cfg,
    const statistics::IEERScorer* eer_scorer,
    uint32_t global_bar_id,
    int* global_completed) -> ClassificationResult
{
    if (feature_vectors.empty())
        throw std::invalid_argument("E05Classifiers: empty feature vectors");
    if (feature_vectors.size() != view.samples.size())
        throw std::invalid_argument("E05Classifiers: features/samples size mismatch");

    if (cfg.classifier.type != "rnn" && cfg.classifier.type != "dsnn")
        throw std::invalid_argument("E05Classifiers: classifier type \"" + cfg.classifier.type +
                                    "\" is not implemented. Supported: \"rnn\", \"dsnn\".");

    const int n_speakers = view.n_subjects;
    const int feat_dim = static_cast<int>(feature_vectors[0].size());

    int hidden_dim = 0, depth = 0;
    parse_layer_spec(cfg.classifier.layer_spec, hidden_dim, depth);

    // Map subject_id → sequential class index.
    std::map<int, int> id_to_class;
    for (const auto& s : view.samples)
    {
        if (!id_to_class.count(s.subject_id))
        {
            int idx = static_cast<int>(id_to_class.size());
            id_to_class[s.subject_id] = idx;
        }
    }

    std::vector<int> labels;
    labels.reserve(view.samples.size());
    for (const auto& s : view.samples) labels.push_back(id_to_class[s.subject_id]);

    nn::training::TrainerConfig trainer_cfg;
    trainer_cfg.epochs = cfg.training.epochs;
    trainer_cfg.learning_rate = cfg.training.effective_learning_rate();
    trainer_cfg.optimizer_type = cfg.training.optimizer_type;
    trainer_cfg.optimizer_momentum = cfg.training.optimizer_momentum;
    trainer_cfg.batch_size = cfg.training.samples_per_batch;
    trainer_cfg.weight_decay = cfg.training.weight_decay; // decoupled L2 (rnn + dsnn)

    std::vector<int> groups;
    groups.reserve(view.samples.size());
    for (const auto& s : view.samples) groups.push_back(s.subject_id);

    const std::size_t k = static_cast<std::size_t>(cfg.training.k_folds);

    ClassificationResult result;
    result.feature_set_label = feature_label;
    result.classifier_type = cfg.classifier.type;
    result.text_mode = cfg.classifier.text_mode;

    // Pre-build full dataset tensors once — avoid per-fold reconstruction.
    const auto N_all = static_cast<nn::Index>(feature_vectors.size());
    const auto D = static_cast<nn::Index>(feat_dim);
    const auto C = static_cast<nn::Index>(n_speakers);

    nn::Tensor all_inputs = nn::Tensor::zeros(N_all, D);
    nn::Tensor all_targets = nn::Tensor::zeros(N_all, C);
    {
        float* inp_ptr = all_inputs.mutable_data_ptr();
        float* tgt_ptr = all_targets.mutable_data_ptr();
        for (nn::Index i = 0; i < N_all; ++i)
        {
            const auto& fv = feature_vectors[static_cast<size_t>(i)];
            for (nn::Index j = 0; j < D; ++j)
                inp_ptr[i * D + j] = static_cast<float>(fv[static_cast<size_t>(j)]);
            tgt_ptr[i * C + labels[static_cast<size_t>(i)]] = 1.0f;
        }
    }

    const auto text_split =
        make_text_split(view.samples, cfg.classifier.text_mode, cfg.experiment.seed);
    std::vector<size_t> text_train_indices = text_split.train_indices;
    std::vector<size_t> text_test_indices = text_split.test_indices;
    std::sort(text_train_indices.begin(), text_train_indices.end());
    std::sort(text_test_indices.begin(), text_test_indices.end());

    // Genuine/impostor EER scorer used both for inner-fold model selection and for
    // reporting. Created once (default when the caller passes none) and shared by
    // every fold via FoldContext.
    statistics::GenuineImpostorEERScorer default_scorer;
    const statistics::IEERScorer& scorer = (eer_scorer != nullptr) ? *eer_scorer : default_scorer;

    const FoldContext ctx{view,
        feature_vectors,
        labels,
        groups,
        all_inputs,
        all_targets,
        cfg,
        trainer_cfg,
        scorer,
        feature_label,
        feat_dim,
        hidden_dim,
        depth,
        n_speakers,
        global_bar_id,
        global_completed};

    if (cfg.training.nested_cv)
        run_nested_cv(ctx, result, text_test_indices, k);
    else
        run_flat_cv(ctx, result, text_train_indices, text_test_indices, k);

    compute_aggregate_stats(result);
    return result;
}

void compute_aggregate_stats(ClassificationResult& result)
{
    if (result.outer_folds.empty()) return;

    // Collect per-fold values, then aggregate each metric over its non-NaN folds
    // with sample SD and a t-based 95% CI (audit M-3). Under verification-only
    // (audit C-1) the closed-set metrics are NaN and aggregate to NaN.
    std::vector<double> acc, f1, prec, rec, spec, eer, auc;
    acc.reserve(result.outer_folds.size());
    f1.reserve(result.outer_folds.size());
    prec.reserve(result.outer_folds.size());
    rec.reserve(result.outer_folds.size());
    spec.reserve(result.outer_folds.size());
    eer.reserve(result.outer_folds.size());
    auc.reserve(result.outer_folds.size());
    for (const auto& f : result.outer_folds)
    {
        acc.push_back(f.accuracy);
        f1.push_back(f.f1);
        prec.push_back(f.precision);
        rec.push_back(f.recall);
        spec.push_back(f.specificity);
        eer.push_back(f.eer);
        auc.push_back(f.auc);
    }

    const MetricAgg a_acc = aggregate_metric(acc);
    const MetricAgg a_f1 = aggregate_metric(f1);
    const MetricAgg a_prec = aggregate_metric(prec);
    const MetricAgg a_rec = aggregate_metric(rec);
    const MetricAgg a_spec = aggregate_metric(spec);
    const MetricAgg a_eer = aggregate_metric(eer);
    const MetricAgg a_auc = aggregate_metric(auc);

    result.mean_accuracy = a_acc.mean;
    result.std_accuracy = a_acc.std;
    result.ci95_accuracy = a_acc.ci95;
    result.mean_f1 = a_f1.mean;
    result.std_f1 = a_f1.std;
    result.mean_precision = a_prec.mean;
    result.mean_recall = a_rec.mean;
    result.mean_specificity = a_spec.mean;
    result.std_specificity = a_spec.std;
    result.mean_eer = a_eer.mean;
    result.std_eer = a_eer.std;
    result.ci95_eer = a_eer.ci95;
    result.mean_auc = a_auc.mean;
    result.std_auc = a_auc.std;
}

} // namespace e05
