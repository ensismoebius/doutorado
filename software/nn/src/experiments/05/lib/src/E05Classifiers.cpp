#include "E05Classifiers.hpp"

#include <cmath>
#include <algorithm>
#include <iterator>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <map>

#include "Backend.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "initializers/kaiming_snn.hpp"
#include "io/StateIO.hpp"
#include "layers/dense/Linear.hpp"
#include "layers/losses/CrossEntropyLoss.hpp"
#include "layers/residual/SimpleResNet.hpp"
#include "layers/spiking/LifBPTT.hpp"
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

    static auto make_pairs_from_indices(const nn::Tensor& inputs,
        const nn::Tensor& targets,
        const std::vector<size_t>& indices) -> std::vector<std::pair<nn::Tensor, nn::Tensor>>
    {
        std::vector<std::pair<nn::Tensor, nn::Tensor>> pairs;
        pairs.reserve(indices.size());
        for (size_t idx : indices)
            pairs.emplace_back(inputs.row(static_cast<nn::Index>(idx)),
                targets.row(static_cast<nn::Index>(idx)));
        return pairs;
    }

struct EvalMetrics
{
    double accuracy    = 0.0;
    double f1          = 0.0;
    double precision   = 0.0;
    double recall      = 0.0;
    double specificity = 0.0;
    double eer         = 0.0;
    double auc         = 0.0;
};

// Number of simulation time steps for the temporal DSNN (audit M-4).
constexpr int kSnnTimeSteps = 16;

// Temporal Deep Spiking Neural Network for E05 (audit M-4).
//
// Unlike a single-step thresholding net, this unrolls genuine LIF dynamics over
// kSnnTimeSteps. The static feature vector is rate-encoded by constant-current
// injection: each of the T steps receives the same input row (time-major layout
// (T*B, F) expected by LifBPTT). Spikes integrate over time through LifBPTT
// neurons; the readout is the spike-count (mean firing rate over T) of the final
// linear layer, giving class logits. Gradients flow through real BPTT.
class E05DsnnClassifier : public Module<nn::Backend>
{
public:
    using Tensor = nn::Tensor;

    E05DsnnClassifier(int input_dim, int hidden_dim, int output_dim, int depth,
                      std::uint32_t seed)
        : fc_in_(std::make_shared<LinearImpl<nn::Backend>>(input_dim, hidden_dim)),
          lif_in_(std::make_shared<LifBPTTImpl<nn::Backend>>(kSnnTimeSteps)),
          fc_out_(std::make_shared<LinearImpl<nn::Backend>>(hidden_dim, output_dim))
    {
        const int n_blocks = std::max(0, depth - 1);
        hidden_fc_.reserve(static_cast<size_t>(n_blocks));
        hidden_lif_.reserve(static_cast<size_t>(n_blocks));
        for (int i = 0; i < n_blocks; ++i)
        {
            hidden_fc_.push_back(std::make_shared<LinearImpl<nn::Backend>>(hidden_dim, hidden_dim));
            hidden_lif_.push_back(std::make_shared<LifBPTTImpl<nn::Backend>>(kSnnTimeSteps));
        }

        kaimingSNNInitializer(fc_in_, seed + 1U, "e05_dsnn");
        kaimingSNNInitializer(fc_out_, seed + 2U, "e05_dsnn");
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
            kaimingSNNInitializer(hidden_fc_[i], seed + 100U + static_cast<unsigned int>(i), "e05_dsnn");
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

        Tensor h = fc_in_->forward(x_t, requires_grad);
        h = lif_in_->forward(h, requires_grad);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            h = hidden_fc_[i]->forward(h, requires_grad);
            h = hidden_lif_[i]->forward(h, requires_grad);
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
                        grad_output.at(static_cast<nn::Index>(b), static_cast<nn::Index>(c)) * inv_T;

        Tensor g = fc_out_->backward(grad_t);
        for (size_t i = hidden_fc_.size(); i-- > 0;)
        {
            g = hidden_lif_[i]->backward(g);
            g = hidden_fc_[i]->backward(g);
        }
        g = lif_in_->backward(g);
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
        auto in_p = fc_in_->params();
        param_ptrs_.insert(param_ptrs_.end(), in_p.begin(), in_p.end());

        auto lif_in_p = lif_in_->params();
        param_ptrs_.insert(param_ptrs_.end(), lif_in_p.begin(), lif_in_p.end());

        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            auto p_fc = hidden_fc_[i]->params();
            param_ptrs_.insert(param_ptrs_.end(), p_fc.begin(), p_fc.end());
            auto p_lif = hidden_lif_[i]->params();
            param_ptrs_.insert(param_ptrs_.end(), p_lif.begin(), p_lif.end());
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
            for (const auto& kv : sd)
                out[prefix + kv.first] = kv.second;
        };

        merge("fc_in.", fc_in_->state_dict());
        merge("lif_in.", lif_in_->state_dict());
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            merge("hidden_fc." + std::to_string(i) + ".", hidden_fc_[i]->state_dict());
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
                if (kv.first.rfind(prefix, 0) == 0)
                    sub[kv.first.substr(prefix.size())] = kv.second;
            }
            layer->load_state_dict(sub);
        };

        load_prefix("fc_in.", fc_in_);
        load_prefix("lif_in.", lif_in_);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            load_prefix("hidden_fc." + std::to_string(i) + ".", hidden_fc_[i]);
            load_prefix("hidden_lif." + std::to_string(i) + ".", hidden_lif_[i]);
        }
        load_prefix("fc_out.", fc_out_);
    }

    void train(bool on) override
    {
        fc_in_->train(on);
        lif_in_->train(on);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            hidden_fc_[i]->train(on);
            hidden_lif_[i]->train(on);
        }
        fc_out_->train(on);
    }

    void reset_state() override
    {
        lif_in_->reset_state();
        for (auto& l : hidden_lif_)
            l->reset_state();
    }

private:
    std::shared_ptr<LinearImpl<nn::Backend>> fc_in_;
    std::shared_ptr<LifBPTTImpl<nn::Backend>> lif_in_;
    std::vector<std::shared_ptr<LinearImpl<nn::Backend>>> hidden_fc_;
    std::vector<std::shared_ptr<LifBPTTImpl<nn::Backend>>> hidden_lif_;
    std::shared_ptr<LinearImpl<nn::Backend>> fc_out_;
    std::vector<Tensor*> param_ptrs_;
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
    m.accuracy    = kNaN;
    m.precision   = kNaN;
    m.recall      = kNaN;
    m.specificity = kNaN;
    m.f1          = kNaN;
    m.eer         = eer_scorer.compute_eer(embeddings, labels, n_classes);
    m.auc         = eer_scorer.compute_auc(embeddings, labels, n_classes);
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
    static const double kTable[] = {
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201,  2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080,  2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
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
    double std  = 0.0;
    double ci95 = 0.0;
};

inline MetricAgg aggregate_metric(const std::vector<double>& values)
{
    double sum = 0.0;
    int count = 0;
    for (double v : values)
        if (!std::isnan(v)) { sum += v; ++count; }

    MetricAgg a;
    if (count == 0) return a; // mean stays NaN, spread 0
    a.mean = sum / count;

    if (count < 2) return a; // SD/CI undefined for a single value → 0
    double var = 0.0;
    for (double v : values)
        if (!std::isnan(v)) { const double d = v - a.mean; var += d * d; }
    a.std  = std::sqrt(var / (count - 1));
    a.ci95 = t_crit_95(count - 1) * a.std / std::sqrt(static_cast<double>(count));
    return a;
}

// Parse layer_spec to extract hidden_dim and residual depth.
// Expected format: ["linear:H:relu", "residual:D", "linear:N_speakers:identity"]
// Falls back to defaults on parse failure.
void parse_layer_spec(const std::vector<std::string>& spec,
    int& hidden_dim, int& depth)
{
    hidden_dim = 128;
    depth      = 2;
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
                    try { hidden_dim = std::stoi(token); } catch (...) {}
                }
            }
        }
        else if (s.rfind("residual:", 0) == 0)
        {
            try { depth = std::stoi(s.substr(9)); } catch (...) {}
        }
    }
}

// Updates the global progress bar at each epoch end so the bar advances
// smoothly throughout training, not only on fold completion.
class GlobalBarFractionalCallback : public nn::training::ITrainingCallback
{
public:
    GlobalBarFractionalCallback(uint32_t bar_id, float fold_base_folds)
        : bar_id_(bar_id), fold_base_(fold_base_folds) {}

    void on_train_begin(int total_epochs) override
    {
        total_epochs_ = std::max(total_epochs, 1);
    }

    void on_epoch_end(const nn::training::TrainingState& state,
                      const nn::training::EpochResult&) override
    {
        const float progress = fold_base_ +
            static_cast<float>(state.epoch) / static_cast<float>(total_epochs_);
        nn::progress::ProgressManager::instance().update_bar(bar_id_, progress);
    }

private:
    uint32_t bar_id_;
    float    fold_base_;
    int      total_epochs_ = 1;
};

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
        throw std::invalid_argument(
            "E05Classifiers: classifier type \"" + cfg.classifier.type +
            "\" is not implemented. Supported: \"rnn\", \"dsnn\".");

    const int n_speakers = view.n_subjects;
    const int feat_dim   = static_cast<int>(feature_vectors[0].size());

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
    for (const auto& s : view.samples)
        labels.push_back(id_to_class[s.subject_id]);

    nn::training::TrainerConfig trainer_cfg;
    trainer_cfg.epochs        = cfg.training.epochs;
    trainer_cfg.learning_rate = cfg.training.learning_rate;
    trainer_cfg.batch_size    = cfg.training.samples_per_batch;

    std::vector<int> groups;
    groups.reserve(view.samples.size());
    for (const auto& s : view.samples)
        groups.push_back(s.subject_id);

    const std::size_t k = static_cast<std::size_t>(cfg.training.k_folds);
    auto outer_policy = std::make_shared<statistics::GroupKFoldPolicy>(
        k, true, cfg.experiment.seed);

    ClassificationResult result;
    result.feature_set_label = feature_label;
    result.classifier_type   = cfg.classifier.type;
    result.text_mode         = cfg.classifier.text_mode;

    // Pre-build full dataset tensors once — avoid per-fold reconstruction.
    const auto N_all = static_cast<nn::Index>(feature_vectors.size());
    const auto D     = static_cast<nn::Index>(feat_dim);
    const auto C     = static_cast<nn::Index>(n_speakers);

    nn::Tensor all_inputs  = nn::Tensor::zeros(N_all, D);
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

    const auto text_split = make_text_split(view.samples, cfg.classifier.text_mode,
        cfg.experiment.seed);
    std::vector<size_t> text_train_indices = text_split.train_indices;
    std::vector<size_t> text_test_indices  = text_split.test_indices;
    std::sort(text_train_indices.begin(), text_train_indices.end());
    std::sort(text_test_indices.begin(), text_test_indices.end());

    // ── Nested CV ──────────────────────────────────────────────────────────
    if (cfg.training.nested_cv)
    {
        auto inner_policy = std::make_shared<statistics::GroupKFoldPolicy>(
            k, true, cfg.experiment.seed ^ 0xDEADBEEFU);
        statistics::NestedKFold nkf(k, k, outer_policy, inner_policy);
        auto nested_splits = nkf.split(view.samples.size(), groups);

        const int total_outer = static_cast<int>(nested_splits.size());

        for (size_t outer_idx = 0; outer_idx < nested_splits.size(); ++outer_idx)
        {
            const auto& outer = nested_splits[outer_idx];

            std::vector<size_t> outer_train_indices;
            outer_train_indices.reserve(view.samples.size() - outer.test_indices.size());
            for (size_t i = 0; i < view.samples.size(); ++i)
            {
                if (!std::binary_search(outer.test_indices.begin(), outer.test_indices.end(), i))
                    outer_train_indices.push_back(i);
            }

            const auto outer_test_indices = intersect_indices(outer.test_indices, text_test_indices);

            if (outer_train_indices.empty() || outer_test_indices.empty())
                continue;

            std::vector<std::vector<double>> test_feats;
            std::vector<int> test_labels;
            for (size_t idx : outer_test_indices)
            {
                test_feats.push_back(feature_vectors[idx]);
                test_labels.push_back(labels[idx]);
            }

            statistics::GenuineImpostorEERScorer default_scorer;
            const statistics::IEERScorer& scorer =
                (eer_scorer != nullptr) ? *eer_scorer : default_scorer;

            double best_val_score = -std::numeric_limits<double>::infinity();
            std::map<std::string, nn::Tensor> best_state;

            for (const auto& inner_ref : outer.inner_splits)
            {
                const auto inner_train_indices =
                    intersect_indices(inner_ref.train_indices, outer_train_indices);
                const auto inner_val_indices =
                    intersect_indices(inner_ref.test_indices, outer_train_indices);

                if (inner_train_indices.empty() || inner_val_indices.empty())
                    continue;

                auto train_pairs =
                    make_pairs_from_indices(all_inputs, all_targets, inner_train_indices);
                auto val_pairs =
                    make_pairs_from_indices(all_inputs, all_targets, inner_val_indices);

                std::vector<std::vector<double>> val_feats;
                std::vector<int> val_labels;
                val_feats.reserve(inner_val_indices.size());
                val_labels.reserve(inner_val_indices.size());
                for (size_t idx : inner_val_indices)
                {
                    val_feats.push_back(feature_vectors[idx]);
                    val_labels.push_back(labels[idx]);
                }

                EvalMetrics val_metrics;
                if (cfg.classifier.type == "rnn")
                {
                    SimpleResNetImpl<nn::Backend> candidate_model(feat_dim, hidden_dim, n_speakers, depth);
                    CrossEntropyLossImpl<nn::Backend> loss_fn;
                    nn::training::Trainer<SimpleResNetImpl<nn::Backend>,
                        CrossEntropyLossImpl<nn::Backend>> trainer(candidate_model, trainer_cfg, loss_fn);

                    const int fold_num = static_cast<int>(outer_idx) + 1;
                    auto prog_cb = std::make_shared<nn::training::ProgressCallback>(
                        "Fold " + std::to_string(fold_num) + "/" + std::to_string(total_outer) +
                        " | " + feature_label);
                    prog_cb->set_metadata(feature_label, fold_num, total_outer, "CrossEntropy");
                    trainer.add_callback(prog_cb);

                    if (global_bar_id != 0)
                    {
                        trainer.add_callback(std::make_shared<GlobalBarFractionalCallback>(
                            global_bar_id, static_cast<float>(outer_idx)));
                    }

                    if (cfg.training.early_stop_patience > 0)
                    {
                        auto stopper = std::make_shared<nn::training::EarlyStoppingCallback>(
                            cfg.training.early_stop_patience);
                        trainer.add_callback(stopper);
                    }

                    trainer.fit_supervised(train_pairs, val_pairs);
                    val_metrics = evaluate(candidate_model, val_feats, val_labels, n_speakers, scorer);

                    if (best_state.empty() || selection_score(val_metrics) > best_val_score)
                    {
                        best_val_score = selection_score(val_metrics);
                        best_state = candidate_model.state_dict();
                    }
                }
                else
                {
                    E05DsnnClassifier candidate_model(
                        feat_dim, hidden_dim, n_speakers, depth, cfg.experiment.seed);
                    CrossEntropyLossImpl<nn::Backend> loss_fn;
                    nn::training::Trainer<E05DsnnClassifier,
                        CrossEntropyLossImpl<nn::Backend>> trainer(candidate_model, trainer_cfg, loss_fn);

                    const int fold_num = static_cast<int>(outer_idx) + 1;
                    auto prog_cb = std::make_shared<nn::training::ProgressCallback>(
                        "Fold " + std::to_string(fold_num) + "/" + std::to_string(total_outer) +
                        " | " + feature_label);
                    prog_cb->set_metadata(feature_label, fold_num, total_outer, "CrossEntropy");
                    trainer.add_callback(prog_cb);

                    if (global_bar_id != 0)
                    {
                        trainer.add_callback(std::make_shared<GlobalBarFractionalCallback>(
                            global_bar_id, static_cast<float>(outer_idx)));
                    }

                    if (cfg.training.early_stop_patience > 0)
                    {
                        auto stopper = std::make_shared<nn::training::EarlyStoppingCallback>(
                            cfg.training.early_stop_patience);
                        trainer.add_callback(stopper);
                    }

                    trainer.fit_supervised(train_pairs, val_pairs);
                    val_metrics = evaluate(candidate_model, val_feats, val_labels, n_speakers, scorer);

                    if (best_state.empty() || selection_score(val_metrics) > best_val_score)
                    {
                        best_val_score = selection_score(val_metrics);
                        best_state = candidate_model.state_dict();
                    }
                }
            }

            if (best_state.empty())
                continue;

            EvalMetrics em;
            std::map<std::string, nn::Tensor> model_state_to_save;
            if (cfg.classifier.type == "rnn")
            {
                SimpleResNetImpl<nn::Backend> model(feat_dim, hidden_dim, n_speakers, depth);
                model.load_state_dict(best_state);
                em = evaluate(model, test_feats, test_labels, n_speakers, scorer);
                model_state_to_save = model.state_dict();
            }
            else
            {
                E05DsnnClassifier model(feat_dim, hidden_dim, n_speakers, depth, cfg.experiment.seed);
                model.load_state_dict(best_state);
                em = evaluate(model, test_feats, test_labels, n_speakers, scorer);
                model_state_to_save = model.state_dict();
            }

            FoldResult fr;
            fr.fold        = static_cast<int>(outer_idx);
            fr.accuracy    = em.accuracy;
            fr.f1          = em.f1;
            fr.precision   = em.precision;
            fr.recall      = em.recall;
            fr.specificity = em.specificity;
            fr.eer         = em.eer;
            fr.auc         = em.auc;

            {
                const std::string model_dir =
                    cfg.dataset.results_dir + "/models/" +
                    cfg.experiment.run_tag + "/" + feature_label;
                std::filesystem::create_directories(model_dir);
                fr.model_path = model_dir + "/fold_" + std::to_string(fr.fold) + ".bin";
                nn::io::save_state_dict(model_state_to_save, fr.model_path);
            }

            result.outer_folds.push_back(fr);

            if (global_bar_id != 0 && global_completed != nullptr)
            {
                nn::progress::ProgressManager::instance().update_bar(
                    global_bar_id, static_cast<float>(++(*global_completed)));
            }
        }
    }
    else // ── Flat K-fold (no inner loop) ──────────────────────────────────
    {
        auto flat_splits = outer_policy->make_splits(view.samples.size(), groups);
        const int total_outer = static_cast<int>(flat_splits.size());

        for (size_t outer_idx = 0; outer_idx < flat_splits.size(); ++outer_idx)
        {
            const auto& split = flat_splits[outer_idx];

            const auto train_indices = intersect_indices(split.train_indices, text_train_indices);
            const auto test_indices   = intersect_indices(split.test_indices, text_test_indices);

            if (train_indices.empty() || test_indices.empty())
                continue;

            std::vector<std::vector<double>> test_feats;
            std::vector<int> test_labels;
            for (size_t idx : test_indices)
            {
                test_feats.push_back(feature_vectors[idx]);
                test_labels.push_back(labels[idx]);
            }

            const int fold_num = static_cast<int>(outer_idx) + 1;
            auto train_pairs = make_pairs_from_indices(all_inputs, all_targets, train_indices);
            statistics::GenuineImpostorEERScorer default_scorer;
            const statistics::IEERScorer& scorer =
                (eer_scorer != nullptr) ? *eer_scorer : default_scorer;

            EvalMetrics em;
            std::map<std::string, nn::Tensor> model_state_to_save;
            if (cfg.classifier.type == "rnn")
            {
                SimpleResNetImpl<nn::Backend> model(feat_dim, hidden_dim, n_speakers, depth);
                CrossEntropyLossImpl<nn::Backend> loss_fn;
                nn::training::Trainer<SimpleResNetImpl<nn::Backend>,
                    CrossEntropyLossImpl<nn::Backend>> trainer(model, trainer_cfg, loss_fn);

                auto prog_cb = std::make_shared<nn::training::ProgressCallback>(
                    "Fold " + std::to_string(fold_num) + "/" + std::to_string(total_outer) +
                    " | " + feature_label);
                prog_cb->set_metadata(feature_label, fold_num, total_outer, "CrossEntropy");
                trainer.add_callback(prog_cb);
                if (global_bar_id != 0)
                {
                    trainer.add_callback(std::make_shared<GlobalBarFractionalCallback>(
                        global_bar_id, static_cast<float>(outer_idx)));
                }
                trainer.fit_supervised(train_pairs, {});
                em = evaluate(model, test_feats, test_labels, n_speakers, scorer);
                model_state_to_save = model.state_dict();
            }
            else
            {
                E05DsnnClassifier model(feat_dim, hidden_dim, n_speakers, depth, cfg.experiment.seed);
                CrossEntropyLossImpl<nn::Backend> loss_fn;
                nn::training::Trainer<E05DsnnClassifier,
                    CrossEntropyLossImpl<nn::Backend>> trainer(model, trainer_cfg, loss_fn);

                auto prog_cb = std::make_shared<nn::training::ProgressCallback>(
                    "Fold " + std::to_string(fold_num) + "/" + std::to_string(total_outer) +
                    " | " + feature_label);
                prog_cb->set_metadata(feature_label, fold_num, total_outer, "CrossEntropy");
                trainer.add_callback(prog_cb);
                if (global_bar_id != 0)
                {
                    trainer.add_callback(std::make_shared<GlobalBarFractionalCallback>(
                        global_bar_id, static_cast<float>(outer_idx)));
                }
                trainer.fit_supervised(train_pairs, {});
                em = evaluate(model, test_feats, test_labels, n_speakers, scorer);
                model_state_to_save = model.state_dict();
            }

            FoldResult fr;
            fr.fold        = static_cast<int>(outer_idx);
            fr.accuracy    = em.accuracy;
            fr.f1          = em.f1;
            fr.precision   = em.precision;
            fr.recall      = em.recall;
            fr.specificity = em.specificity;
            fr.eer         = em.eer;
            fr.auc         = em.auc;

            {
                const std::string model_dir =
                    cfg.dataset.results_dir + "/models/" +
                    cfg.experiment.run_tag + "/" + feature_label;
                std::filesystem::create_directories(model_dir);
                fr.model_path = model_dir + "/fold_" + std::to_string(fr.fold) + ".bin";
                nn::io::save_state_dict(model_state_to_save, fr.model_path);
            }

            result.outer_folds.push_back(fr);

            if (global_bar_id != 0 && global_completed != nullptr)
            {
                nn::progress::ProgressManager::instance().update_bar(
                    global_bar_id, static_cast<float>(++(*global_completed)));
            }
        }
    }

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

    const MetricAgg a_acc  = aggregate_metric(acc);
    const MetricAgg a_f1   = aggregate_metric(f1);
    const MetricAgg a_prec = aggregate_metric(prec);
    const MetricAgg a_rec  = aggregate_metric(rec);
    const MetricAgg a_spec = aggregate_metric(spec);
    const MetricAgg a_eer  = aggregate_metric(eer);
    const MetricAgg a_auc  = aggregate_metric(auc);

    result.mean_accuracy    = a_acc.mean;
    result.std_accuracy     = a_acc.std;
    result.ci95_accuracy    = a_acc.ci95;
    result.mean_f1          = a_f1.mean;
    result.std_f1           = a_f1.std;
    result.mean_precision   = a_prec.mean;
    result.mean_recall      = a_rec.mean;
    result.mean_specificity = a_spec.mean;
    result.std_specificity  = a_spec.std;
    result.mean_eer         = a_eer.mean;
    result.std_eer          = a_eer.std;
    result.ci95_eer         = a_eer.ci95;
    result.mean_auc         = a_auc.mean;
    result.std_auc          = a_auc.std;
}

} // namespace e05
