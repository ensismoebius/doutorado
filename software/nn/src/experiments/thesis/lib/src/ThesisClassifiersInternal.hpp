// Internal (not public API) helpers behind run_classifier()/compute_aggregate_stats().
// Split out of ThesisClassifiers.cpp purely to keep that file under the
// project's file-length limit. with_classifier<Fn> and train_model<ModelT>
// are templates instantiated from run_classifier() in ThesisClassifiers.cpp,
// so everything reachable from them has to be visible wherever they are
// instantiated. Kept as a single unnamed namespace, exactly as it was inline
// in the .cpp: this header is included by exactly one translation unit
// (ThesisClassifiers.cpp), so an unnamed namespace here behaves identically
// to writing this code directly in that .cpp -- no linkage change, no ODR
// risk, pure code motion.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "Backend.hpp"
#include "ThesisClassifiers.hpp"
#include "ThesisDsnnClassifier.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "io/StateIO.hpp"
#include "layers/losses/CrossEntropyLoss.hpp"
#include "layers/residual/SimpleResNet.hpp"
#include "progress/ProgressManager.hpp"
#include "statistics/eer_scorer.hpp"
#include "statistics/kfold.hpp"
#include "tensor/Tensor.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ProgressCallback.hpp"

namespace thesis
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
    const ThesisDatasetView& view;
    const std::vector<std::vector<double>>& feature_vectors;
    const std::vector<int>& labels;
    const std::vector<int>& groups; // subject_id per sample (for GroupKFold)
    const nn::Tensor& all_inputs;   // (N, feat_dim) feature matrix
    const nn::Tensor& all_targets;  // (N, n_speakers) one-hot labels
    const ThesisConfig& cfg;
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
    ThesisDsnnClassifier model(ctx.feat_dim,
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

// Build a Trainer for `model`, attach the standard callbacks, and fit. Returns
// the per-epoch learning-curve history (train/val loss, epoch time, SNN spike
// rate + SOPs) so the caller can persist it — the Guayaquil-style run diagnostics.
// val_pairs may be empty (flat CV trains without validation); patience <= 0
// disables early stopping (flat CV does not early-stop).
template <typename ModelT>
auto train_model(ModelT& model,
    const FoldContext& ctx,
    const std::vector<std::pair<nn::Tensor, nn::Tensor>>& train_pairs,
    const std::vector<std::pair<nn::Tensor, nn::Tensor>>& val_pairs,
    size_t fold_idx,
    int total_folds,
    int patience) -> std::vector<nn::training::EpochResult>
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

    return trainer.fit_supervised(train_pairs, val_pairs);
}

// Trainable-parameter count of a model = sum of element counts over params().
// Mirrors the param_count the Guayaquil pipeline records per run.
template <typename ModelT>
auto count_trainable_params(ModelT& model) -> std::size_t
{
    std::size_t n = 0;
    for (const nn::Tensor* p : model.params())
        if (p != nullptr) n += static_cast<std::size_t>(p->size());
    return n;
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

// Persist a trained model to <results_dir>/models/<run_tag>/<feature>/fold_<i>.bin
// (results_dir defaults to results/thesis) and record its path in fr (fr.fold must
// already be set).
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
        const auto train_t0 = std::chrono::steady_clock::now();
        struct InnerCandidate
        {
            double score = -std::numeric_limits<double>::infinity();
            std::map<std::string, nn::Tensor> state;
            std::vector<nn::training::EpochResult> history;
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
                        auto hist = train_model(model,
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
                        cand.history = std::move(hist);
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
        const InnerCandidate* best_cand = nullptr;
        for (const auto& cand : candidates)
        {
            if (!cand.valid) continue;
            if (best_cand == nullptr || cand.score > best_val_score)
            {
                best_val_score = cand.score;
                best_cand = &cand;
            }
        }

        if (best_cand == nullptr) continue;

        FoldResult fr;
        fr.fold = static_cast<int>(outer_idx);
        // train_ms covers the whole inner-CV region that produced the selected model
        // (parallel wall-clock); the winner's learning curve is its inner-training history.
        fr.train_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - train_t0)
                .count();
        fr.history = best_cand->history;

        // ── Outer eval: reload the selected model and score the test fold once ──
        const auto infer_t0 = std::chrono::steady_clock::now();
        const EvalMetrics em = with_classifier(ctx,
            [&](auto& model)
            {
                model.load_state_dict(best_cand->state);
                const EvalMetrics m =
                    evaluate(model, test_feats, test_labels, ctx.n_speakers, ctx.scorer);
                save_fold_model(model.state_dict(), ctx, fr);
                return m;
            });
        fr.infer_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - infer_t0)
                .count();
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
                const auto train_t0 = std::chrono::steady_clock::now();
                fr.history = train_model(
                    model, ctx, train_pairs, {}, outer_idx, total_outer, /*patience=*/0);
                fr.train_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - train_t0)
                                  .count();
                const auto infer_t0 = std::chrono::steady_clock::now();
                const EvalMetrics m =
                    evaluate(model, test_feats, test_labels, ctx.n_speakers, ctx.scorer);
                fr.infer_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - infer_t0)
                                  .count();
                save_fold_model(model.state_dict(), ctx, fr);
                return m;
            });
        set_fold_metrics(fr, em);
        result.outer_folds.push_back(fr);
        advance_global_bar(ctx);
    }
}

} // namespace

} // namespace thesis
