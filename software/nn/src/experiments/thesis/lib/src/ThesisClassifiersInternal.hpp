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
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#include "Backend.hpp"
#include "IndexIntersection.hpp"
#include "ThesisAnnClassifier.hpp"
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
    std::optional<InferenceEnergy> energy; // set by evaluate() for a network
};

inline auto to_inference_energy(const meeting01::EnergyLedger& primary,
    const meeting01::EnergyLedger& as_executed,
    double spikes) -> InferenceEnergy
{
    namespace en = meeting01::energy;
    InferenceEnergy r;
    r.nj = primary.total_nj();
    r.ops_nj = primary.ops_pj() / 1000.0;
    r.mem_nj = primary.mem_pj / 1000.0;
    r.nj_as_executed = as_executed.total_nj();
    r.nj_sram8k = primary.total_nj_at(en::kSramPoints[0].bytes);
    r.nj_sram32k = primary.total_nj_at(en::kSramPoints[1].bytes);
    r.nj_sram1m = primary.total_nj_at(en::kSramPoints[2].bytes);
    r.fp_macs = primary.fp_macs;
    r.fp_adds = primary.fp_adds;
    r.accesses = primary.accesses();
    r.spikes = spikes;
    return r;
}

// The energy of the last forward's samples (call right after the evaluation forward).
inline auto measure_energy(const ThesisDsnnClassifier& m) -> InferenceEnergy
{
    return to_inference_energy(m.inference_ledger(/*repeated_once=*/true),
        m.inference_ledger(false),
        m.spikes_per_sample());
}
inline auto measure_energy(const ThesisAnnClassifier& m) -> InferenceEnergy
{
    const auto e = m.inference_ledger();
    return to_inference_energy(e, e, 0.0);
}

EvalMetrics score_embeddings(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int n_classes,
    const statistics::IEERScorer& eer_scorer,
    const std::string& where);

// Single batched forward on (N, D); computes all metrics.
template <typename ModelType>
EvalMetrics evaluate(ModelType& model,
    const std::vector<std::vector<double>>& inputs,
    const std::vector<int>& labels,
    int n_classes,
    const statistics::IEERScorer& eer_scorer,
    const std::string& where)
{
    if (inputs.empty())
        throw std::logic_error("ThesisClassifiers: " + where + " has no evaluation rows");

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
    const InferenceEnergy energy = measure_energy(model); // this pass's events
    std::vector<std::vector<float>> embeddings(static_cast<size_t>(N));
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    {
        embeddings[i].resize(static_cast<size_t>(n_classes));
        for (int j = 0; j < n_classes; ++j)
            embeddings[i][static_cast<size_t>(j)] = all_logits.at(static_cast<nn::Index>(i), j);
    }
    EvalMetrics m = score_embeddings(embeddings, labels, n_classes, eer_scorer, where);
    m.energy = energy;
    return m;
}

// EER and AUC of one set of embeddings (the model's logits, or the features themselves for
// the cosine control).
EvalMetrics score_embeddings(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int n_classes,
    const statistics::IEERScorer& eer_scorer,
    const std::string& where)
{
    const auto N = static_cast<nn::Index>(embeddings.size());

    // Verification-only protocol (audit C-1): folds are speaker-disjoint
    // (GroupKFold), so test speakers are never seen in training. Closed-set
    // argmax accuracy/precision/recall/F1/specificity are structurally invalid
    // for unseen speakers and are therefore NOT reported (set to NaN). The model
    // output is used purely as an embedding for genuine/impostor EER/AUC, i.e.
    // an x-vector-style verification protocol (train on background speakers,
    // enrol unseen test speakers at evaluation time).
    const double kNaN = std::numeric_limits<double>::quiet_NaN();

    EvalMetrics m;
    m.accuracy = kNaN;
    m.precision = kNaN;
    m.recall = kNaN;
    m.specificity = kNaN;
    m.f1 = kNaN;
    // The scorers refuse a set they cannot score (a speaker without probes, fewer than two
    // enrolled speakers), so a NaN here is not an expected outcome.
    try
    {
        m.eer = eer_scorer.compute_eer(embeddings, labels, n_classes);
        m.auc = eer_scorer.compute_auc(embeddings, labels, n_classes);
    }
    catch (const std::exception& e)
    {
        const std::set<int> speakers(labels.begin(), labels.end());
        throw std::runtime_error("ThesisClassifiers: " + where + " (" + std::to_string(N) +
                                 " samples, " + std::to_string(speakers.size()) +
                                 " speakers) cannot be scored: " + e.what());
    }
    if (!std::isfinite(m.eer) || !std::isfinite(m.auc))
        throw std::logic_error("ThesisClassifiers: " + where +
                               ": EER/AUC came out non-finite (EER " + std::to_string(m.eer) +
                               ", AUC " + std::to_string(m.auc) + ")");
    return m;
}

// Verification-only model-selection score: higher is better. The AUC, which evaluate()
// guarantees is finite; there is no second-choice score.
inline double selection_score(const EvalMetrics& m)
{
    return m.auc;
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
    // Outside the table there is no value to give: 1.96 (the normal quantile) is what this
    // used to return, which understates the interval at df 31-120 and is meaningless at df 0.
    if (df < 1 || df > 30)
        throw std::invalid_argument("ThesisClassifiers: no t_{0.975} for df " + std::to_string(df) +
                                    " (table covers 1..30, i.e. 2..31 folds); extend kTable");
    return kTable[df - 1];
}

// Aggregate one metric across folds: mean, sample SD (÷(count-1)) and 95% CI
// (t_{0.975,count-1}·SD/√count).
//
// A metric is either measured in every fold or in none. "None" is the closed-set
// metrics under verification-only (audit C-1): NaN in every fold by design, so mean, SD
// and CI are all NaN (not applicable), never a spread of 0. A metric missing in SOME
// folds used to be averaged over the folds that had it, which silently changes what
// the mean is over; it is refused, as is a single fold (no SD exists for it).
struct MetricAgg
{
    double mean = std::numeric_limits<double>::quiet_NaN();
    double std = std::numeric_limits<double>::quiet_NaN();
    double ci95 = std::numeric_limits<double>::quiet_NaN();
};

inline MetricAgg aggregate_metric(const std::vector<double>& values)
{
    const auto n_nan =
        std::count_if(values.begin(), values.end(), [](double v) { return std::isnan(v); });
    MetricAgg a;
    if (!values.empty() && static_cast<size_t>(n_nan) == values.size()) return a; // not applicable
    if (n_nan != 0)
        throw std::logic_error("ThesisClassifiers: a metric is NaN in " + std::to_string(n_nan) +
                               " of " + std::to_string(values.size()) +
                               " folds; it must be measured in every fold or in none");
    if (values.size() < 2)
        throw std::logic_error("ThesisClassifiers: " + std::to_string(values.size()) +
                               " fold(s) to aggregate; SD and CI need at least 2");

    const auto count = static_cast<double>(values.size());
    double sum = 0.0;
    for (double v : values) sum += v;
    a.mean = sum / count;
    double var = 0.0;
    for (double v : values) var += (v - a.mean) * (v - a.mean);
    a.std = std::sqrt(var / (count - 1.0));
    a.ci95 = t_crit_95(static_cast<int>(values.size()) - 1) * a.std / std::sqrt(count);
    return a;
}

struct ClassifierShape
{
    int hidden_dim;
    int depth;
};

// Hidden width and residual depth from layer_spec, which must be exactly
//   ["linear:<H>:relu", "residual:<D>", "linear:N_speakers:identity"]  ->  {H, D}
// — the one graph both classifiers build (Linear -> nonlinearity -> D - 1 residual blocks
// Linear -> nonlinearity + skip -> Linear to the speakers; LIF in the DSNN, ReLU in the ANN).
// Anything else is refused: an unreadable spec used to become {128, 2}, and items the code does not
// build (a second hidden layer, another activation) used to be passed over in silence.
ClassifierShape parse_layer_spec(const std::vector<std::string>& spec)
{
    const auto refuse = [&](const std::string& why)
    {
        std::string got;
        for (const auto& s : spec) got += (got.empty() ? "\"" : ", \"") + s + "\"";
        return std::invalid_argument("ThesisClassifiers: classifier.layer_spec [" + got + "] " +
                                     why +
                                     "; it must be [\"linear:<H>:relu\", \"residual:<D>\", "
                                     "\"linear:N_speakers:identity\"] with H, D positive integers");
    };
    const auto positive_int = [&](const std::string& token)
    {
        if (token.empty() || token.size() > 9 ||
            token.find_first_not_of("0123456789") != std::string::npos || std::stoi(token) < 1)
            throw refuse("has \"" + token + "\" where a positive integer belongs");
        return std::stoi(token);
    };

    if (spec.size() != 3) throw refuse("has " + std::to_string(spec.size()) + " items, not 3");
    const std::string& hidden = spec[0];
    const auto colon = hidden.find(':', 7);
    if (hidden.rfind("linear:", 0) != 0 || colon == std::string::npos ||
        hidden.substr(colon + 1) != "relu")
        throw refuse("does not start with linear:<H>:relu");
    if (spec[1].rfind("residual:", 0) != 0) throw refuse("has no residual:<D> second item");
    if (spec[2] != "linear:N_speakers:identity")
        throw refuse("does not end with linear:N_speakers:identity");
    return {positive_int(hidden.substr(7, colon - 7)), positive_int(spec[1].substr(9))};
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
// types (ann = ThesisAnnClassifier, dsnn = spiking; the cosine control has no network). Naively
// that is a 2×2 matrix of near-identical training code. The helpers below factor out everything the
// four cases share, so each CV loop reads as plain control flow:
//   build model → train → evaluate → record fold.
// ─────────────────────────────────────────────────────────────────────────────

// Immutable per-run data shared by both CV strategies. Bundled into one struct so
// the loop helpers take a single argument instead of a dozen. All members are
// references/values valid for the duration of run_classifier.
struct FoldContext
{
    const ThesisDatasetView& view;
    // cppcheck-suppress uninitMemberVarNoCtor ; a reference member is bound by aggregate init
    const std::vector<std::vector<double>>& feature_vectors;
    // cppcheck-suppress uninitMemberVarNoCtor ; a reference member is bound by aggregate init
    const std::vector<int>& labels;
    // cppcheck-suppress uninitMemberVarNoCtor ; a reference member is bound by aggregate init
    const std::vector<int>& groups; // subject_id per sample (for GroupKFold)
    const nn::Tensor& all_inputs;   // (N, feat_dim) feature matrix
    const nn::Tensor& all_targets;  // (N, n_speakers) one-hot labels
    const ThesisConfig& cfg;
    const nn::training::TrainerConfig& trainer_cfg;
    const statistics::IEERScorer& scorer;
    // cppcheck-suppress uninitMemberVarNoCtor ; a reference member is bound by aggregate init
    const std::string& feature_label;
    int feat_dim = 0;
    int hidden_dim = 0;
    int depth = 0;
    int n_speakers = 0;
    uint32_t global_bar_id = 0;
    int* global_completed = nullptr;
    // Simulated dysphonia: the degraded twin of all_inputs (nullptr = clean run), and
    // whether each test speaker's first enrol_per_speaker rows (the scorer's enrolment
    // template) stay clean.
    const nn::Tensor* degraded_inputs = nullptr;
    bool enrol_clean = false;
    std::size_t enrol_per_speaker = 0;
};

// Construct the classifier selected by cfg.classifier.type and invoke fn(model).
// This is the single place that knows the concrete model types, so every CV loop
// stays model-agnostic and works through the generic-lambda parameter `model`.
template <typename Fn>
auto with_classifier(const FoldContext& ctx, Fn&& fn)
{
    if (ctx.cfg.classifier.type == "ann")
    {
        // The DSNN's graph with ReLU for LIF and the DSNN's initial weights.
        ThesisAnnClassifier model(
            ctx.feat_dim, ctx.hidden_dim, ctx.n_speakers, ctx.depth, ctx.cfg.experiment.seed);
        return fn(model);
    }
    // The profile declares the band only when lambda > 0 and alpha only with tdBN on
    // (ThesisConfig refuses the rest), so "off" reaches the model as nullopt.
    const auto& t = ctx.cfg.training;
    std::optional<ThesisDsnnClassifier::FiringRateBand> rate_reg;
    if (t.firing_rate_reg_lambda > 0.0f)
        rate_reg = ThesisDsnnClassifier::FiringRateBand{
            t.firing_rate_reg_lambda, t.firing_rate_min, t.firing_rate_max};
    std::optional<float> tdbn_alpha;
    if (t.batch_normalization == "threshold-dependent") tdbn_alpha = t.tdbn_alpha.get();
    ThesisDsnnClassifier model(ctx.feat_dim,
        ctx.hidden_dim,
        ctx.n_speakers,
        ctx.depth,
        ctx.cfg.experiment.seed,
        rate_reg,
        tdbn_alpha);
    return fn(model);
}

// Build a Trainer for `model`, attach the standard callbacks, and fit. Returns
// the per-epoch learning-curve history (train/val loss, epoch time, SNN spike
// rate + SOPs) so the caller can persist it — the Meeting01-style run diagnostics.
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
// Mirrors the param_count the Meeting01 pipeline records per run.
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
    fr.energy = em.energy;
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
    // No training rows used to yield the identity scaler: the fold then trained on
    // unstandardized features while its profile declared standardize_features = true.
    if (idx.empty())
        throw std::logic_error("ThesisClassifiers: cannot fit the feature scaler on 0 rows");
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

// The test rows of one outer fold. Clean run: the standardized clean rows. Degraded run:
// the degraded rows, standardized with the SAME scaler (fitted on clean training rows);
// with enrol_clean, the first enrol_per_speaker rows of each speaker — the ones the scorer
// averages into the enrolment template, in this same order — keep the clean features.
void gather_test_rows(const FoldContext& ctx,
    const nn::Tensor& Xn,
    const FeatureScaler& scaler,
    const std::vector<size_t>& test_idx,
    std::vector<std::vector<double>>& feats,
    std::vector<int>& labs)
{
    gather_subset_std(Xn, ctx.labels, test_idx, feats, labs);
    if (ctx.degraded_inputs == nullptr) return;
    const nn::Tensor Xd = apply_scaler(*ctx.degraded_inputs, scaler);
    std::map<int, std::size_t> taken;
    const auto D = static_cast<size_t>(Xd.cols());
    for (size_t k = 0; k < test_idx.size(); ++k)
    {
        const bool enrolment = ctx.enrol_clean && taken[labs[k]]++ < ctx.enrol_per_speaker;
        if (enrolment) continue;
        for (size_t j = 0; j < D; ++j)
            feats[k][j] = Xd.at(static_cast<nn::Index>(test_idx[k]), static_cast<nn::Index>(j));
    }
}

// One candidate model trained on one inner fold, plus whether it trained successfully
// (an empty inner-train/val intersection leaves an entry with `valid == false`).
struct InnerCandidate
{
    double score = -std::numeric_limits<double>::infinity();
    std::map<std::string, nn::Tensor> state;
    std::vector<nn::training::EpochResult> history;
    bool valid = false;
};

// Trains one candidate model per inner fold of `outer`, in parallel. Inner folds are
// independent (each trains its own model on its own subset), so they run in an OpenMP
// parallel-for; every candidate's score/state is collected per fold index and returned in
// fold order — this keeps the selected model bit-identical to a serial loop (same
// first-wins tie-break in the caller's reduction) regardless of thread scheduling. Any
// exception raised inside a worker is captured and rethrown once all workers have finished.
std::vector<InnerCandidate> train_inner_candidates(const FoldContext& ctx,
    const statistics::NestedFoldSplit& outer,
    const std::vector<size_t>& outer_train_indices,
    const nn::Tensor& Xn,
    size_t outer_idx,
    int total_outer)
{
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
            const auto inner_val = intersect_indices(inner_ref.test_indices, outer_train_indices);
            // An empty inner fold used to be skipped, so model selection ran on fewer
            // candidates than the k_folds the profile declared, and nothing said so.
            if (inner_train.empty() || inner_val.empty())
                throw std::runtime_error(
                    "ThesisClassifiers: outer fold " + std::to_string(outer_idx) + ", inner fold " +
                    std::to_string(ii) + " has " + std::to_string(inner_train.size()) +
                    " train and " + std::to_string(inner_val.size()) + " validation rows");

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
                    const EvalMetrics m = evaluate(model,
                        val_feats,
                        val_labels,
                        ctx.n_speakers,
                        ctx.scorer,
                        "outer fold " + std::to_string(outer_idx) + ", inner validation fold " +
                            std::to_string(ii));
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

    return candidates;
}

// Reduces the per-inner-fold candidates to the single best (highest validation score),
// first-wins on ties, matching the original serial-loop selection order. Every inner
// fold either trains a candidate or throws (train_inner_candidates), so an untrained
// one here is a bug, not a fold to pass over.
const InnerCandidate& select_best_candidate(const std::vector<InnerCandidate>& candidates)
{
    if (candidates.empty())
        throw std::logic_error("ThesisClassifiers: no inner-CV candidates to select from");
    double best_val_score = -std::numeric_limits<double>::infinity();
    const InnerCandidate* best_cand = nullptr;
    for (const auto& cand : candidates)
    {
        if (!cand.valid)
            throw std::logic_error("ThesisClassifiers: an inner-CV candidate was never trained");
        if (best_cand == nullptr || cand.score > best_val_score)
        {
            best_val_score = cand.score;
            best_cand = &cand;
        }
    }
    return *best_cand;
}

// A fold with no train or no test rows used to be skipped without a word, so a run could
// report fewer folds than its profile's k_folds and still look complete.
void require_fold_rows(const char* cv, size_t outer_idx, size_t n_train, size_t n_test)
{
    if (n_train == 0 || n_test == 0)
        throw std::runtime_error(std::string("ThesisClassifiers: ") + cv + " outer fold " +
                                 std::to_string(outer_idx) + " has " + std::to_string(n_train) +
                                 " train and " + std::to_string(n_test) +
                                 " test rows after the text split");
}

// Processes a single outer fold: trains the inner-CV candidates, selects the best by
// validation score, then scores it once on the held-out outer test fold, appending the
// resulting FoldResult to `result.outer_folds`. A fold without train or test rows throws.
void run_one_outer_fold(const FoldContext& ctx,
    ClassificationResult& result,
    const std::vector<size_t>& text_train_indices,
    const std::vector<size_t>& text_test_indices,
    const statistics::NestedFoldSplit& outer,
    size_t outer_idx,
    int total_outer)
{
    // Outer-train = every sample not held out as outer test, AND in the text-split train
    // set — the same rule as flat CV. Until 2026-11-12 the nested path skipped the text
    // filter, so under text_mode=independent its models trained on the very phrases the
    // test set holds out (pitfall: train and test must use disjoint phrase sets), and the
    // nested and flat columns of the Phase 01 grid did not mean the same condition.
    // A mask, not a binary search: the latter would silently depend on the policy
    // returning the test bucket sorted.
    std::vector<bool> in_outer_test(ctx.view.samples.size(), false);
    for (size_t i : outer.test_indices) in_outer_test.at(i) = true;
    std::vector<size_t> not_outer_test;
    not_outer_test.reserve(ctx.view.samples.size() - outer.test_indices.size());
    for (size_t i = 0; i < ctx.view.samples.size(); ++i)
        if (!in_outer_test[i]) not_outer_test.push_back(i);
    const auto outer_train_indices = intersect_indices(not_outer_test, text_train_indices);

    // Keep only test samples whose phrase belongs to the text-split test set.
    const auto outer_test_indices = intersect_indices(outer.test_indices, text_test_indices);
    require_fold_rows("nested", outer_idx, outer_train_indices.size(), outer_test_indices.size());

    // Standardize features with statistics fit on the outer-train rows only,
    // so the outer test fold never leaks into the scaler (audit G1). One
    // scaler per outer fold keeps the model and its test set on the same scale.
    const FeatureScaler scaler = make_fold_scaler(ctx, outer_train_indices);
    const nn::Tensor Xn = apply_scaler(ctx.all_inputs, scaler);

    std::vector<std::vector<double>> test_feats;
    std::vector<int> test_labels;
    gather_test_rows(ctx, Xn, scaler, outer_test_indices, test_feats, test_labels);

    // ── Inner loop: train one candidate per inner fold, keep the best ──
    const auto train_t0 = std::chrono::steady_clock::now();
    const std::vector<InnerCandidate> candidates =
        train_inner_candidates(ctx, outer, outer_train_indices, Xn, outer_idx, total_outer);
    const InnerCandidate* best_cand = &select_best_candidate(candidates);

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
            const EvalMetrics m = evaluate(model,
                test_feats,
                test_labels,
                ctx.n_speakers,
                ctx.scorer,
                "nested outer test fold " + std::to_string(outer_idx));
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

// Nested k-fold CV: an inner loop selects the best model by validation score; the
// selected model is then scored once on the held-out outer test fold. Gives an
// unbiased performance estimate when hyperparameters are tuned on the inner folds.
void run_nested_cv(const FoldContext& ctx,
    ClassificationResult& result,
    const std::vector<size_t>& text_train_indices,
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
        run_one_outer_fold(ctx,
            result,
            text_train_indices,
            text_test_indices,
            nested_splits[outer_idx],
            outer_idx,
            total_outer);
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
        require_fold_rows("flat", outer_idx, train_indices.size(), test_indices.size());

        // Fit the scaler on this fold's train rows only (audit G1).
        const FeatureScaler scaler = make_fold_scaler(ctx, train_indices);
        const nn::Tensor Xn = apply_scaler(ctx.all_inputs, scaler);

        std::vector<std::vector<double>> test_feats;
        std::vector<int> test_labels;
        gather_test_rows(ctx, Xn, scaler, test_indices, test_feats, test_labels);

        FoldResult fr;
        fr.fold = static_cast<int>(outer_idx);
        if (ctx.cfg.classifier.type == "cosine")
        {
            // No network: the standardized features are the embedding. The training rows
            // serve only the scaler above.
            const auto infer_t0 = std::chrono::steady_clock::now();
            std::vector<std::vector<float>> emb(test_feats.size());
            for (size_t i = 0; i < test_feats.size(); ++i)
                emb[i].assign(test_feats[i].begin(), test_feats[i].end());
            const EvalMetrics em = score_embeddings(emb,
                test_labels,
                ctx.n_speakers,
                ctx.scorer,
                "flat test fold " + std::to_string(outer_idx) + " (cosine)");
            fr.train_ms = 0.0;
            fr.infer_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - infer_t0)
                              .count();
            set_fold_metrics(fr, em);
            result.outer_folds.push_back(fr);
            advance_global_bar(ctx);
            continue;
        }
        const auto train_pairs = make_pairs_from_indices(Xn, ctx.all_targets, train_indices);
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
                const EvalMetrics m = evaluate(model,
                    test_feats,
                    test_labels,
                    ctx.n_speakers,
                    ctx.scorer,
                    "flat test fold " + std::to_string(outer_idx));
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
