#include "E05Classifiers.hpp"

#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "Backend.hpp"
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

namespace e05
{

namespace
{

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

// Single batched forward on (N, D); computes all metrics.
EvalMetrics evaluate(SimpleResNetImpl<nn::Backend>& model,
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

    int correct = 0;
    std::vector<int> preds(static_cast<size_t>(N));
    std::vector<std::vector<float>> embeddings(static_cast<size_t>(N));

    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    {
        embeddings[i].resize(static_cast<size_t>(n_classes));
        int pred   = 0;
        float best = all_logits.at(static_cast<nn::Index>(i), 0);
        for (int j = 0; j < n_classes; ++j)
        {
            float v = all_logits.at(static_cast<nn::Index>(i), j);
            embeddings[i][static_cast<size_t>(j)] = v;
            if (v > best) { best = v; pred = j; }
        }
        preds[i] = pred;
        if (pred == labels[i]) ++correct;
    }

    // Per-class TP, FP, FN, TN for macro metrics.
    std::vector<int> tp(static_cast<size_t>(n_classes), 0);
    std::vector<int> fp(static_cast<size_t>(n_classes), 0);
    std::vector<int> fn(static_cast<size_t>(n_classes), 0);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    {
        const int p = preds[i];
        const int t = labels[i];
        if (p == t)
            ++tp[static_cast<size_t>(p)];
        else
        {
            ++fp[static_cast<size_t>(p)];
            ++fn[static_cast<size_t>(t)];
        }
    }

    double sum_p = 0.0, sum_r = 0.0, sum_f1 = 0.0, sum_spec = 0.0;
    const int total = static_cast<int>(N);
    for (int c = 0; c < n_classes; ++c)
    {
        const int tn_c = total - tp[c] - fp[c] - fn[c];

        const double prec_c = (tp[c] + fp[c] > 0)
            ? static_cast<double>(tp[c]) / static_cast<double>(tp[c] + fp[c])
            : 0.0;
        const double rec_c  = (tp[c] + fn[c] > 0)
            ? static_cast<double>(tp[c]) / static_cast<double>(tp[c] + fn[c])
            : 0.0;
        const double spec_c = (tn_c + fp[c] > 0)
            ? static_cast<double>(tn_c) / static_cast<double>(tn_c + fp[c])
            : 0.0;

        sum_p    += prec_c;
        sum_r    += rec_c;
        sum_spec += spec_c;
        sum_f1   += (prec_c + rec_c > 0.0)
            ? (2.0 * prec_c * rec_c / (prec_c + rec_c))
            : 0.0;
    }
    const double nc = static_cast<double>(n_classes);

    EvalMetrics m;
    m.accuracy    = static_cast<double>(correct) / static_cast<double>(N);
    m.precision   = sum_p    / nc;
    m.recall      = sum_r    / nc;
    m.specificity = sum_spec / nc;
    m.f1          = sum_f1   / nc;
    m.eer         = eer_scorer.compute_eer(embeddings, labels, n_classes);
    m.auc         = eer_scorer.compute_auc(embeddings, labels, n_classes);
    return m;
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

    if (cfg.classifier.type != "rnn")
        throw std::invalid_argument(
            "E05Classifiers: classifier type \"" + cfg.classifier.type +
            "\" is not implemented. Supported: \"rnn\".");

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

            std::vector<std::vector<double>> test_feats;
            std::vector<int> test_labels;
            for (size_t idx : outer.test_indices)
            {
                test_feats.push_back(feature_vectors[idx]);
                test_labels.push_back(labels[idx]);
            }

            SimpleResNetImpl<nn::Backend> model(feat_dim, hidden_dim, n_speakers, depth);
            CrossEntropyLossImpl<nn::Backend> loss_fn;
            nn::training::Trainer<SimpleResNetImpl<nn::Backend>,
                CrossEntropyLossImpl<nn::Backend>> trainer(model, trainer_cfg, loss_fn);

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

            using SamplePair = std::pair<nn::Tensor, nn::Tensor>;
            std::vector<SamplePair> train_pairs;
            train_pairs.reserve(outer.inner_splits[0].train_indices.size());
            for (size_t idx : outer.inner_splits[0].train_indices)
                train_pairs.emplace_back(all_inputs.row(static_cast<nn::Index>(idx)),
                    all_targets.row(static_cast<nn::Index>(idx)));

            std::vector<SamplePair> val_pairs;
            val_pairs.reserve(outer.inner_splits[0].test_indices.size());
            for (size_t idx : outer.inner_splits[0].test_indices)
                val_pairs.emplace_back(all_inputs.row(static_cast<nn::Index>(idx)),
                    all_targets.row(static_cast<nn::Index>(idx)));

            trainer.fit_supervised(train_pairs, val_pairs);

            statistics::GenuineImpostorEERScorer default_scorer;
            const statistics::IEERScorer& scorer =
                (eer_scorer != nullptr) ? *eer_scorer : default_scorer;

            auto em = evaluate(model, test_feats, test_labels, n_speakers, scorer);

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
                nn::io::save_state_dict(model.state_dict(), fr.model_path);
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

            std::vector<std::vector<double>> test_feats;
            std::vector<int> test_labels;
            for (size_t idx : split.test_indices)
            {
                test_feats.push_back(feature_vectors[idx]);
                test_labels.push_back(labels[idx]);
            }

            SimpleResNetImpl<nn::Backend> model(feat_dim, hidden_dim, n_speakers, depth);
            CrossEntropyLossImpl<nn::Backend> loss_fn;
            nn::training::Trainer<SimpleResNetImpl<nn::Backend>,
                CrossEntropyLossImpl<nn::Backend>> trainer(model, trainer_cfg, loss_fn);

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
            // Early stopping disabled without inner validation set.

            using SamplePair = std::pair<nn::Tensor, nn::Tensor>;
            std::vector<SamplePair> train_pairs;
            train_pairs.reserve(split.train_indices.size());
            for (size_t idx : split.train_indices)
                train_pairs.emplace_back(all_inputs.row(static_cast<nn::Index>(idx)),
                    all_targets.row(static_cast<nn::Index>(idx)));

            // No validation set — pass empty; Trainer handles it gracefully.
            trainer.fit_supervised(train_pairs, {});

            statistics::GenuineImpostorEERScorer default_scorer;
            const statistics::IEERScorer& scorer =
                (eer_scorer != nullptr) ? *eer_scorer : default_scorer;

            auto em = evaluate(model, test_feats, test_labels, n_speakers, scorer);

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
                nn::io::save_state_dict(model.state_dict(), fr.model_path);
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

    const double n = static_cast<double>(result.outer_folds.size());

    double sum_acc  = 0.0, sum_f1 = 0.0, sum_p   = 0.0;
    double sum_r    = 0.0, sum_sp = 0.0;
    double sum_eer  = 0.0, sum_auc = 0.0;
    int n_eer = 0, n_auc = 0;

    for (const auto& f : result.outer_folds)
    {
        sum_acc += f.accuracy;
        sum_f1  += f.f1;
        sum_p   += f.precision;
        sum_r   += f.recall;
        sum_sp  += f.specificity;
        if (!std::isnan(f.eer)) { sum_eer += f.eer; ++n_eer; }
        if (!std::isnan(f.auc)) { sum_auc += f.auc; ++n_auc; }
    }

    result.mean_accuracy    = sum_acc / n;
    result.mean_f1          = sum_f1  / n;
    result.mean_precision   = sum_p   / n;
    result.mean_recall      = sum_r   / n;
    result.mean_specificity = sum_sp  / n;
    result.mean_eer = (n_eer > 0) ? sum_eer / n_eer : std::numeric_limits<double>::quiet_NaN();
    result.mean_auc = (n_auc > 0) ? sum_auc / n_auc : std::numeric_limits<double>::quiet_NaN();

    double var_acc  = 0.0, var_f1   = 0.0, var_sp  = 0.0;
    double var_eer  = 0.0, var_auc  = 0.0;
    for (const auto& f : result.outer_folds)
    {
        double da = f.accuracy    - result.mean_accuracy;
        double df = f.f1          - result.mean_f1;
        double ds = f.specificity - result.mean_specificity;
        var_acc += da * da;
        var_f1  += df * df;
        var_sp  += ds * ds;
        if (!std::isnan(f.eer) && !std::isnan(result.mean_eer))
        {
            double de = f.eer - result.mean_eer;
            var_eer += de * de;
        }
        if (!std::isnan(f.auc) && !std::isnan(result.mean_auc))
        {
            double du = f.auc - result.mean_auc;
            var_auc += du * du;
        }
    }
    result.std_accuracy    = std::sqrt(var_acc / n);
    result.std_f1          = std::sqrt(var_f1  / n);
    result.std_specificity = std::sqrt(var_sp  / n);
    result.std_eer         = (n_eer > 1) ? std::sqrt(var_eer / n_eer) : 0.0;
    result.std_auc         = (n_auc > 1) ? std::sqrt(var_auc / n_auc) : 0.0;

    const double inv_sqrt_n  = 1.0 / std::sqrt(n);
    result.ci95_accuracy = 1.96 * result.std_accuracy * inv_sqrt_n;
    result.ci95_eer      = 1.96 * result.std_eer      * inv_sqrt_n;
}

} // namespace e05
