#include "ThesisClassifiers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ThesisClassifiersInternal.hpp"
#include "statistics/GenuineImpostorEERScorer.hpp"
#include "statistics/eer_scorer.hpp"

namespace thesis
{

auto run_classifier(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer* eer_scorer,
    uint32_t global_bar_id,
    int* global_completed) -> ClassificationResult
{
    if (feature_vectors.empty())
        throw std::invalid_argument("ThesisClassifiers: empty feature vectors");
    if (feature_vectors.size() != view.samples.size())
        throw std::invalid_argument("ThesisClassifiers: features/samples size mismatch");

    if (cfg.classifier.type != "rnn" && cfg.classifier.type != "dsnn")
        throw std::invalid_argument("ThesisClassifiers: classifier type \"" + cfg.classifier.type +
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
    trainer_cfg.grad_clip_norm = cfg.training.gradient_clip_norm;
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

    // Count trainable parameters once on a fresh model (identical across folds) —
    // the Guayaquil-style model-complexity stat.
    with_classifier(ctx, [&](auto& model) { result.param_count = count_trainable_params(model); });

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

    // Run-cost stats: mean per-fold train/infer wall-clock, and (SNN) the last-epoch
    // firing rate + synaptic-op count of each fold's final model. SOPs are ~constant
    // across folds (same architecture), so the last fold's value is representative.
    double sum_train = 0.0, sum_infer = 0.0, sum_rate = 0.0;
    int n_rate = 0;
    for (const auto& f : result.outer_folds)
    {
        sum_train += f.train_ms;
        sum_infer += f.infer_ms;
        if (!f.history.empty())
        {
            const auto& last = f.history.back();
            if (!std::isnan(last.mean_spike_rate))
            {
                sum_rate += static_cast<double>(last.mean_spike_rate);
                ++n_rate;
            }
            result.final_sops = last.sops;
        }
    }
    const auto n_folds = static_cast<double>(result.outer_folds.size());
    result.mean_train_ms = sum_train / n_folds;
    result.mean_infer_ms = sum_infer / n_folds;
    result.mean_spike_rate =
        n_rate > 0 ? sum_rate / n_rate : std::numeric_limits<double>::quiet_NaN();
}

} // namespace thesis
