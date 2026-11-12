#include "ThesisClassifiers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "ThesisClassifiersInternal.hpp"
#include "statistics/eer_scorer.hpp"

namespace thesis
{

auto run_classifier(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer,
    uint32_t global_bar_id,
    int* global_completed,
    const DegradedTestFeatures* degraded) -> ClassificationResult
{
    if (feature_vectors.empty())
        throw std::invalid_argument("ThesisClassifiers: empty feature vectors");
    if (feature_vectors.size() != view.samples.size())
        throw std::invalid_argument("ThesisClassifiers: features/samples size mismatch");

    const std::string& type = cfg.classifier.type;
    if (type != "ann" && type != "dsnn" && type != "cosine")
        throw std::invalid_argument("ThesisClassifiers: classifier type \"" + type +
                                    "\" is not implemented. Supported: dsnn, ann, cosine.");
    if (type == "cosine" && cfg.training.nested_cv)
        throw std::invalid_argument(
            "ThesisClassifiers: the cosine control has no model to select; "
            "it runs flat CV only");

    const int n_speakers = view.n_subjects;
    const int feat_dim = static_cast<int>(feature_vectors[0].size());

    // The cosine control builds no network: no layer_spec, shape unused.
    const auto [hidden_dim, depth] =
        type == "cosine" ? ClassifierShape{0, 0} : parse_layer_spec(cfg.classifier.layer_spec);

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

    // The cosine control trains nothing and declares no training keys.
    nn::training::TrainerConfig trainer_cfg;
    if (type != "cosine")
    {
        trainer_cfg.epochs = cfg.training.epochs;
        trainer_cfg.learning_rate = cfg.training.learning_rate;
        trainer_cfg.optimizer_type = cfg.training.optimizer_type;
        // Momentum exists for sgd only; the profile declares it exactly then.
        if (cfg.training.optimizer_type == "sgd")
            trainer_cfg.optimizer_momentum = cfg.training.optimizer_momentum;
        trainer_cfg.grad_clip_norm = cfg.training.gradient_clip_norm;
        trainer_cfg.batch_size = cfg.training.samples_per_batch;
        trainer_cfg.weight_decay = cfg.training.weight_decay; // decoupled L2 (ann + dsnn)
    }

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

    // Degraded test features: same rows, same width as the clean ones.
    std::optional<nn::Tensor> degraded_inputs;
    if (degraded != nullptr)
    {
        if (degraded->vectors == nullptr || degraded->vectors->size() != feature_vectors.size())
            throw std::invalid_argument(
                "ThesisClassifiers: the degraded features are not aligned with the samples");
        if (degraded->enrol_clean && degraded->enrol_per_speaker == 0)
            throw std::invalid_argument(
                "ThesisClassifiers: scope \"probes\" needs the scorer's enrolment count");
        nn::Tensor x = nn::Tensor::zeros(N_all, D);
        float* p = x.mutable_data_ptr();
        for (nn::Index i = 0; i < N_all; ++i)
        {
            const auto& fv = (*degraded->vectors)[static_cast<size_t>(i)];
            if (static_cast<nn::Index>(fv.size()) != D)
                throw std::invalid_argument(
                    "ThesisClassifiers: a degraded feature vector has another width");
            for (nn::Index j = 0; j < D; ++j)
                p[i * D + j] = static_cast<float>(fv[static_cast<size_t>(j)]);
        }
        degraded_inputs = std::move(x);
    }

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

    // The caller's EER scorer, used both for inner-fold model selection and for
    // reporting, shared by every fold via FoldContext.
    const statistics::IEERScorer& scorer = eer_scorer;

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
        global_completed,
        degraded_inputs ? &*degraded_inputs : nullptr,
        degraded != nullptr && degraded->enrol_clean,
        degraded != nullptr ? degraded->enrol_per_speaker : 0};

    // Count trainable parameters once on a fresh model (identical across folds) —
    // the Meeting01-style model-complexity stat. The cosine control has none.
    if (type == "cosine")
        result.param_count = 0;
    else
        with_classifier(
            ctx, [&](auto& model) { result.param_count = count_trainable_params(model); });

    if (cfg.training.nested_cv)
        run_nested_cv(ctx, result, text_train_indices, text_test_indices, k);
    else
        run_flat_cv(ctx, result, text_train_indices, text_test_indices, k);

    compute_aggregate_stats(result);
    return result;
}

void compute_aggregate_stats(ClassificationResult& result)
{
    // A result with no folds used to come back with every statistic at its initial
    // value, as if it had been aggregated.
    if (result.outer_folds.empty())
        throw std::logic_error("ThesisClassifiers: \"" + result.feature_set_label +
                               "\" has no outer folds to aggregate");

    // Collect per-fold values, then aggregate each metric with sample SD and a t-based
    // 95% CI (audit M-3). Under verification-only (audit C-1) the closed-set metrics are
    // NaN in every fold and aggregate to NaN; any other NaN throws (aggregate_metric).
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
    //
    // Every fold trained, so every fold has a history. The firing rate is NaN in every
    // fold of a non-spiking classifier (ann: not applicable) and finite in every fold of
    // the DSNN; a mix used to be averaged over the folds that had it.
    double sum_train = 0.0, sum_infer = 0.0;
    std::vector<double> rates;
    rates.reserve(result.outer_folds.size());
    if (result.classifier_type == "cosine")
    {
        // Nothing trains: no history, no firing rate, no training time.
        for (const auto& f : result.outer_folds)
        {
            if (!f.history.empty())
                throw std::logic_error(
                    "ThesisClassifiers: the cosine control has a training "
                    "history in fold " +
                    std::to_string(f.fold));
            sum_infer += f.infer_ms;
        }
        result.mean_train_ms = 0.0;
        result.mean_infer_ms = sum_infer / static_cast<double>(result.outer_folds.size());
        result.mean_spike_rate = std::numeric_limits<double>::quiet_NaN();
        for (const auto& f : result.outer_folds)
            if (f.energy)
                throw std::logic_error(
                    "ThesisClassifiers: the cosine control has an energy in "
                    "fold " +
                    std::to_string(f.fold));
        return;
    }
    for (const auto& f : result.outer_folds)
    {
        if (f.history.empty())
            throw std::logic_error("ThesisClassifiers: \"" + result.feature_set_label + "\" fold " +
                                   std::to_string(f.fold) + " has no training history");
        sum_train += f.train_ms;
        sum_infer += f.infer_ms;
        const auto& last = f.history.back();
        rates.push_back(static_cast<double>(last.mean_spike_rate));
        result.final_sops = last.sops;
    }
    // Every network fold measured its test pass; the mean is per field.
    InferenceEnergy mean_e;
    for (const auto& f : result.outer_folds)
    {
        if (!f.energy)
            throw std::logic_error("ThesisClassifiers: \"" + result.feature_set_label + "\" fold " +
                                   std::to_string(f.fold) + " has no measured energy");
        const auto& e = *f.energy;
        mean_e.nj += e.nj;
        mean_e.ops_nj += e.ops_nj;
        mean_e.mem_nj += e.mem_nj;
        mean_e.nj_as_executed += e.nj_as_executed;
        mean_e.nj_sram8k += e.nj_sram8k;
        mean_e.nj_sram32k += e.nj_sram32k;
        mean_e.nj_sram1m += e.nj_sram1m;
        mean_e.fp_macs += e.fp_macs;
        mean_e.fp_adds += e.fp_adds;
        mean_e.accesses += e.accesses;
        mean_e.spikes += e.spikes;
    }
    const double inv = 1.0 / static_cast<double>(result.outer_folds.size());
    for (double* v : {&mean_e.nj,
             &mean_e.ops_nj,
             &mean_e.mem_nj,
             &mean_e.nj_as_executed,
             &mean_e.nj_sram8k,
             &mean_e.nj_sram32k,
             &mean_e.nj_sram1m,
             &mean_e.fp_macs,
             &mean_e.fp_adds,
             &mean_e.accesses,
             &mean_e.spikes})
        *v *= inv;
    result.mean_energy = mean_e;
    const auto n_folds = static_cast<double>(result.outer_folds.size());
    result.mean_train_ms = sum_train / n_folds;
    result.mean_infer_ms = sum_infer / n_folds;
    result.mean_spike_rate = aggregate_metric(rates).mean;
    // The DSNN records a firing rate every epoch; NaN in every fold would otherwise pass as
    // "not applicable", which is only true of the non-spiking ann.
    if (result.classifier_type == "dsnn" && std::isnan(result.mean_spike_rate))
        throw std::logic_error("ThesisClassifiers: \"" + result.feature_set_label +
                               "\" is a dsnn run without a recorded firing rate");
}

} // namespace thesis
