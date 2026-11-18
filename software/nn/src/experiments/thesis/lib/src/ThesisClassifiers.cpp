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

namespace
{

void require_supported_type(const std::string& type)
{
    if (type != "ann" && type != "dsnn" && type != "cosine")
        throw std::invalid_argument("ThesisClassifiers: classifier type \"" + type +
                                    "\" is not implemented. Supported: dsnn, ann, cosine.");
}

// Everything a run derives from (view, features, cfg) before its first fold, owned here so the
// FoldContext built over it (references only) stays valid for the whole run.
struct PreparedRun
{
    std::vector<int> labels; // sequential class index per sample
    std::vector<int> groups; // subject_id per sample (GroupKFold)
    nn::training::TrainerConfig trainer_cfg;
    nn::Tensor all_inputs;
    nn::Tensor all_targets;
    std::optional<nn::Tensor> degraded_inputs;
    int feat_dim = 0;
    int hidden_dim = 0;
    int depth = 0;
    int n_speakers = 0;
};

auto prepare_run(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const ThesisConfig& cfg,
    const DegradedTestFeatures* degraded) -> PreparedRun
{
    if (feature_vectors.empty())
        throw std::invalid_argument("ThesisClassifiers: empty feature vectors");
    if (feature_vectors.size() != view.samples.size())
        throw std::invalid_argument("ThesisClassifiers: features/samples size mismatch");

    const std::string& type = cfg.classifier.type;
    require_supported_type(type);

    PreparedRun run;
    run.n_speakers = view.n_subjects;
    run.feat_dim = static_cast<int>(feature_vectors[0].size());

    // The cosine control builds no network: no layer_spec, shape unused.
    const auto shape =
        type == "cosine" ? ClassifierShape{0, 0} : parse_layer_spec(cfg.classifier.layer_spec);
    run.hidden_dim = shape.hidden_dim;
    run.depth = shape.depth;

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

    run.labels.reserve(view.samples.size());
    for (const auto& s : view.samples) run.labels.push_back(id_to_class[s.subject_id]);

    // The cosine control trains nothing and declares no training keys.
    if (type != "cosine")
    {
        auto& tc = run.trainer_cfg;
        tc.epochs = cfg.training.epochs;
        tc.learning_rate = cfg.training.learning_rate;
        tc.optimizer_type = cfg.training.optimizer_type;
        // Momentum exists for sgd only; the profile declares it exactly then.
        if (cfg.training.optimizer_type == "sgd")
            tc.optimizer_momentum = cfg.training.optimizer_momentum;
        tc.grad_clip_norm = cfg.training.gradient_clip_norm;
        tc.batch_size = cfg.training.samples_per_batch;
        tc.weight_decay = cfg.training.weight_decay; // decoupled L2 (ann + dsnn)
    }

    run.groups.reserve(view.samples.size());
    for (const auto& s : view.samples) run.groups.push_back(s.subject_id);

    // Pre-build full dataset tensors once — avoid per-fold reconstruction.
    const auto N_all = static_cast<nn::Index>(feature_vectors.size());
    const auto D = static_cast<nn::Index>(run.feat_dim);
    const auto C = static_cast<nn::Index>(run.n_speakers);

    // Degraded test features: same rows, same width as the clean ones.
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
        run.degraded_inputs = std::move(x);
    }

    run.all_inputs = nn::Tensor::zeros(N_all, D);
    run.all_targets = nn::Tensor::zeros(N_all, C);
    {
        float* inp_ptr = run.all_inputs.mutable_data_ptr();
        float* tgt_ptr = run.all_targets.mutable_data_ptr();
        for (nn::Index i = 0; i < N_all; ++i)
        {
            const auto& fv = feature_vectors[static_cast<size_t>(i)];
            if (static_cast<nn::Index>(fv.size()) != D)
                throw std::invalid_argument(
                    "ThesisClassifiers: a feature vector has another width");
            for (nn::Index j = 0; j < D; ++j)
                inp_ptr[i * D + j] = static_cast<float>(fv[static_cast<size_t>(j)]);
            tgt_ptr[i * C + run.labels[static_cast<size_t>(i)]] = 1.0f;
        }
    }
    return run;
}

auto make_context(const PreparedRun& run,
    const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& scorer,
    const std::string& feature_label,
    uint32_t global_bar_id,
    int* global_completed,
    const DegradedTestFeatures* degraded) -> FoldContext
{
    return FoldContext{view,
        feature_vectors,
        run.labels,
        run.groups,
        run.all_inputs,
        run.all_targets,
        cfg,
        run.trainer_cfg,
        scorer,
        feature_label,
        run.feat_dim,
        run.hidden_dim,
        run.depth,
        run.n_speakers,
        global_bar_id,
        global_completed,
        run.degraded_inputs ? &*run.degraded_inputs : nullptr,
        degraded != nullptr && degraded->enrol_clean,
        degraded != nullptr ? degraded->enrol_per_speaker : 0};
}

auto params_of(const FoldContext& ctx) -> std::size_t
{
    // Counted on a fresh model (the count does not depend on the weights). The cosine
    // control has none.
    if (ctx.cfg.classifier.type == "cosine") return 0;
    return with_classifier(ctx, [](auto& model) { return count_trainable_params(model); });
}

} // namespace

auto run_classifier(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer,
    uint32_t global_bar_id,
    int* global_completed,
    const DegradedTestFeatures* degraded) -> ClassificationResult
{
    const std::string& type = cfg.classifier.type;
    require_supported_type(type);
    if (type == "cosine" && cfg.training.nested_cv)
        throw std::invalid_argument(
            "ThesisClassifiers: the cosine control has no model to select; "
            "it runs flat CV only");

    const PreparedRun run = prepare_run(view, feature_vectors, cfg, degraded);
    const std::size_t k = static_cast<std::size_t>(cfg.training.k_folds);

    ClassificationResult result;
    result.feature_set_label = feature_label;
    result.classifier_type = cfg.classifier.type;
    result.text_mode = cfg.classifier.text_mode;

    const auto text_split =
        make_text_split(view.samples, cfg.classifier.text_mode, cfg.experiment.seed);
    std::vector<size_t> text_train_indices = text_split.train_indices;
    std::vector<size_t> text_test_indices = text_split.test_indices;
    std::sort(text_train_indices.begin(), text_train_indices.end());
    std::sort(text_test_indices.begin(), text_test_indices.end());

    // The caller's EER scorer, used both for inner-fold model selection and for
    // reporting, shared by every fold via FoldContext.
    const FoldContext ctx = make_context(run,
        view,
        feature_vectors,
        cfg,
        eer_scorer,
        feature_label,
        global_bar_id,
        global_completed,
        degraded);

    // Count trainable parameters once on a fresh model (identical across folds) —
    // the Meeting01-style model-complexity stat.
    result.param_count = params_of(ctx);

    if (cfg.training.nested_cv)
        run_nested_cv(ctx, result, text_train_indices, text_test_indices, k);
    else
        run_flat_cv(ctx, result, text_train_indices, text_test_indices, k);

    compute_aggregate_stats(result);
    return result;
}

auto run_split(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer,
    const std::vector<std::size_t>& train_rows,
    const std::vector<std::size_t>& test_rows,
    const SplitRun& split) -> FoldResult
{
    if (split.total_folds < 1 || split.fold < 0 || split.fold >= split.total_folds)
        throw std::invalid_argument("ThesisClassifiers: split " + std::to_string(split.fold) +
                                    " of " + std::to_string(split.total_folds) +
                                    " is not a fold of the run");
    const PreparedRun run = prepare_run(view, feature_vectors, cfg, split.degraded);
    const FoldContext ctx = make_context(run,
        view,
        feature_vectors,
        cfg,
        eer_scorer,
        feature_label,
        split.global_bar_id,
        nullptr,
        split.degraded);
    return run_flat_fold(ctx,
        train_rows,
        test_rows,
        static_cast<size_t>(split.fold),
        split.total_folds,
        split.save_model);
}

auto classifier_param_count(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer) -> std::size_t
{
    const PreparedRun run = prepare_run(view, feature_vectors, cfg, nullptr);
    const std::string label = "param-count";
    return params_of(
        make_context(run, view, feature_vectors, cfg, eer_scorer, label, 0, nullptr, nullptr));
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
