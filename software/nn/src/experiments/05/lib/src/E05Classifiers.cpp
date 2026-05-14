#include "E05Classifiers.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "Backend.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
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
// Single batched forward on (N, D); accuracy via argmax, EER via eer_scorer.
std::pair<double, double> evaluate(SimpleResNetImpl<nn::Backend>& model,
    const std::vector<std::vector<double>>& inputs,
    const std::vector<int>& labels,
    int n_classes,
    const statistics::IEERScorer& eer_scorer)
{
    if (inputs.empty()) return {0.0, 0.0};

    // Stack all test samples into one (N, D) tensor — one GPU kernel instead of N.
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

    nn::Tensor all_logits = model.forward(batch, false); // (N, n_classes)

    // Extract logit rows for accuracy (argmax) and EER scoring.
    int correct = 0;
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
        if (pred == labels[i]) ++correct;
    }

    double accuracy = static_cast<double>(correct) / static_cast<double>(N);
    double eer = eer_scorer.compute_eer(embeddings, labels, n_classes);
    return {accuracy, eer};
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

    const int n_speakers = view.n_subjects;
    const int feat_dim   = static_cast<int>(feature_vectors[0].size());

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

    // Build label vector.
    std::vector<int> labels;
    labels.reserve(view.samples.size());
    for (const auto& s : view.samples)
        labels.push_back(id_to_class[s.subject_id]);

    // Parser profile training config → TrainerConfig.
    nn::training::TrainerConfig trainer_cfg;
    trainer_cfg.epochs           = cfg.training.epochs;
    trainer_cfg.learning_rate    = cfg.training.learning_rate;
    trainer_cfg.batch_size       = cfg.training.samples_per_batch;

    // Build group labels: one speaker ID per sample — keeps all utterances
    // from the same speaker in the same fold (prevents data leakage).
    std::vector<int> groups;
    groups.reserve(view.samples.size());
    for (const auto& s : view.samples)
        groups.push_back(s.subject_id);

    const std::size_t k = static_cast<std::size_t>(cfg.training.k_folds);
    auto outer_policy = std::make_shared<statistics::GroupKFoldPolicy>(
        k, true, cfg.experiment.seed);
    auto inner_policy = std::make_shared<statistics::GroupKFoldPolicy>(
        k, true, cfg.experiment.seed ^ 0xDEADBEEFU);

    statistics::NestedKFold nkf(k, k, outer_policy, inner_policy);
    auto nested_splits = nkf.split(view.samples.size(), groups);

    ClassificationResult result;
    result.feature_set_label = feature_label;
    result.classifier_type   = cfg.classifier.type;
    result.text_mode         = cfg.classifier.text_mode;

    const int hidden_dim = 128;
    const int depth      = 2; // residual blocks

    // Build full dataset tensors once — avoid per-fold reconstruction.
    const auto N_all  = static_cast<nn::Index>(feature_vectors.size());
    const auto D      = static_cast<nn::Index>(feat_dim);
    const auto C      = static_cast<nn::Index>(n_speakers);

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

    const int total_outer = static_cast<int>(nested_splits.size());

    for (size_t outer_idx = 0; outer_idx < nested_splits.size(); ++outer_idx)
    {
        const auto& outer = nested_splits[outer_idx];

        // Collect test data (still as double vectors for evaluate()).
        std::vector<std::vector<double>> test_feats;
        std::vector<int> test_labels;
        for (size_t idx : outer.test_indices)
        {
            test_feats.push_back(feature_vectors[idx]);
            test_labels.push_back(labels[idx]);
        }

        // Build and train classifier on the training fold.
        SimpleResNetImpl<nn::Backend> model(feat_dim, hidden_dim, n_speakers, depth);
        CrossEntropyLossImpl<nn::Backend> loss_fn;
        nn::training::Trainer<SimpleResNetImpl<nn::Backend>,
            CrossEntropyLossImpl<nn::Backend>> trainer(model, trainer_cfg, loss_fn);

        // Per-fold progress bars (epoch + batch).
        const int fold_num = static_cast<int>(outer_idx) + 1;
        auto prog_cb = std::make_shared<nn::training::ProgressCallback>(
            "Fold " + std::to_string(fold_num) + "/" + std::to_string(total_outer) +
            " | " + feature_label);
        prog_cb->set_metadata(feature_label, fold_num, total_outer, "CrossEntropy");
        trainer.add_callback(prog_cb);

        // Early stopping via inner validation fold.
        if (cfg.training.early_stop_patience > 0)
        {
            auto stopper = std::make_shared<nn::training::EarlyStoppingCallback>(
                cfg.training.early_stop_patience);
            trainer.add_callback(stopper);
        }

        // Slice pre-built tensors by fold index — no per-fold tensor construction.
        using SamplePair = std::pair<nn::Tensor, nn::Tensor>;
        std::vector<SamplePair> train_pairs;
        train_pairs.reserve(outer.inner_splits[0].train_indices.size());
        for (size_t idx : outer.inner_splits[0].train_indices)
        {
            train_pairs.emplace_back(all_inputs.row(static_cast<nn::Index>(idx)),
                all_targets.row(static_cast<nn::Index>(idx)));
        }

        std::vector<SamplePair> val_pairs;
        val_pairs.reserve(outer.inner_splits[0].test_indices.size());
        for (size_t idx : outer.inner_splits[0].test_indices)
        {
            val_pairs.emplace_back(all_inputs.row(static_cast<nn::Index>(idx)),
                all_targets.row(static_cast<nn::Index>(idx)));
        }

        trainer.fit_supervised(train_pairs, val_pairs);

        // Default scorer: genuine/impostor cosine-similarity (SOTA protocol).
        statistics::GenuineImpostorEERScorer default_scorer;
        const statistics::IEERScorer& scorer =
            (eer_scorer != nullptr) ? *eer_scorer : default_scorer;

        auto [acc, eer] = evaluate(model, test_feats, test_labels, n_speakers, scorer);

        FoldResult fr;
        fr.fold     = static_cast<int>(outer_idx);
        fr.accuracy = acc;
        fr.eer      = eer;
        result.outer_folds.push_back(fr);

        // Advance global bar after each completed outer fold.
        if (global_bar_id != 0 && global_completed != nullptr)
        {
            nn::progress::ProgressManager::instance().update_bar(
                global_bar_id, static_cast<float>(++(*global_completed)));
        }
    }

    compute_aggregate_stats(result);
    return result;
}

void compute_aggregate_stats(ClassificationResult& result)
{
    if (result.outer_folds.empty()) return;

    double sum_acc = 0.0;
    double sum_eer = 0.0;
    for (const auto& f : result.outer_folds)
    {
        sum_acc += f.accuracy;
        sum_eer += f.eer;
    }
    double n = static_cast<double>(result.outer_folds.size());
    result.mean_accuracy = sum_acc / n;
    result.mean_eer      = sum_eer / n;

    double var = 0.0;
    for (const auto& f : result.outer_folds)
    {
        double d = f.accuracy - result.mean_accuracy;
        var += d * d;
    }
    result.std_accuracy = std::sqrt(var / n);
}

} // namespace e05
