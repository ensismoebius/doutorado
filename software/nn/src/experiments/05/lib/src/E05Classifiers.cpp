#include "E05Classifiers.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "Backend.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "layers/Layers.hpp"
#include "layers/losses/CrossEntropyLoss.hpp"
#include "layers/residual/SimpleResNet.hpp"
#include "progress/ProgressManager.hpp"
#include "statistics/confusion_matrix.hpp"
#include "statistics/kfold.hpp"
#include "tensor/Tensor.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ProgressCallback.hpp"

namespace e05
{

namespace
{
// Convert a flat feature vector to a (1 x D) nn::Tensor.
nn::Tensor vec_to_row(const std::vector<double>& v)
{
    nn::Tensor t = nn::Tensor::zeros(1, static_cast<long>(v.size()));
    for (long i = 0; i < static_cast<long>(v.size()); ++i)
        t.at(0, i) = static_cast<float>(v[static_cast<size_t>(i)]);
    return t;
}

// One-hot encode label as (1 x n_classes) tensor.
nn::Tensor one_hot(int label, int n_classes)
{
    nn::Tensor t = nn::Tensor::zeros(1, n_classes);
    t.at(0, label) = 1.0f;
    return t;
}

// Return argmax of a (1 x N) tensor.
int argmax(const nn::Tensor& t)
{
    int best = 0;
    float best_val = t.at(0, 0);
    for (long j = 1; j < t.cols(); ++j)
    {
        float v = t.at(0, j);
        if (v > best_val)
        {
            best_val = v;
            best = static_cast<int>(j);
        }
    }
    return best;
}

// Evaluate classifier accuracy and EER on a set of (input, label) pairs.
std::pair<double, double> evaluate(SimpleResNetImpl<nn::Backend>& model,
    const std::vector<std::vector<double>>& inputs,
    const std::vector<int>& labels,
    int n_classes)
{
    int correct = 0;
    std::vector<statistics::ConfusionMatrix> cms;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        nn::Tensor x = vec_to_row(inputs[i]);
        nn::Tensor logits = model.forward(x, false);
        int pred = argmax(logits);
        if (pred == labels[i]) ++correct;

        // One-vs-rest confusion matrix per class.
        for (int c = 0; c < n_classes; ++c)
        {
            bool is_c  = (labels[i] == c);
            bool pred_c = (pred == c);
            statistics::ConfusionMatrix cm{};
            if (is_c  && pred_c)  ++cm.truePositive;
            if (!is_c && pred_c)  ++cm.falsePositive;
            if (is_c  && !pred_c) ++cm.falseNegative;
            if (!is_c && !pred_c) ++cm.trueNegative;
            if (c < static_cast<int>(cms.size()))
            {
                cms[static_cast<size_t>(c)].truePositive  += cm.truePositive;
                cms[static_cast<size_t>(c)].falsePositive += cm.falsePositive;
                cms[static_cast<size_t>(c)].falseNegative += cm.falseNegative;
                cms[static_cast<size_t>(c)].trueNegative  += cm.trueNegative;
            }
            else
            {
                cms.push_back(cm);
            }
        }
    }

    double accuracy = static_cast<double>(correct) / static_cast<double>(inputs.size());
    double eer = 0.0;
    {
        std::vector<double> fprs, fnrs;
        statistics::calculateEER(cms, eer, fprs, fnrs);
    }
    return {accuracy, eer};
}
} // namespace

auto run_classifier(const E05DatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const E05Config& cfg,
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

    // Nested KFold.
    statistics::NestedKFold nkf(
        static_cast<size_t>(cfg.training.k_folds),
        static_cast<size_t>(cfg.training.k_folds),
        true,
        cfg.experiment.seed);

    auto nested_splits = nkf.split(view.samples.size());

    ClassificationResult result;
    result.feature_set_label = feature_label;
    result.classifier_type   = cfg.classifier.type;
    result.text_mode         = cfg.classifier.text_mode;

    const int hidden_dim = 128;
    const int depth      = 2; // residual blocks

    const int total_outer = static_cast<int>(nested_splits.size());

    for (size_t outer_idx = 0; outer_idx < nested_splits.size(); ++outer_idx)
    {
        const auto& outer = nested_splits[outer_idx];

        // Collect test data.
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

        // Convert train data to SamplePair list.
        using SamplePair = std::pair<nn::Tensor, nn::Tensor>;
        std::vector<SamplePair> train_pairs;
        for (size_t idx : outer.inner_splits[0].train_indices)
        {
            train_pairs.emplace_back(vec_to_row(feature_vectors[idx]),
                one_hot(labels[idx], n_speakers));
        }

        std::vector<SamplePair> val_pairs;
        for (size_t idx : outer.inner_splits[0].test_indices)
        {
            val_pairs.emplace_back(vec_to_row(feature_vectors[idx]),
                one_hot(labels[idx], n_speakers));
        }

        trainer.fit_supervised(train_pairs, val_pairs);

        auto [acc, eer] = evaluate(model, test_feats, test_labels, n_speakers);

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
