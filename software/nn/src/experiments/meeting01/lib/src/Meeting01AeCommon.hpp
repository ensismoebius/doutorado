#pragma once

/**
 * @file Meeting01AeCommon.hpp
 * @brief Model-generic train / evaluate helpers for the frame-consuming autoencoders
 *        (LSTM-AE, GRU-AE, bottlenecked Transformer-AE).
 *
 * All three consume a window reshaped by to_lstm_frames() into (T, frame_size) and
 * reconstruct it in the same layout, expose the Module contract
 * (forward/backward/params/reset_state), and are driven one sequence at a time
 * (Trainer batch_size = 1). Only the concrete model type and its analytic MAC
 * estimate differ, so the loop below is written once and instantiated per model.
 *
 * Private to Meeting01Training.cpp — not part of the public lib interface.
 */

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "../include/Meeting01BatchLossCollector.hpp"
#include "../include/Meeting01DatasetSplit.hpp"
#include "../include/Meeting01Encoding.hpp"
#include "../include/Meeting01EpochHistory.hpp"
#include "../include/Meeting01EpochLogger.hpp"
#include "../include/Meeting01EventCallback.hpp"
#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01PerWindow.hpp"
#include "../include/Meeting01RunMetrics.hpp"
#include "../include/Meeting01Training.hpp"
#include "core/training/Trainer.hpp"
#include "core/training/TrainerConfig.hpp"
#include "tensor/Tensor.hpp"
#include "training/EarlyStoppingCallback.hpp"
#include "training/ProgressCallback.hpp"

namespace meeting01
{

// Generic reconstruction evaluation in framed space. `Model::forward` consumes and
// emits (T, frame_size); MSE/MAE/R2 are elementwise so framing both sides is a no-op
// on the metrics. Mirrors the former evaluate_lstm() body verbatim.
template <typename Model>
auto evaluate_ae(Model& model,
    const std::vector<Tensor>& val_samples,
    const std::vector<int>& val_labels,
    float max_reconstruct_mean_deviation,
    std::size_t macs,
    std::size_t param_count,
    const std::string& encoding,
    std::uint32_t seed,
    float infer_ms,
    int frame_size) -> RunMetrics
{
    using ModelTensor = typename Model::Tensor;

    RunMetrics m;
    m.macs = macs;
    m.parameter_count = param_count;
    m.infer_ms = infer_ms;

    std::vector<int> pred_labels;
    pred_labels.reserve(val_samples.size());

    float mse_acc = 0.0f;
    float mae_acc = 0.0f;
    float y_mean_acc = 0.0f;
    std::size_t n_values = 0;

    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded = to_lstm_frames(
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i)),
            frame_size);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(ModelTensor(encoded), false));

        mse_acc += mse_between(encoded, recon);
        mae_acc += mae_between(encoded, recon);

        float sample_residual_mean = 0.0f;
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            sample_residual_mean += std::fabs(encoded.at(k) - recon.at(k));
            y_mean_acc += encoded.at(k);
            ++n_values;
        }
        sample_residual_mean /= static_cast<float>(std::max<nn::Index>(1, encoded.size()));
        pred_labels.push_back(sample_residual_mean > max_reconstruct_mean_deviation ? 1 : 0);
    }

    m.mse = val_samples.empty() ? 0.0f : mse_acc / static_cast<float>(val_samples.size());
    m.mae = val_samples.empty() ? 0.0f : mae_acc / static_cast<float>(val_samples.size());

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    const float y_mean = (n_values > 0) ? y_mean_acc / static_cast<float>(n_values) : 0.0f;
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded = to_lstm_frames(
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i)),
            frame_size);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(ModelTensor(encoded), false));
        for (nn::Index k = 0; k < encoded.size(); ++k)
        {
            const float y = encoded.at(k);
            const float yh = recon.at(k);
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - y_mean) * (y - y_mean);
        }
    }
    m.r2 = (ss_tot > 1e-8f) ? (1.0f - (ss_res / ss_tot)) : 0.0f;

    compute_precision_recall_f1(val_labels, pred_labels, m.precision, m.recall, m.f1);

    m.spike_rate = 0.0f;
    m.energy = 10.0f * static_cast<float>(m.macs);

    return m;
}

// Per-window reconstruction error for a frame-consuming AE. `proto` carries the shared
// identity fields (model, architecture, v_th, alpha, run_id, seed, cv_fold, split); this
// fills the per-window identity (from `meta`, parallel to `samples`) and mse/mae. A
// forward pass per window — cheap next to training, and keeps the eval path simple.
template <typename Model>
auto per_window_errors_ae(Model& model,
    const std::vector<Tensor>& samples,
    const std::vector<WindowMetadata>& meta,
    const std::string& encoding,
    std::uint32_t seed,
    int frame_size,
    PerWindowError proto) -> std::vector<PerWindowError>
{
    using ModelTensor = typename Model::Tensor;
    std::vector<PerWindowError> out;
    out.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const Tensor encoded = to_lstm_frames(
            encode_sample(samples[i], encoding, seed + static_cast<std::uint32_t>(i)), frame_size);
        model.reset_state();
        const Tensor recon = Tensor(model.forward(ModelTensor(encoded), false));

        PerWindowError r = proto;
        if (i < meta.size())
        {
            r.speaker_id = meta[i].speaker_id;
            r.recording_id = meta[i].recording_id;
            r.window_id = meta[i].window_id;
            r.source_window_index = meta[i].source_window_index;
        }
        r.mse = mse_between(encoded, recon);
        r.mae = mae_between(encoded, recon);
        out.push_back(r);
    }
    return out;
}

// Generic early-stopping training loop for a frame-consuming AE. `macs` is the
// caller-supplied analytic estimate for this concrete model. Mirrors the former
// train_with_early_stopping_lstm() body.
template <typename Model>
auto train_ae(Model& model,
    const Meeting01Config& cfg,
    const std::vector<Tensor>& train_samples,
    const std::vector<Tensor>& val_samples,
    const std::string& encoding,
    std::uint32_t seed,
    std::size_t run_id,
    std::size_t total_runs,
    const std::string& progress_label,
    const std::string& progress_title,
    std::size_t macs,
    float& train_ms,
    float& infer_ms) -> TrainResult
{
    using ModelTensor = typename Model::Tensor;

    nn::training::TrainerConfig tcfg;
    tcfg.epochs = cfg.training.epochs;
    tcfg.batch_size = 1; // one sequence at a time — no 3-D batching for recurrent / attention AEs
    tcfg.learning_rate = cfg.training.learning_rate;
    tcfg.adam_beta1 = cfg.training.beta1;
    tcfg.adam_beta2 = cfg.training.beta2;
    tcfg.adam_epsilon = cfg.training.epsilon;

    nn::training::Trainer<Model> trainer(model, tcfg);

    auto cb = std::make_shared<nn::training::ProgressCallback>(progress_label);
    cb->set_metadata(
        progress_title, static_cast<int>(run_id + 1), static_cast<int>(total_runs), "MSE");
    trainer.add_callback(cb);

    // Plain per-epoch log lines (survive nohup / pipes, where the live bars collapse).
    trainer.add_callback(std::make_shared<Meeting01EpochLogger>(
        progress_context(cfg, run_id, total_runs, seed), progress_label));

    // Structured JSONL events for the live monitor (identity from the driver's
    // pending context). No-op when the events sink was never opened.
    trainer.add_callback(std::make_shared<Meeting01EventCallback>());

    trainer.add_callback(
        std::make_shared<nn::training::EarlyStoppingCallback>(cfg.training.early_stop_patience));

    auto batch_collector = std::make_shared<BatchLossCollector>();
    trainer.add_callback(batch_collector);

    std::vector<ModelTensor> train_backend_samples;
    train_backend_samples.reserve(train_samples.size());
    for (const auto& sample : train_samples) train_backend_samples.emplace_back(sample);

    std::vector<ModelTensor> val_backend_samples;
    val_backend_samples.reserve(val_samples.size());
    for (const auto& sample : val_samples) val_backend_samples.emplace_back(sample);

    const int frame = cfg.model.lstm_frame_size;
    trainer.set_sample_transform(
        [&model, &encoding, seed, frame](const ModelTensor& s, std::size_t idx) -> ModelTensor
        {
            model.reset_state();
            return ModelTensor(to_lstm_frames(
                encode_sample(Tensor(s), encoding, seed + static_cast<std::uint32_t>(idx)), frame));
        });

    const auto t0 = std::chrono::steady_clock::now();
    const auto epoch_results = trainer.fit_autoencoder(train_backend_samples, val_backend_samples);
    const auto t1 = std::chrono::steady_clock::now();
    train_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const auto infer_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < val_samples.size(); ++i)
    {
        const Tensor encoded = to_lstm_frames(
            encode_sample(val_samples[i], encoding, seed + static_cast<std::uint32_t>(i)), frame);
        model.reset_state();
        (void) model.forward(ModelTensor(encoded), false);
    }
    const auto infer_end = std::chrono::steady_clock::now();
    infer_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();

    RunMetrics metrics = evaluate_ae(model,
        val_samples,
        std::vector<int>(val_samples.size(), 0),
        cfg.training.max_reconstruct_mean_deviation,
        macs,
        parameter_count(model.params()),
        encoding,
        seed,
        infer_ms,
        frame);

    EpochHistory history;
    for (const auto& er : epoch_results)
    {
        history.epoch_nums.push_back(static_cast<float>(er.epoch));
        history.train_losses.push_back(er.train_loss);
        history.val_losses.push_back(er.val_loss);
    }

    const int batches_per_epoch =
        static_cast<int>(train_samples.size()) / std::max(1, cfg.training.samples_per_batch);
    int batch_idx = 0;
    for (const auto& batch_loss : batch_collector->batch_losses)
    {
        const int current_epoch = (batch_idx / std::max(1, batches_per_epoch)) + 1;
        history.batch_losses.push_back(batch_loss);
        history.batch_epochs.push_back(static_cast<float>(current_epoch));
        ++batch_idx;
    }

    return TrainResult{metrics, history};
}

} // namespace meeting01
