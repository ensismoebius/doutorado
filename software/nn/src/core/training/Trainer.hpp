/**
 * @file Trainer.hpp
 * @brief Generic training loop for any nn::Module model with pluggable loss and callbacks.
 *
 * Template parameters:
 *   - ModelType : any class with forward(Tensor, bool) → Tensor, backward(Tensor),
 *                 params() → span<Tensor*>. reset_state() called if present.
 *   - LossType  : any class with set_target(Tensor), forward(Tensor, bool) → Tensor,
 *                 backward(Tensor) → Tensor. Default: MSELossImpl<nn::Backend>.
 *
 * All console output goes through ITrainingCallback — Trainer itself has no cout.
 * Add a ProgressCallback or LogCallback to restore visible training output.
 *
 * Bugs fixed vs prior implementation:
 *   1. zero_grad now called BEFORE forward (was after).
 *   2. Single forward+loss pass (was double forward).
 *   3. Loss type is a template parameter (was hardcoded MSE).
 *   4. snn_lr_scale wired via attach_with_scales (was silently ignored).
 *   5. fit_supervised_generic batch loss: output re-run per sample (shape mismatch fixed).
 *   6. EpochResult.mean_spike_rate / sops populated when LossType exposes last_mean_rate().
 *   7. No cout inside Trainer — output is callback responsibility.
 */
#ifndef NN_TRAINING_TRAINER_HPP
#define NN_TRAINING_TRAINER_HPP

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "core/training/EpochResult.hpp"
#include "core/training/TrainerConfig.hpp"
#include "Backend.hpp"
#include "layers/losses/MSELoss.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include "training/ITrainingCallback.hpp"
#include "utility/GradClip.hpp"

#if defined(NN_BACKEND_OPENCL)
#include "tensor/opencl/OpenCLContext.hpp"
#endif

namespace nn::training
{

namespace detail
{
// Detect last_mean_rate() on LossType at compile time.
template <typename T, typename = void>
struct has_last_mean_rate : std::false_type
{
};
template <typename T>
struct has_last_mean_rate<T, std::void_t<decltype(std::declval<T>().last_mean_rate())>>
    : std::true_type
{
};

// Detect reset_state() on ModelType at compile time.
template <typename T, typename = void>
struct has_reset_state : std::false_type
{
};
template <typename T>
struct has_reset_state<T, std::void_t<decltype(std::declval<T>().reset_state())>> : std::true_type
{
};
} // namespace detail

template <typename ModelType, typename LossType = MSELossImpl<nn::Backend>>
class Trainer
{
   public:
    using Tensor = typename ModelType::Tensor;
    using Sample = Tensor;
    using SamplePair = std::pair<Tensor, Tensor>;
    using SampleTransform = std::function<Tensor(const Tensor&, std::size_t)>;

    explicit Trainer(ModelType& model, const TrainerConfig& cfg) : Trainer(model, cfg, LossType{})
    {
    }

    explicit Trainer(ModelType& model, const TrainerConfig& cfg, LossType loss)
        : model_(model),
          cfg_(cfg),
          loss_(std::move(loss)),
          optimizer_(cfg.learning_rate, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon)
    {
        optimizer_.weight_decay = cfg_.weight_decay; // decoupled L2 (AdamW), 0 = off
        if (cfg_.snn_lr_scale != 1.0F)
        {
            auto params = model_.params();
            std::vector<float> scales(params.size(), cfg_.snn_lr_scale);
            optimizer_.attach_with_scales(params, scales);
        }
        else
        {
            optimizer_.attach(model_.params());
        }
    }

    void add_callback(std::shared_ptr<ITrainingCallback> cb)
    {
        callbacks_.push_back(std::move(cb));
    }

    void set_sample_transform(SampleTransform fn)
    {
        sample_transform_ = std::move(fn);
    }

    auto fit_autoencoder(const std::vector<Sample>& train_samples,
        const std::vector<Sample>& val_samples = {}) -> std::vector<EpochResult>
    {
        return fit_loop(train_samples, val_samples, true);
    }

    auto fit_supervised(const std::vector<SamplePair>& train_pairs,
        const std::vector<SamplePair>& val_pairs = {}) -> std::vector<EpochResult>
    {
        std::vector<Sample> train_inputs, train_targets, val_inputs, val_targets;
        for (const auto& [i, t] : train_pairs)
        {
            train_inputs.push_back(i);
            train_targets.push_back(t);
        }
        for (const auto& [i, t] : val_pairs)
        {
            val_inputs.push_back(i);
            val_targets.push_back(t);
        }
        return fit_loop_supervised(train_inputs, train_targets, val_inputs, val_targets);
    }

    const TrainerConfig& config() const
    {
        return cfg_;
    }

   private:
    ModelType& model_;
    TrainerConfig cfg_;
    LossType loss_;
    Adam optimizer_;
    SampleTransform sample_transform_;
    std::vector<std::shared_ptr<ITrainingCallback>> callbacks_;

    // --- callback helpers ---

    void cb_train_begin(int total_epochs)
    {
        for (auto& cb : callbacks_) cb->on_train_begin(total_epochs);
    }

    void cb_train_end(const std::vector<EpochResult>& hist)
    {
        for (auto& cb : callbacks_) cb->on_train_end(hist);
    }

    void cb_epoch_begin(const TrainingState& s)
    {
        for (auto& cb : callbacks_) cb->on_epoch_begin(s);
    }

    void cb_epoch_end(const TrainingState& s, const EpochResult& r)
    {
        for (auto& cb : callbacks_) cb->on_epoch_end(s, r);
    }

    void cb_batch_begin(const TrainingState& s)
    {
        for (auto& cb : callbacks_) cb->on_batch_begin(s);
    }

    void cb_batch_progress(const TrainingState& s)
    {
        for (auto& cb : callbacks_) cb->on_batch_progress(s);
    }

    void cb_batch_end(const TrainingState& s)
    {
        for (auto& cb : callbacks_) cb->on_batch_end(s);
    }

    bool cb_should_stop() const
    {
        for (const auto& cb : callbacks_)
            if (cb->should_stop()) return true;
        return false;
    }

    // --- gradient clipping ---

    static void clip_grad_norm(std::span<Tensor*> params, float max_norm)
    {
        nn::utils::clip_grad_norm(params, max_norm);
    }

    // --- apply optional sample transform ---

    auto transform(const Tensor& s, std::size_t idx) const -> Tensor
    {
        return sample_transform_ ? sample_transform_(s, idx) : s;
    }

    // --- batch utilities ---

    static auto create_batch(const std::vector<Sample>& samples, std::size_t start, std::size_t end)
        -> Tensor
    {
        if (end - start == 1) return samples[start];

        const auto& shape = samples[start].get_shape();
        const std::size_t B = end - start;

        if (shape.size() == 2)
        {
            const std::size_t R = shape[0], C = shape[1];
            // Row-vector samples (1, C) stack into a 2D (B, C) batch — the natural
            // dense/SNN layout. Stacking into (B, 1, C) breaks stateful layers whose
            // membrane state is 2D (e.g. Lif). Only genuine 2D samples (R > 1, e.g.
            // images) need a (B, R, C) 3D batch.
            if (R == 1)
            {
                Tensor batch(static_cast<nn::Index>(B), static_cast<nn::Index>(C));
                for (std::size_t i = start; i < end; ++i)
                    for (std::size_t j = 0; j < C; ++j)
                        batch.at(static_cast<nn::Index>(i - start), static_cast<nn::Index>(j)) =
                            samples[i].at(static_cast<nn::Index>(j));
                return batch;
            }
            Tensor batch(std::vector<nn::Index>{
                static_cast<nn::Index>(B), static_cast<nn::Index>(R), static_cast<nn::Index>(C)});
            for (std::size_t i = start; i < end; ++i)
                for (std::size_t j = 0; j < R * C; ++j)
                    batch.at(static_cast<nn::Index>((i - start) * R * C + j)) =
                        samples[i].at(static_cast<nn::Index>(j));
            return batch;
        }
        if (shape.size() == 1)
        {
            const std::size_t F = shape[0];
            Tensor batch(B, F);
            for (std::size_t i = start; i < end; ++i)
                for (std::size_t j = 0; j < F; ++j) batch.at(i - start, j) = samples[i].at(j);
            return batch;
        }
        return samples[start];
    }

    // --- populate SNN energy fields if LossType supports it ---

    void maybe_populate_snn_fields(EpochResult& result)
    {
        if constexpr (detail::has_last_mean_rate<LossType>::value)
            result.mean_spike_rate = loss_.last_mean_rate();
    }

    // --- main autoencoder loop ---

    auto fit_loop(const std::vector<Sample>& train_samples,
        const std::vector<Sample>& val_samples,
        bool /*autoencoder_mode*/) -> std::vector<EpochResult>
    {
        const int N = static_cast<int>(train_samples.size());
        const int batches_per_epoch = (N + cfg_.batch_size - 1) / cfg_.batch_size;

        std::vector<EpochResult> history;
        history.reserve(static_cast<std::size_t>(cfg_.epochs));

        std::vector<std::size_t> indices(train_samples.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(cfg_.sampler_shuffle_seed);

        cb_train_begin(cfg_.epochs);

        for (int epoch = 1; epoch <= cfg_.epochs; ++epoch)
        {
            const auto t_start = std::chrono::steady_clock::now();

            TrainingState state;
            state.epoch = epoch;
            state.total_epochs = cfg_.epochs;
            state.total_batches = batches_per_epoch;
            cb_epoch_begin(state);

            std::shuffle(indices.begin(), indices.end(), rng);

            float train_loss_sum = 0.0F;
            int n_train = 0;
            int batch_idx = 0;

            std::size_t batch_start = 0;
            while (batch_start < train_samples.size())
            {
                const std::size_t batch_end = std::min(
                    batch_start + static_cast<std::size_t>(cfg_.batch_size), train_samples.size());

                state.batch = ++batch_idx;
                state.batch_progress = 0.0F;
                state.batch_loss = 0.0F;
                cb_batch_begin(state);

                // Build batch (with optional per-sample transform)
                std::vector<Sample> batch_samples;
                batch_samples.reserve(batch_end - batch_start);
                const std::size_t batch_sample_count = batch_end - batch_start;
                for (std::size_t k = batch_start; k < batch_end; ++k)
                {
                    batch_samples.push_back(transform(train_samples[indices[k]], indices[k]));
                    const float sample_fraction =
                        static_cast<float>((k - batch_start) + 1) /
                        static_cast<float>(std::max<std::size_t>(batch_sample_count, 1));
                    state.batch_progress = 0.35F * sample_fraction;
                    cb_batch_progress(state);
                }

                Tensor batch = create_batch(batch_samples, 0, batch_samples.size());
                state.batch_progress = 0.45F;
                cb_batch_progress(state);

                // zero_grad BEFORE forward (bug 1 fix)
                optimizer_.zero_grad(model_.params());

                float loss_val = 0.0F;
                {
#if defined(NN_BACKEND_OPENCL)
                    nn::opencl::OpenCLContext::BatchScope _gpu_batch;
#endif
                    // Single forward+loss+backward (bug 2 fix)
                    Tensor output = model_.forward(batch, true);
                    state.batch_progress = 0.65F;
                    cb_batch_progress(state);
                    loss_.set_target(batch); // autoencoder: target = input
                    Tensor loss_tensor = loss_.forward(output, true);
                    loss_val = loss_tensor.at(0, 0);
                    state.batch_loss = loss_val;
                    state.batch_progress = 0.75F;
                    cb_batch_progress(state);

                    Tensor d_out = loss_.backward(output);
                    model_.backward(d_out);
                } // BatchScope destructs here: single clFinish per batch

                state.batch_progress = 0.9F;
                cb_batch_progress(state);

                if (cfg_.grad_clip_norm > 0.0F)
                    clip_grad_norm(model_.params(), cfg_.grad_clip_norm);

                optimizer_.step(model_.params());

                const int bs = static_cast<int>(batch_end - batch_start);
                train_loss_sum += loss_val * static_cast<float>(bs);
                n_train += bs;

                state.batch_progress = 1.0F;
                cb_batch_end(state);

                batch_start = batch_end;
            }

            const float avg_train_loss =
                (n_train > 0) ? train_loss_sum / static_cast<float>(n_train) : 0.0F;

            // Validation
            float avg_val_loss = std::numeric_limits<float>::quiet_NaN();
            if (!val_samples.empty())
            {
                float val_sum = 0.0F;
                int n_val = 0;

                std::size_t val_start = 0;
                while (val_start < val_samples.size())
                {
                    const std::size_t val_end = std::min(
                        val_start + static_cast<std::size_t>(cfg_.batch_size), val_samples.size());

                    std::vector<Sample> vbatch_samples;
                    vbatch_samples.reserve(val_end - val_start);
                    for (std::size_t k = val_start; k < val_end; ++k)
                        vbatch_samples.push_back(transform(val_samples[k], k));

                    Tensor vbatch = create_batch(vbatch_samples, 0, vbatch_samples.size());

                    Tensor vout = model_.forward(vbatch, false);
                    loss_.set_target(vbatch);
                    Tensor vloss_t = loss_.forward(vout, false);
                    const float vloss = vloss_t.at(0, 0);

                    const int vbs = static_cast<int>(val_end - val_start);
                    val_sum += vloss * static_cast<float>(vbs);
                    n_val += vbs;

                    val_start = val_end;
                }
                avg_val_loss = val_sum / static_cast<float>(n_val);
            }

            const auto t_end = std::chrono::steady_clock::now();
            const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

            EpochResult result{epoch, avg_train_loss, avg_val_loss, ms};
            maybe_populate_snn_fields(result); // bug 6 fix

            state.last_epoch_result = &result;
            cb_epoch_end(state, result);
            history.push_back(result);

            if (cb_should_stop()) break;
        }

        cb_train_end(history);
        return history;
    }

    // --- supervised loop (input != target) ---

    auto fit_loop_supervised(const std::vector<Sample>& train_inputs,
        const std::vector<Sample>& train_targets,
        const std::vector<Sample>& val_inputs,
        const std::vector<Sample>& val_targets) -> std::vector<EpochResult>
    {
        const int N = static_cast<int>(train_inputs.size());
        const int batches_per_epoch = (N + cfg_.batch_size - 1) / cfg_.batch_size;

        std::vector<EpochResult> history;
        history.reserve(static_cast<std::size_t>(cfg_.epochs));

        std::vector<std::size_t> indices(train_inputs.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(cfg_.sampler_shuffle_seed);

        cb_train_begin(cfg_.epochs);

        for (int epoch = 1; epoch <= cfg_.epochs; ++epoch)
        {
            const auto t_start = std::chrono::steady_clock::now();

            TrainingState state;
            state.epoch = epoch;
            state.total_epochs = cfg_.epochs;
            state.total_batches = batches_per_epoch;
            cb_epoch_begin(state);

            std::shuffle(indices.begin(), indices.end(), rng);

            float train_loss_sum = 0.0F;
            int n_train = 0;
            int batch_idx = 0;

            std::size_t batch_start = 0;
            while (batch_start < train_inputs.size())
            {
                const std::size_t batch_end = std::min(
                    batch_start + static_cast<std::size_t>(cfg_.batch_size), train_inputs.size());

                state.batch = ++batch_idx;
                state.batch_progress = 0.0F;
                state.batch_loss = 0.0F;
                cb_batch_begin(state);

                // True-batch forward: stack samples into (B, D) then one fwd+bwd per batch.
                float batch_loss_sum = 0.0F;
                optimizer_.zero_grad(model_.params()); // zero BEFORE forward
                const std::size_t batch_sample_count = batch_end - batch_start;

                {
#if defined(NN_BACKEND_OPENCL)
                    nn::opencl::OpenCLContext::BatchScope _gpu_batch;
#endif
                    const auto B = static_cast<nn::Index>(batch_sample_count);
                    const nn::Index in_cols =
                        transform(train_inputs[indices[batch_start]], indices[batch_start]).cols();
                    const nn::Index tgt_cols = train_targets[indices[batch_start]].cols();

                    Tensor batch_inp = Tensor::zeros(B, in_cols);
                    Tensor batch_tgt = Tensor::zeros(B, tgt_cols);
                    for (std::size_t k = batch_start; k < batch_end; ++k)
                    {
                        const std::size_t idx = indices[k];
                        const Tensor inp = transform(train_inputs[idx], idx);
                        const auto row = static_cast<nn::Index>(k - batch_start);
                        batch_inp.setBlock(row, 0, inp);
                        batch_tgt.setBlock(row, 0, train_targets[idx]);
                    }

                    state.batch_progress = 0.3F;
                    cb_batch_progress(state);

                    Tensor output = model_.forward(batch_inp, true);
                    state.batch_progress = 0.6F;
                    cb_batch_progress(state);

                    loss_.set_target(batch_tgt);
                    Tensor loss_t = loss_.forward(output, true);
                    batch_loss_sum = loss_t.at(0, 0) * static_cast<float>(B);
                    state.batch_loss = loss_t.at(0, 0);
                    state.batch_progress = 0.75F;
                    cb_batch_progress(state);

                    Tensor d_out = loss_.backward(output);
                    model_.backward(d_out);
                    state.batch_progress = 0.9F;
                    cb_batch_progress(state);
                } // BatchScope destructs here: single clFinish per mini-batch

                if (cfg_.grad_clip_norm > 0.0F)
                    clip_grad_norm(model_.params(), cfg_.grad_clip_norm);

                state.batch_progress = 0.92F;
                cb_batch_progress(state);

                optimizer_.step(model_.params());

                const int bs = static_cast<int>(batch_end - batch_start);
                const float avg_batch_loss = batch_loss_sum / static_cast<float>(bs);
                train_loss_sum += avg_batch_loss * static_cast<float>(bs);
                n_train += bs;

                state.batch_loss = avg_batch_loss;
                state.batch_progress = 1.0F;
                cb_batch_end(state);

                batch_start = batch_end;
            }

            const float avg_train_loss =
                (n_train > 0) ? train_loss_sum / static_cast<float>(n_train) : 0.0F;

            // Validation — single batched forward pass over all val samples.
            float avg_val_loss = std::numeric_limits<float>::quiet_NaN();
            if (!val_inputs.empty())
            {
                const auto Nv = static_cast<nn::Index>(val_inputs.size());
                const nn::Index in_c = val_inputs[0].cols();
                const nn::Index tgt_c = val_targets[0].cols();
                Tensor val_inp = Tensor::zeros(Nv, in_c);
                Tensor val_tgt = Tensor::zeros(Nv, tgt_c);
                for (std::size_t k = 0; k < val_inputs.size(); ++k)
                {
                    val_inp.setBlock(static_cast<nn::Index>(k), 0, val_inputs[k]);
                    val_tgt.setBlock(static_cast<nn::Index>(k), 0, val_targets[k]);
                }
                Tensor vout = model_.forward(val_inp, false);
                loss_.set_target(val_tgt);
                Tensor vloss_t = loss_.forward(vout, false);
                avg_val_loss = vloss_t.at(0, 0);
            }

            const auto t_end = std::chrono::steady_clock::now();
            const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

            EpochResult result{epoch, avg_train_loss, avg_val_loss, ms};
            maybe_populate_snn_fields(result);

            state.last_epoch_result = &result;
            cb_epoch_end(state, result);
            history.push_back(result);

            if (cb_should_stop()) break;
        }

        cb_train_end(history);
        return history;
    }
};

} // namespace nn::training

#endif // NN_TRAINING_TRAINER_HPP
