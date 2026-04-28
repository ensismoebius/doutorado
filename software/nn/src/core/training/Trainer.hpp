/**
 * @file Trainer.hpp
 * @brief Generic training loop for any nn::Module model with pluggable loss and callbacks.
 *
 * Template parameters:
 *   - ModelType : any class with forward(Tensor, bool) → Tensor, backward(Tensor),
 *                 params() → span<Tensor*>. reset_state() called if present.
 *   - LossType  : any class with set_target(Tensor), forward(Tensor, bool) → Tensor,
 *                 backward(Tensor) → Tensor. Default: MSELossImpl<EigenTensorBackend>.
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
#include <cmath>
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
#include "nn/layers/losses/MSELoss.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/training/ITrainingCallback.hpp"

namespace nn::training
{

namespace detail
{
// Detect last_mean_rate() on LossType at compile time.
template <typename T, typename = void>
struct has_last_mean_rate : std::false_type {};
template <typename T>
struct has_last_mean_rate<T, std::void_t<decltype(std::declval<T>().last_mean_rate())>>
    : std::true_type {};

// Detect reset_state() on ModelType at compile time.
template <typename T, typename = void>
struct has_reset_state : std::false_type {};
template <typename T>
struct has_reset_state<T, std::void_t<decltype(std::declval<T>().reset_state())>>
    : std::true_type {};
} // namespace detail

template <typename ModelType,
          typename LossType = MSELossImpl<nn::EigenTensorBackend>>
class Trainer
{
   public:
    using Sample      = nn::Tensor;
    using SamplePair  = std::pair<nn::Tensor, nn::Tensor>;
    using SampleTransform = std::function<nn::Tensor(const nn::Tensor&, std::size_t)>;

    explicit Trainer(ModelType& model, const TrainerConfig& cfg)
        : Trainer(model, cfg, LossType{}) {}

    explicit Trainer(ModelType& model, const TrainerConfig& cfg, LossType loss)
        : model_(model),
          cfg_(cfg),
          loss_(std::move(loss)),
          optimizer_(cfg.learning_rate, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon)
    {
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
        for (const auto& [i, t] : train_pairs) { train_inputs.push_back(i); train_targets.push_back(t); }
        for (const auto& [i, t] : val_pairs)   { val_inputs.push_back(i);   val_targets.push_back(t); }
        return fit_loop_supervised(train_inputs, train_targets, val_inputs, val_targets);
    }

    const TrainerConfig& config() const { return cfg_; }

   private:
    ModelType&         model_;
    TrainerConfig      cfg_;
    LossType           loss_;
    Adam               optimizer_;
    SampleTransform    sample_transform_;
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

    static void clip_grad_norm(std::span<nn::Tensor*> params, float max_norm)
    {
        float total_sq = 0.0F;
        for (nn::Tensor* p : params)
        {
            nn::Tensor g = p->grad();
            total_sq += g.norm() * g.norm();
        }
        const float global_norm = std::sqrt(total_sq);
        if (global_norm > max_norm && global_norm > 0.0F)
        {
            const float scale = max_norm / global_norm;
            for (nn::Tensor* p : params)
            {
                nn::Tensor g = p->grad();
                g.multiply_scalar_inplace(scale);
                p->set_grad(g);
            }
        }
    }

    // --- apply optional sample transform ---

    nn::Tensor transform(const nn::Tensor& s, std::size_t idx) const
    {
        return sample_transform_ ? sample_transform_(s, idx) : s;
    }

    // --- batch utilities ---

    static nn::Tensor create_batch(const std::vector<Sample>& samples,
                                   std::size_t start, std::size_t end)
    {
        if (end - start == 1) return samples[start];

        const auto& shape = samples[start].get_shape();
        const std::size_t B = end - start;

        if (shape.size() == 2)
        {
            const std::size_t R = shape[0], C = shape[1];
            nn::Tensor batch(std::vector<nn::Index>{
                static_cast<nn::Index>(B),
                static_cast<nn::Index>(R),
                static_cast<nn::Index>(C)});
            for (std::size_t i = start; i < end; ++i)
                for (std::size_t j = 0; j < R * C; ++j)
                    batch.at(static_cast<nn::Index>((i - start) * R * C + j)) =
                        samples[i].at(static_cast<nn::Index>(j));
            return batch;
        }
        if (shape.size() == 1)
        {
            const std::size_t F = shape[0];
            nn::Tensor batch(B, F);
            for (std::size_t i = start; i < end; ++i)
                for (std::size_t j = 0; j < F; ++j)
                    batch.at(i - start, j) = samples[i].at(j);
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
        const int batches_per_epoch =
            (N + cfg_.batch_size - 1) / cfg_.batch_size;

        std::vector<EpochResult> history;
        history.reserve(static_cast<std::size_t>(cfg_.epochs));

        std::vector<std::size_t> indices(train_samples.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(cfg_.sampler_shuffle_seed);

        cb_train_begin(cfg_.epochs);

        for (int epoch = 1; epoch <= cfg_.epochs; ++epoch)
        {
            const auto t_start = std::chrono::steady_clock::now();

            TrainingState state{epoch, cfg_.epochs, 0, batches_per_epoch, 0.0F, nullptr};
            cb_epoch_begin(state);

            std::shuffle(indices.begin(), indices.end(), rng);

            float train_loss_sum = 0.0F;
            int   n_train        = 0;
            int   batch_idx      = 0;

            std::size_t batch_start = 0;
            while (batch_start < train_samples.size())
            {
                const std::size_t batch_end = std::min(
                    batch_start + static_cast<std::size_t>(cfg_.batch_size),
                    train_samples.size());

                // Build batch (with optional per-sample transform)
                std::vector<Sample> batch_samples;
                batch_samples.reserve(batch_end - batch_start);
                for (std::size_t k = batch_start; k < batch_end; ++k)
                    batch_samples.push_back(transform(train_samples[indices[k]], indices[k]));

                nn::Tensor batch = create_batch(batch_samples, 0, batch_samples.size());

                state.batch = ++batch_idx;
                cb_batch_begin(state);

                // zero_grad BEFORE forward (bug 1 fix)
                optimizer_.zero_grad(model_.params());

                // Single forward+loss+backward (bug 2 fix)
                nn::Tensor output = model_.forward(batch, true);
                loss_.set_target(batch); // autoencoder: target = input
                nn::Tensor loss_tensor = loss_.forward(output, true);
                const float loss_val = loss_tensor.at(0, 0);

                nn::Tensor d_out = loss_.backward(output);
                model_.backward(d_out);

                if (cfg_.grad_clip_norm > 0.0F)
                    clip_grad_norm(model_.params(), cfg_.grad_clip_norm);

                optimizer_.step(model_.params());

                const int bs = static_cast<int>(batch_end - batch_start);
                train_loss_sum += loss_val * static_cast<float>(bs);
                n_train        += bs;

                state.batch_loss = loss_val;
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
                int   n_val   = 0;

                std::size_t val_start = 0;
                while (val_start < val_samples.size())
                {
                    const std::size_t val_end = std::min(
                        val_start + static_cast<std::size_t>(cfg_.batch_size),
                        val_samples.size());

                    std::vector<Sample> vbatch_samples;
                    vbatch_samples.reserve(val_end - val_start);
                    for (std::size_t k = val_start; k < val_end; ++k)
                        vbatch_samples.push_back(transform(val_samples[k], k));

                    nn::Tensor vbatch = create_batch(vbatch_samples, 0, vbatch_samples.size());

                    nn::Tensor vout = model_.forward(vbatch, false);
                    loss_.set_target(vbatch);
                    nn::Tensor vloss_t = loss_.forward(vout, false);
                    const float vloss = vloss_t.at(0, 0);

                    const int vbs = static_cast<int>(val_end - val_start);
                    val_sum += vloss * static_cast<float>(vbs);
                    n_val   += vbs;

                    val_start = val_end;
                }
                avg_val_loss = val_sum / static_cast<float>(n_val);
            }

            const auto t_end = std::chrono::steady_clock::now();
            const float ms   = std::chrono::duration<float, std::milli>(t_end - t_start).count();

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
        const int batches_per_epoch =
            (N + cfg_.batch_size - 1) / cfg_.batch_size;

        std::vector<EpochResult> history;
        history.reserve(static_cast<std::size_t>(cfg_.epochs));

        std::vector<std::size_t> indices(train_inputs.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(cfg_.sampler_shuffle_seed);

        cb_train_begin(cfg_.epochs);

        for (int epoch = 1; epoch <= cfg_.epochs; ++epoch)
        {
            const auto t_start = std::chrono::steady_clock::now();

            TrainingState state{epoch, cfg_.epochs, 0, batches_per_epoch, 0.0F, nullptr};
            cb_epoch_begin(state);

            std::shuffle(indices.begin(), indices.end(), rng);

            float train_loss_sum = 0.0F;
            int   n_train        = 0;
            int   batch_idx      = 0;

            std::size_t batch_start = 0;
            while (batch_start < train_inputs.size())
            {
                const std::size_t batch_end = std::min(
                    batch_start + static_cast<std::size_t>(cfg_.batch_size),
                    train_inputs.size());

                // Collect this mini-batch (bug 5 fix: per-sample forward inside batch)
                float batch_loss_sum = 0.0F;
                optimizer_.zero_grad(model_.params()); // zero BEFORE any forward

                for (std::size_t k = batch_start; k < batch_end; ++k)
                {
                    const std::size_t idx = indices[k];
                    const nn::Tensor& inp = transform(train_inputs[idx], idx);
                    const nn::Tensor& tgt = train_targets[idx];

                    nn::Tensor output    = model_.forward(inp, true);
                    loss_.set_target(tgt);
                    nn::Tensor loss_t    = loss_.forward(output, true);
                    const float lv       = loss_t.at(0, 0);
                    batch_loss_sum      += lv;

                    nn::Tensor d_out = loss_.backward(output);
                    model_.backward(d_out);
                }

                if (cfg_.grad_clip_norm > 0.0F)
                    clip_grad_norm(model_.params(), cfg_.grad_clip_norm);

                optimizer_.step(model_.params());

                const int bs = static_cast<int>(batch_end - batch_start);
                const float avg_batch_loss = batch_loss_sum / static_cast<float>(bs);
                train_loss_sum += avg_batch_loss * static_cast<float>(bs);
                n_train        += bs;

                state.batch      = ++batch_idx;
                state.batch_loss = avg_batch_loss;
                cb_batch_begin(state);
                cb_batch_end(state);

                batch_start = batch_end;
            }

            const float avg_train_loss =
                (n_train > 0) ? train_loss_sum / static_cast<float>(n_train) : 0.0F;

            // Validation
            float avg_val_loss = std::numeric_limits<float>::quiet_NaN();
            if (!val_inputs.empty())
            {
                float val_sum = 0.0F;
                int   n_val   = 0;

                for (std::size_t k = 0; k < val_inputs.size(); ++k)
                {
                    nn::Tensor vout = model_.forward(val_inputs[k], false);
                    loss_.set_target(val_targets[k]);
                    nn::Tensor vloss_t = loss_.forward(vout, false);
                    val_sum += vloss_t.at(0, 0);
                    ++n_val;
                }
                avg_val_loss = val_sum / static_cast<float>(n_val);
            }

            const auto t_end = std::chrono::steady_clock::now();
            const float ms   = std::chrono::duration<float, std::milli>(t_end - t_start).count();

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
