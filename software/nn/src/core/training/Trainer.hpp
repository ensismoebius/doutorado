/**
 * @file Trainer.hpp
 * @brief Generic training loop for any nn::Module model.
 *
 * This class provides a PyTorch Lightning-style training loop:
 *   - Works with any model: autoencoders, classifiers, LSTMs, etc.
 *   - Configurable loss function (MSE, CrossEntropy, MAE)
 *   - Supports both autoencoder mode (input=target) and supervised (input,target) pairs
 *   - Batch processing
 *
 * PyTorch vs This Code:
 *   PyTorch:                  This Code:
 *   model.train()             Trainer handles automatically
 *   optimizer.zero_grad()     trainer.fit()
 *   loss = criterion(out, target)
 *   loss.backward()
 *   optimizer.step()
 *
 * Example Usage (Autoencoder):
 *   @code
 *   Autoencoder model(cfg);
 *   Trainer<decltype(model), nn::layers::MSELoss> trainer(model, config);
 *   auto history = trainer.fit_autoencoder(train_data, val_data);
 *   @endcode
 *
 * Example Usage (Supervised):
 *   @code
 *   Classifier model(cfg);
 *   Trainer<decltype(model), nn::layers::CrossEntropyLoss> trainer(model, config);
 *   std::vector<std::pair<nn::Tensor, nn::Tensor>> train_pairs = ...;
 *   auto history = trainer.fit_supervised(train_pairs, val_pairs);
 *   @endcode
 */
#ifndef NN_TRAINING_TRAINER_HPP
#define NN_TRAINING_TRAINER_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "nn/layers/eigen/Layers.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::training
{

struct EpochResult
{
    int epoch;
    float train_loss;
    float val_loss;
    float epoch_ms;
};

struct TrainerConfig
{
    int epochs = 10;
    float learning_rate = 0.001F;
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1e-8F;
    float grad_clip_norm = 0.0F;
    int batch_size = 1;
    unsigned int sampler_shuffle_seed = 42;
};

template <typename ModelType>
class Trainer
{
public:
    using Sample = nn::Tensor;
    using SamplePair = std::pair<nn::Tensor, nn::Tensor>;

    explicit Trainer(ModelType& model, const TrainerConfig& cfg)
        : model_(model),
          cfg_(cfg),
          optimizer_(cfg.learning_rate, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon)
    {
        optimizer_.attach(model_.params());
    }

    auto fit_autoencoder(
        const std::vector<Sample>& train_samples,
        const std::vector<Sample>& val_samples = {}
    ) -> std::vector<EpochResult>
    {
        return fit_generic(
            train_samples,
            val_samples,
            TrainingMode::Autoencoder
        );
    }

    auto fit_supervised(
        const std::vector<SamplePair>& train_pairs,
        const std::vector<SamplePair>& val_pairs = {}
    ) -> std::vector<EpochResult>
    {
        return fit_supervised_generic(
            train_pairs,
            val_pairs
        );
    }

    auto config() const -> const TrainerConfig&
    {
        return cfg_;
    }

private:
    enum class TrainingMode
    {
        Autoencoder,
        Supervised
    };

    ModelType& model_;
    TrainerConfig cfg_;
    nn::optimizers::Adam optimizer_;

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

    auto compute_loss(const nn::Tensor& output, const nn::Tensor& target, bool requires_grad) -> float
    {
        nn::layers::MSELossImpl<nn::EigenTensorBackend> loss_fn;
        loss_fn.set_target(target);
        nn::Tensor loss_tensor = loss_fn.forward(output, requires_grad);
        return loss_tensor.at(0, 0);
    }

    nn::Tensor create_batch(const std::vector<Sample>& samples, size_t start, size_t end)
    {
        if (end - start == 1)
        {
            return samples[start];
        }

        const auto& first_shape = samples[start].get_shape();
        size_t batch_size = end - start;

        if (first_shape.size() == 2)
        {
            size_t rows = first_shape[0];
            size_t cols = first_shape[1];
            nn::Tensor batch(batch_size, rows, cols);
            for (size_t i = start; i < end; ++i)
            {
                for (size_t j = 0; j < rows * cols; ++j)
                {
                    batch.at(i - start, j) = samples[i].at(j);
                }
            }
            return batch;
        }
        else if (first_shape.size() == 1)
        {
            size_t features = first_shape[0];
            nn::Tensor batch(batch_size, features);
            for (size_t i = start; i < end; ++i)
            {
                for (size_t j = 0; j < features; ++j)
                {
                    batch.at(i - start, j) = samples[i].at(j);
                }
            }
            return batch;
        }

        return samples[start];
    }

    auto fit_generic(
        const std::vector<Sample>& train_samples,
        const std::vector<Sample>& val_samples,
        TrainingMode mode
    ) -> std::vector<EpochResult>
    {
        std::vector<EpochResult> history;
        history.reserve(static_cast<size_t>(cfg_.epochs));

        std::vector<size_t> indices(train_samples.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(cfg_.sampler_shuffle_seed);

        for (int epoch = 1; epoch <= cfg_.epochs; ++epoch)
        {
            const auto t_start = std::chrono::steady_clock::now();

            std::shuffle(indices.begin(), indices.end(), rng);

            float train_loss_accum = 0.0F;
            int n_train = 0;

            size_t batch_start = 0;
            while (batch_start < train_samples.size())
            {
                size_t batch_end = std::min(
                    batch_start + static_cast<size_t>(cfg_.batch_size),
                    train_samples.size()
                );

                nn::Tensor batch = create_batch(train_samples, batch_start, batch_end);
                nn::Tensor target = batch;

                nn::Tensor output = model_.forward(batch, true);
                float loss_val = compute_loss(output, target, true);

                train_loss_accum += loss_val;
                n_train += static_cast<int>(batch_end - batch_start);

                optimizer_.zero_grad(model_.params());

                nn::layers::MSELossImpl<nn::EigenTensorBackend> loss_fn;
                loss_fn.set_target(target);
                nn::Tensor loss_tensor = loss_fn.forward(output, true);
                nn::Tensor d_out = loss_fn.backward(output);
                model_.backward(d_out);

                if (cfg_.grad_clip_norm > 0.0F)
                {
                    clip_grad_norm(model_.params(), cfg_.grad_clip_norm);
                }

                optimizer_.step(model_.params());

                batch_start = batch_end;
            }

            const float avg_train_loss = (n_train > 0)
                ? train_loss_accum / static_cast<float>(n_train)
                : 0.0F;

            float avg_val_loss = std::numeric_limits<float>::quiet_NaN();

            if (!val_samples.empty())
            {
                float val_accum = 0.0F;
                int n_val = 0;

                size_t val_batch_start = 0;
                while (val_batch_start < val_samples.size())
                {
                    size_t val_batch_end = std::min(
                        val_batch_start + static_cast<size_t>(cfg_.batch_size),
                        val_samples.size()
                    );

                    nn::Tensor vbatch = create_batch(val_samples, val_batch_start, val_batch_end);
                    nn::Tensor vtarget = vbatch;

                    nn::Tensor vrecon = model_.forward(vbatch, false);
                    float vloss = compute_loss(vrecon, vtarget, false);

                    val_accum += vloss;
                    n_val += static_cast<int>(val_batch_end - val_batch_start);

                    val_batch_start = val_batch_end;
                }

                avg_val_loss = val_accum / static_cast<float>(n_val);
            }

            const auto t_end = std::chrono::steady_clock::now();
            const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

            EpochResult result{epoch, avg_train_loss, avg_val_loss, ms};
            history.push_back(result);

            log_epoch(result, val_samples.empty());
        }

        return history;
    }

    auto fit_supervised_generic(
        const std::vector<SamplePair>& train_pairs,
        const std::vector<SamplePair>& val_pairs
    ) -> std::vector<EpochResult>
    {
        std::vector<EpochResult> history;
        history.reserve(static_cast<size_t>(cfg_.epochs));

        std::vector<size_t> indices(train_pairs.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(cfg_.sampler_shuffle_seed);

        for (int epoch = 1; epoch <= cfg_.epochs; ++epoch)
        {
            const auto t_start = std::chrono::steady_clock::now();

            std::shuffle(indices.begin(), indices.end(), rng);

            float train_loss_accum = 0.0F;
            int n_train = 0;

            std::vector<nn::Tensor> batch_inputs;
            std::vector<nn::Tensor> batch_targets;

            for (size_t idx : indices)
            {
                const auto& [input, target] = train_pairs[idx];

                batch_inputs.push_back(input);
                batch_targets.push_back(target);

                if (static_cast<int>(batch_inputs.size()) >= cfg_.batch_size ||
                    n_train + batch_inputs.size() == train_pairs.size())
                {
                    nn::Tensor output = model_.forward(merge_batch(batch_inputs), true);

                    float loss_val = 0.0F;
                    for (size_t i = 0; i < batch_inputs.size(); ++i)
                    {
                        loss_val += compute_loss(
                            output,
                            batch_targets[i],
                            true
                        );
                    }
                    loss_val /= static_cast<float>(batch_inputs.size());

                    train_loss_accum += loss_val * static_cast<float>(batch_inputs.size());
                    n_train += static_cast<int>(batch_inputs.size());

                    optimizer_.zero_grad(model_.params());

                    nn::layers::MSELossImpl<nn::EigenTensorBackend> loss_fn;
                    nn::Tensor merged_target = merge_batch(batch_targets);
                    loss_fn.set_target(merged_target);
                    nn::Tensor loss_tensor = loss_fn.forward(output, true);
                    nn::Tensor d_out = loss_fn.backward(output);
                    model_.backward(d_out);

                    if (cfg_.grad_clip_norm > 0.0F)
                    {
                        clip_grad_norm(model_.params(), cfg_.grad_clip_norm);
                    }

                    optimizer_.step(model_.params());

                    batch_inputs.clear();
                    batch_targets.clear();
                }
            }

            const float avg_train_loss = (n_train > 0)
                ? train_loss_accum / static_cast<float>(n_train)
                : 0.0F;

            float avg_val_loss = std::numeric_limits<float>::quiet_NaN();

            if (!val_pairs.empty())
            {
                float val_accum = 0.0F;
                int n_val = 0;

                std::vector<nn::Tensor> vbatch_inputs;
                std::vector<nn::Tensor> vbatch_targets;

                for (const auto& [vinput, vtarget] : val_pairs)
                {
                    vbatch_inputs.push_back(vinput);
                    vbatch_targets.push_back(vtarget);

                    if (static_cast<int>(vbatch_inputs.size()) >= cfg_.batch_size ||
                        n_val + vbatch_inputs.size() == val_pairs.size())
                    {
                        nn::Tensor voutput = model_.forward(merge_batch(vbatch_inputs), false);

                        float vloss = 0.0F;
                        for (size_t i = 0; i < vbatch_inputs.size(); ++i)
                        {
                            vloss += compute_loss(voutput, vbatch_targets[i], false);
                        }
                        vloss /= static_cast<float>(vbatch_inputs.size());

                        val_accum += vloss * static_cast<float>(vbatch_inputs.size());
                        n_val += static_cast<int>(vbatch_inputs.size());

                        vbatch_inputs.clear();
                        vbatch_targets.clear();
                    }
                }

                avg_val_loss = val_accum / static_cast<float>(n_val);
            }

            const auto t_end = std::chrono::steady_clock::now();
            const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

            EpochResult result{epoch, avg_train_loss, avg_val_loss, ms};
            history.push_back(result);

            log_epoch(result, val_pairs.empty());
        }

        return history;
    }

    nn::Tensor merge_batch(const std::vector<nn::Tensor>& samples)
    {
        if (samples.empty())
        {
            return nn::Tensor(1, 1);
        }

        if (samples.size() == 1)
        {
            return samples[0];
        }

        const auto& first_shape = samples[0].get_shape();
        size_t batch_size = samples.size();

        if (first_shape.size() == 2)
        {
            size_t rows = first_shape[0];
            size_t cols = first_shape[1];
            nn::Tensor batch(batch_size, rows, cols);
            for (size_t i = 0; i < batch_size; ++i)
            {
                for (size_t j = 0; j < rows * cols; ++j)
                {
                    batch.at(i, j) = samples[i].at(j);
                }
            }
            return batch;
        }
        else if (first_shape.size() == 1)
        {
            size_t features = first_shape[0];
            nn::Tensor batch(batch_size, features);
            for (size_t i = 0; i < batch_size; ++i)
            {
                for (size_t j = 0; j < features; ++j)
                {
                    batch.at(i, j) = samples[i].at(j);
                }
            }
            return batch;
        }

        return samples[0];
    }

    static void log_epoch(const EpochResult& r, bool no_val)
    {
        std::cout << "[Trainer] epoch=" << r.epoch
                  << "  train_loss=" << r.train_loss;

        if (!no_val && !std::isnan(r.val_loss))
        {
            std::cout << "  val_loss=" << r.val_loss;
        }

        std::cout << "  time=" << static_cast<int>(r.epoch_ms) << "ms\n";
    }
};

} // namespace nn::training

#endif // NN_TRAINING_TRAINER_HPP
