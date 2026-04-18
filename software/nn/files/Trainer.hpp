#pragma once

/**
 * @file Trainer.hpp
 * @brief Training loop for the LSTM Autoencoder in Experiment04.
 *
 * Provides an epoch-level training + validation runner that is structurally
 * consistent with the Experiment03 training loop so results are directly
 * comparable:
 *   - Same loss function: MSE
 *   - Same optimizer: Adam
 *   - Same per-epoch scalar logging: reconstruction loss
 *   - Same gradient-clipping option
 *
 * Each "sample" is a 2-D Tensor [seq_len × input_size].  The Trainer owns no
 * dataset abstraction — callers supply raw nn::Tensor samples which simplifies
 * integration with TensorDataset or any other source.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "Experiment04Config.hpp"
#include "LSTMAutoencoder.hpp"

#include "nn/layers/MSELoss.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"

namespace experiment04
{

// ---------------------------------------------------------------------------
// Per-epoch result record
// ---------------------------------------------------------------------------
struct EpochResult
{
    int    epoch;
    float  train_loss;
    float  val_loss;        ///< NaN when no validation set is given
    double epoch_ms;        ///< wall-clock time in milliseconds
};

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------
class Trainer
{
public:
    using Sample = nn::Tensor;  // [seq_len × input_size]

    explicit Trainer(LSTMAutoencoder& model, const Experiment04Config& cfg)
        : model_(model), cfg_(cfg),
          optimizer_(cfg.learning_rate, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon)
    {
        optimizer_.attach(model_.params());
    }

    /**
     * @brief Run the full training loop.
     *
     * @param train_samples  Vector of [seq_len × input_size] tensors.
     * @param val_samples    Optional validation set (may be empty).
     * @return Per-epoch metrics.
     */
    auto fit(const std::vector<Sample>& train_samples,
             const std::vector<Sample>& val_samples = {}) -> std::vector<EpochResult>
    {
        std::vector<EpochResult> history;
        history.reserve(static_cast<size_t>(cfg_.epochs));

        // Shuffle index for training
        std::vector<size_t> indices(train_samples.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(cfg_.sampler_shuffle_seed);

        MSELossImpl<nn::EigenTensorBackend> loss_fn;

        for (int epoch = 1; epoch <= cfg_.epochs; ++epoch)
        {
            auto t_start = std::chrono::steady_clock::now();

            // ---- Shuffle ----
            std::shuffle(indices.begin(), indices.end(), rng);

            // ---- Training pass ----
            float train_loss_accum = 0.0f;
            int   n_train          = 0;

            for (size_t idx : indices)
            {
                const Sample& sample = train_samples[idx];

                // Reset LSTM state between independent sequences
                model_.reset_state();

                // Zero gradients
                optimizer_.zero_grad(model_.params());

                // Forward
                nn::Tensor recon = model_.forward(sample, /*requires_grad=*/true);

                // Loss
                loss_fn.set_target(sample);
                nn::Tensor loss_t = loss_fn.forward(recon, /*requires_grad=*/true);
                float loss_val = loss_t.at(0, 0);
                train_loss_accum += loss_val;
                ++n_train;

                // Backward
                nn::Tensor d_recon = loss_fn.backward(recon);
                model_.backward(d_recon);

                // Optional gradient clipping
                if (cfg_.grad_clip_norm > 0.0f)
                {
                    clip_grad_norm(model_.params(), cfg_.grad_clip_norm);
                }

                // Optimizer step
                optimizer_.step(model_.params());

                // Honour max_batches_per_epoch limit
                if (cfg_.max_batches_per_epoch > 0 &&
                    n_train >= cfg_.max_batches_per_epoch)
                {
                    break;
                }
            }

            float avg_train_loss = (n_train > 0)
                                       ? train_loss_accum / static_cast<float>(n_train)
                                       : 0.0f;

            // ---- Validation pass ----
            float avg_val_loss = std::numeric_limits<float>::quiet_NaN();
            if (!val_samples.empty())
            {
                float val_accum = 0.0f;
                int   n_val     = 0;
                for (const Sample& vs : val_samples)
                {
                    model_.reset_state();
                    nn::Tensor vrecon = model_.forward(vs, /*requires_grad=*/false);
                    loss_fn.set_target(vs);
                    nn::Tensor vl = loss_fn.forward(vrecon, /*requires_grad=*/false);
                    val_accum += vl.at(0, 0);
                    ++n_val;
                }
                avg_val_loss = val_accum / static_cast<float>(n_val);
            }

            auto t_end   = std::chrono::steady_clock::now();
            double ms    = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            EpochResult res{epoch, avg_train_loss, avg_val_loss, ms};
            history.push_back(res);

            log_epoch(res, val_samples.empty());
        }

        return history;
    }

private:
    LSTMAutoencoder&  model_;
    Experiment04Config cfg_;
    Adam              optimizer_;

    // ---- Gradient clipping ----
    static void clip_grad_norm(std::span<nn::Tensor*> params, float max_norm)
    {
        // Compute global gradient L2 norm across all parameters
        float total_sq = 0.0f;
        for (nn::Tensor* p : params)
        {
            nn::Tensor g = p->grad();
            total_sq += g.norm() * g.norm();
        }
        float global_norm = std::sqrt(total_sq);
        if (global_norm > max_norm && global_norm > 0.0f)
        {
            float scale = max_norm / global_norm;
            for (nn::Tensor* p : params)
            {
                nn::Tensor g = p->grad();
                g.multiply_scalar_inplace(scale);
                p->set_grad(g);
            }
        }
    }

    // ---- Logging (mirrors experiment03 format) ----
    static void log_epoch(const EpochResult& r, bool no_val)
    {
        std::cout << "[experiment04] epoch=" << r.epoch
                  << "  train_loss=" << r.train_loss;
        if (!no_val && !std::isnan(r.val_loss))
        {
            std::cout << "  val_loss=" << r.val_loss;
        }
        std::cout << "  time=" << static_cast<int>(r.epoch_ms) << "ms\n";
    }
};

} // namespace experiment04
