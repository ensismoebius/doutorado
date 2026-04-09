/**
 * @file autoEncoderLeakyReLUAndSpikeTest.cpp
 * @brief End-to-end spiking autoencoder demo (snnTorch-like structure).
 *
 * This executable is written to be read like a small PyTorch/snnTorch script:
 * - a model class (encoder/decoder built from `Sequential` blocks)
 * - a clean training loop (zero_grad → reset_state → forward → loss → backward → step)
 * - explicit notes about the project-specific shape conventions for time-flattened SNN input.
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/LeakyBPTT.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/MSELoss.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/testing.hpp"
#include "nn/utility/EigenParallel.hpp"
#include "nn/utility/reset.hpp"
#include "nn/utility/synthetic_spike_data.hpp"
#include "nn/utility/vectorizationCheck.hpp"

using namespace std;
using ModuleEigen = Module<nn::EigenTensorBackend>;

namespace
{
// Configuration holding all hyperparameters
struct ModelConfig
{
    int input_dim = 100;
    int hidden_dims[5] = {50, 40, 30, 20, 10}; // 5 hidden layers
    int bottleneck_dim = 10;

    int steps = 100;
    float dt = 0.001f;
    float R = 5.0f;
    float C = 1.0f;
    float thr = 0.01f;

    float lr = 0.001f;
    int epochs = 4000;
    int batch_size = 32;
};
} // namespace

// =============================================================================
// SNN Encoder-Decoder Model (PyTorch/snntorch style)
// =============================================================================
class SpikeAutoEncoder : public ModuleEigen
{
   public:
    Sequential encoder;
    Sequential decoder;

    // Storage for non-owning return of `params()`
    std::vector<nn::Tensor*> param_ptrs_;

    explicit SpikeAutoEncoder(const ModelConfig& cfg)
    {
        // --- Helper to create layers succinctly ---
        auto lin = [](int in, int out) -> std::shared_ptr<ModuleEigen>
        { return make_shared<Linear>(in, out); };

        auto leaky = [&](bool readout = false) -> std::shared_ptr<ModuleEigen>
        {
            // `LeakyBPTT` expects its input as a single matrix with shape (T*B, F)
            // (time-major flatten). This demo flattens the per-step tensors that way.
            //
            // In spiking mode it outputs spikes; in readout_mode it outputs the membrane value
            // (useful for continuous reconstruction at the final layer).
            return make_shared<LeakyBPTT>( //
                cfg.steps,
                cfg.dt,
                cfg.R,
                cfg.C,
                cfg.thr,
                true, // reset_zero
                0.0f, // reset_pot
                readout,
                make_shared<ExponentialSurrogate>(1.0f));
        };

        // --- Build Encoder ---
        // Layers: Input -> H1 -> H2 -> H3 -> H4 -> H5 -> Bottleneck
        encoder = Sequential({
            lin(cfg.input_dim, cfg.hidden_dims[0]),
            leaky(),
            lin(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            leaky(),
            lin(cfg.hidden_dims[1], cfg.hidden_dims[2]),
            leaky(),
            lin(cfg.hidden_dims[2], cfg.hidden_dims[3]),
            leaky(),
            lin(cfg.hidden_dims[3], cfg.hidden_dims[4]),
            leaky(),
            lin(cfg.hidden_dims[4], cfg.bottleneck_dim),
            leaky(),
        });

        // --- Build Decoder ---
        // Layers: Bottleneck -> H5 -> H4 -> H3 -> H2 -> H1 -> Output
        decoder = Sequential({
            lin(cfg.bottleneck_dim, cfg.hidden_dims[4]),
            leaky(),
            lin(cfg.hidden_dims[4], cfg.hidden_dims[3]),
            leaky(),
            lin(cfg.hidden_dims[3], cfg.hidden_dims[2]),
            leaky(),
            lin(cfg.hidden_dims[2], cfg.hidden_dims[1]),
            leaky(),
            lin(cfg.hidden_dims[1], cfg.hidden_dims[0]),
            leaky(),
            lin(cfg.hidden_dims[0], cfg.input_dim),
            // Readout layer: do not spike; return membrane potential as a continuous output.
            leaky(true),
        });
    }

    // Forward pass: x -> Encoder -> z -> Decoder -> x_recon
    nn::Tensor forward(const nn::Tensor& x, bool requires_grad = true) override
    {
        auto z = encoder.forward(x, requires_grad);
        return decoder.forward(z, requires_grad);
    }

    // Backward pass
    nn::Tensor backward(const nn::Tensor& grad_output) override
    {
        auto grad_decoder = decoder.backward(grad_output);
        return encoder.backward(grad_decoder);
    }

    // Collect parameters (non-owning view). The returned span references storage
    // owned by this module in `param_ptrs_` so callers must not outlive the
    // module instance.
    auto params() -> std::span<nn::Tensor*> override
    {
        param_ptrs_.clear();
        auto p = encoder.params();
        param_ptrs_.insert(param_ptrs_.end(), p.begin(), p.end());
        auto p_dec = decoder.params();
        param_ptrs_.insert(param_ptrs_.end(), p_dec.begin(), p_dec.end());
        return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    // Reset internal states (membrane potentials)
    void reset_states()
    {
        nn::utility::reset(encoder);
        nn::utility::reset(decoder);
    }

    // Initialize or Load Weights
    void initialize_weights(const string& enc_path, const string& dec_path)
    {
        (void) load_weights_from_file(encoder, enc_path);
        (void) load_weights_from_file(decoder, dec_path);

        cout << "Weights not found. Using Kaiming initialization.\n";
        // Initialize Linear layers manually
        // Iterate and apply kaimingSNNInitializer
        auto init_seq = [](Sequential& seq)
        {
            for (auto& layer : seq.layers)
            {
                if (auto l = dynamic_pointer_cast<Linear>(layer))
                {
                    kaimingSNNInitializer(l, nn::testing::kSeed);
                    // Scale weights slightly to prevent explosion in deep SNNs
                    for (int i = 0; i < l->weight.rows(); ++i)
                        for (int j = 0; j < l->weight.cols(); ++j) l->weight.at(i, j) *= 0.01f;
                }
            }
        };
        init_seq(encoder);
        init_seq(decoder);
    }

   private:
    bool load_weights_from_file(Sequential& seq, const string& file)
    {
        (void) seq;
        (void) file;
        std::cerr << "NPZ weight loading disabled in this build; using Kaiming init instead."
                  << std::endl;
        return false;
    }
};

// =============================================================================
// Utility: Gradient Clipping
// =============================================================================
void clip_gradients(std::span<nn::Tensor*> params, float max_norm)
{
    // Global-norm gradient clipping (PyTorch-style):
    // - Compute ||g|| over all parameters.
    // - If it exceeds max_norm, scale every gradient tensor by max_norm/||g||.
    //
    // Why it matters here:
    // - Deep SNN stacks + surrogate gradients can produce unstable/large gradients.
    // - Clipping helps keep training numerically stable without changing the forward dynamics.

    // C++20 Ranges: Compute total norm squared
    auto param_norms = params | std::views::transform(
                                    [](auto* p)
                                    {
                                        float n = p->grad().norm();
                                        return n * n;
                                    });

    const float total_norm_sq = std::accumulate(param_norms.begin(), param_norms.end(), 0.0f);

    float total_norm = std::sqrt(total_norm_sq);

    if (total_norm > max_norm)
    {
        float scale = max_norm / (total_norm + 1e-6f);

        // Important Tensor API note:
        // - `p->grad()` returns a *copy* in this codebase, so we must `set_grad()` after editing.
        // - We scale elementwise via a std::span over the contiguous buffer.

        // C++20 Ranges: Scale gradients
        std::ranges::for_each(params,
            [scale](auto* p)
            {
                nn::Tensor g = p->grad();
                if (g.size() > 0)
                {
                    std::span<float> data(g.mutable_data(), g.size());
                    std::ranges::for_each(data, [scale](float& val) { val *= scale; });
                    p->set_grad(g);
                }
            });
    }
}

// =============================================================================
// Main
// =============================================================================
auto main(int, char*[]) -> int
{
    try
    {
        util::initializeEigenParallel();
        printVectorizationSupport();
        cout << fixed << scientific << setprecision(4);

        // 1. Setup
        ModelConfig config{.input_dim = 100,
            .hidden_dims = {50, 40, 30, 20, 10},
            .bottleneck_dim = 10,
            .steps = 100,
            .dt = 0.001f,
            .R = 5.0f,
            .C = 1.0f,
            .thr = 0.01f,
            .lr = 0.001f,
            .epochs = 4000,
            .batch_size = 32};

        SpikeAutoEncoder model(config);
        // Runtime NPZ weight loading is disabled in this build. Pass empty paths
        // to avoid referring to .npz at runtime; the model will use Kaiming init.
        model.initialize_weights("", "");

        auto params = model.params();
        Adam optimizer(config.lr);
        optimizer.attach(params);
        auto criterion = make_shared<MSELoss>();

        // 2. Data Generation
        // Data layout note:
        // - `generate_autoencoder_spike_data_*` returns a vector of length T.
        // - Each entry is a tensor of shape (B, F).
        // - `LeakyBPTT` expects a single flattened tensor of shape (T*B, F), time-major.
        //   So we stack time on the row axis: row = t*B + b.
        //
        // Demo note:
        // - `config.batch_size` is not used in this particular demo: we use `n_samples` as B.
        //   (Kept in config because the same model structure can be used with a DataLoader.)
        int n_samples = 10;
        auto [input_seq, _] =
            generate_autoencoder_spike_data_of_ones(n_samples, config.input_dim, config.steps);

        nn::Tensor inputs(n_samples * config.steps, config.input_dim);

        // C++20 Ranges: Nested loops for data flattening
        for (int t : std::views::iota(0, config.steps))
        {
            for (int i : std::views::iota(0, n_samples))
            {
                for (int j : std::views::iota(0, config.input_dim))
                {
                    inputs.at(t * n_samples + i, j) = input_seq[t].at(i, j);
                }
            }
        }

        // 3. Training Loop
        ofstream log_file("cpp_loss_log.txt");
        cout << "Starting training...\n";

        for (int epoch : std::views::iota(0, config.epochs))
        {
            // Typical training step structure:
            // 1) zero gradients
            // 2) reset spiking state (membrane potentials) at sequence boundaries
            // 3) forward + loss
            // 4) backward
            // 5) optional gradient clipping
            // 6) optimizer step

            // Zero Grad
            optimizer.zero_grad(params);
            model.reset_states();

            // Forward
            // Autoencoder target = input: learn to reconstruct the spike train.
            criterion->set_target(inputs);
            nn::Tensor recon = model.forward(inputs);
            nn::Tensor loss_val = criterion->forward(recon);

            // Backward
            // `criterion->backward(recon)` returns dL/d(recon).
            nn::Tensor grad_loss = criterion->backward(recon);
            model.backward(grad_loss);

            // Clip & Step
            clip_gradients(params, 1.0f);
            optimizer.step(params);

            // Log
            float current_loss = loss_val.at(0, 0);
            log_file << epoch << "," << current_loss << "\n";
            if (epoch % 10 == 0 || epoch == config.epochs - 1)
            {
                cout << "Epoch " << epoch << " | Loss: " << current_loss << "\r" << flush;
            }
        }
        cout << "\nTraining finished.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred." << std::endl;
        return 1;
    }
}
