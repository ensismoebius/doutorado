/**
 * @file src/demos/cppdemos/rede_snn.cpp
 * @brief Implementation for Rede snn.
 *

 */

#include <random>
#include <vector>

#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/testing.hpp"

using nn::Index;
using nn::Tensor;
using ModuleEigen = Module<nn::Backend>;
using std::make_shared;
using std::shared_ptr;
using std::vector;

// =============================================================================

// Simple POD to hold model configuration parameters (hardcoded defaults).
struct ModelConfig
{
    // Hardcoded defaults matching common demo usage
    int input_size = 100;
    int output_size = 10;
    int num_residual_blocks = 3;
    int hidden_size = 100;
    float input_scale = 1.0f;
    int num_steps = 100;
    float time_step = 0.001f;
    float resistance = 5.0f;
    float capacitance = 1.0f;
    float voltage_threshold = 0.01f;
    float learning_rate = 0.001f;
    int epochs = 4000;
    int batch_size = 32;
};

// =============================================================================
// --- Helper to create layers succinctly ---
// =============================================================================

/** Create a Linear layer */
auto lin(int in, int out) -> std::shared_ptr<ModuleEigen>
{
    return make_shared<Linear>(in, out);
};

/** Create a LeakyBPTT layer */
auto leaky(const ModelConfig& cfg, bool readout = false) -> std::shared_ptr<ModuleEigen>
{
    // `LeakyBPTT` expects its input as a single matrix with shape (T*B, F)
    // (time-major flatten). This demo flattens the per-step tensors that way.
    //
    // In spiking mode it outputs spikes; in readout_mode it outputs the membrane value
    // (useful for continuous reconstruction at the final layer).
    return make_shared<LeakyBPTT>( //
        cfg.num_steps,
        cfg.time_step,
        cfg.resistance,
        cfg.capacitance,
        cfg.voltage_threshold,
        true, // reset_zero
        0.0f, // reset_pot
        readout,
        make_shared<ExponentialSurrogate>(1.0f));
};

// =============================================================================
// Residual SNN Block: Linear -> Leaky -> Linear -> Leaky + skip connection
// =============================================================================
struct ResidualSNNBlock : public ModuleEigen
{
   public:
    Sequential model;

    explicit ResidualSNNBlock(const ModelConfig& cfg)
        : model(Sequential({
              lin(cfg.hidden_size, cfg.hidden_size),
              leaky(cfg),
              lin(cfg.hidden_size, cfg.hidden_size),
              leaky(cfg),
          }))
    {
    }

    auto forward(const Tensor& x, bool requires_grad = true) -> Tensor override
    {
        auto z = model.forward(x, requires_grad);
        return z + x; // residual connection
    }

    auto backward(const Tensor& grad_out) -> Tensor override
    {
        auto grad = model.backward(grad_out);
        return grad + grad_out; // residual connection
    }

    auto params() -> std::span<Tensor*> override
    {
        return model.params();
    }

    void reset_state() override
    {
        model.reset_state();
    }
};

struct SnnModel : public ModuleEigen
{
   public:
    Sequential model;

    explicit SnnModel(const ModelConfig& cfg)
        : model(Sequential({
              lin(cfg.input_size, cfg.hidden_size),
              leaky(cfg),
              make_shared<ResidualSNNBlock>(cfg),
              make_shared<ResidualSNNBlock>(cfg),
              make_shared<ResidualSNNBlock>(cfg),
              lin(cfg.hidden_size, cfg.output_size),
              leaky(cfg, true),
          }))
    {
        // Initialize RNG
        std::mt19937 rng(42);

        // // Calculate tau from beta
        // const float tau = 1.0F / std::max(1e-6F, -std::log(beta_));
        // lif_in->resistance.at(0, 0) = tau;
        // lif_out->resistance.at(0, 0) = tau;

        // // Initialize layers
        // fc_in->weight = Tensor::rand((Index) num_ocultos, (Index) num_entradas_, rng);
        // fc_in->bias = Tensor::zeros((Index) num_ocultos, 1);

        // res_blocks.reserve(num_blocos_residuais);
        // for (int i = 0; i < num_blocos_residuais; ++i)
        // {
        //     res_blocks.emplace_back(make_shared<ResidualSNNBlock>(num_ocultos, tau, rng));
        // }

        // fc_out->weight = Tensor::rand((Index) num_saidas, (Index) num_ocultos, rng);
        // fc_out->bias = Tensor::zeros((Index) num_saidas, 1);
    }

    auto initialize_state(Index /*batch_size*/) -> void
    {
        // Initialize all Linear layers inside the Sequential model using
        // kaimingSNNInitializer. We walk the top-level layers and also
        // inspect nested Sequentials / Residual blocks to find `Linear`s.
        for (auto& layer : model.layers)
        {
            // If top-level is a Linear
            if (auto linptr = std::dynamic_pointer_cast<Linear>(layer))
            {
                kaimingSNNInitializer(linptr, nn::testing::kSeed);
                continue;
            }

            // If layer is a Sequential (or Residual block exposing inner Sequential)
            if (auto seq = std::dynamic_pointer_cast<Sequential>(layer))
            {
                for (auto& sub : seq->layers)
                {
                    if (auto sublin = std::dynamic_pointer_cast<Linear>(sub))
                    {
                        kaimingSNNInitializer(sublin, nn::testing::kSeed);
                    }
                }
            }
            else
            {
                // Some custom blocks (e.g., ResidualSNNBlock) contain a `model` member
                // which is a Sequential. Try to dynamic_cast and inspect.
                // Use RTTI-safe approach by attempting to cast to ResidualSNNBlock.
                if (auto rb = std::dynamic_pointer_cast<ResidualSNNBlock>(layer))
                {
                    for (auto& sub : rb->model.layers)
                    {
                        if (auto sublin = std::dynamic_pointer_cast<Linear>(sub))
                        {
                            kaimingSNNInitializer(sublin, nn::testing::kSeed);
                        }
                    }
                }
            }
        }
    }

    // Module interface: delegate to internal Sequential
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        return model.forward(input, requires_grad);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        return model.backward(grad_output);
    }

    void reset_state() override
    {
        // Call reset_state on all submodules if implemented
        for (auto& layer : model.layers)
        {
            layer->reset_state();
        }
    }

    auto params() -> std::span<Tensor*> override
    {
        return model.params();
    }
};