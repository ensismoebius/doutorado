#include <cnpy.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/LeakyBPTT.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/MSELoss.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/utility/EigenParallel.hpp"
#include "nn/utility/batching.hpp"
#include "nn/utility/reset.hpp"
#include "nn/utility/synthetic_spike_data.hpp"
#include "nn/utility/vectorizationCheck.hpp"

using namespace std;

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

// =============================================================================
// SNN Encoder-Decoder Model (PyTorch/snntorch style)
// =============================================================================
class SpikeAutoEncoder : public Module
{
   public:
    Sequential encoder;
    Sequential decoder;

    SpikeAutoEncoder(const ModelConfig& cfg)
    {
        // --- Helper to create layers succinctly ---
        auto lin = [](int in, int out) { return make_shared<Linear>(in, out); };
        auto leaky = [&](bool readout = false)
        {
            return make_shared<LeakyBPTT>(cfg.steps,
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
        std::vector<std::shared_ptr<Module>> enc_layers;
        enc_layers.push_back(lin(cfg.input_dim, cfg.hidden_dims[0]));
        enc_layers.push_back(leaky());
        enc_layers.push_back(lin(cfg.hidden_dims[0], cfg.hidden_dims[1]));
        enc_layers.push_back(leaky());
        enc_layers.push_back(lin(cfg.hidden_dims[1], cfg.hidden_dims[2]));
        enc_layers.push_back(leaky());
        enc_layers.push_back(lin(cfg.hidden_dims[2], cfg.hidden_dims[3]));
        enc_layers.push_back(leaky());
        enc_layers.push_back(lin(cfg.hidden_dims[3], cfg.hidden_dims[4]));
        enc_layers.push_back(leaky());
        enc_layers.push_back(lin(cfg.hidden_dims[4], cfg.bottleneck_dim));
        enc_layers.push_back(leaky());
        encoder = Sequential(enc_layers);

        // --- Build Decoder ---
        // Layers: Bottleneck -> H5 -> H4 -> H3 -> H2 -> H1 -> Output
        std::vector<std::shared_ptr<Module>> dec_layers;
        dec_layers.push_back(lin(cfg.bottleneck_dim, cfg.hidden_dims[4]));
        dec_layers.push_back(leaky());
        dec_layers.push_back(lin(cfg.hidden_dims[4], cfg.hidden_dims[3]));
        dec_layers.push_back(leaky());
        dec_layers.push_back(lin(cfg.hidden_dims[3], cfg.hidden_dims[2]));
        dec_layers.push_back(leaky());
        dec_layers.push_back(lin(cfg.hidden_dims[2], cfg.hidden_dims[1]));
        dec_layers.push_back(leaky());
        dec_layers.push_back(lin(cfg.hidden_dims[1], cfg.hidden_dims[0]));
        dec_layers.push_back(leaky());
        dec_layers.push_back(lin(cfg.hidden_dims[0], cfg.input_dim));
        dec_layers.push_back(leaky(true)); // Readout layer (leaky integrator)
        decoder = Sequential(dec_layers);
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

    // Collect parameters
    vector<nn::Tensor*> params() override
    {
        auto p = encoder.params();
        auto p_dec = decoder.params();
        p.insert(p.end(), p_dec.begin(), p_dec.end());
        return p;
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
        bool loaded =
            load_weights_from_file(encoder, enc_path) && load_weights_from_file(decoder, dec_path);

        if (!loaded)
        {
            cout << "Weights not found. Using Kaiming initialization.\n";
            // Initialize Linear layers manually
            // Iterate and apply kaimingSNNInitializer
            auto init_seq = [](Sequential& seq)
            {
                for (auto& layer : seq.layers)
                {
                    if (auto l = dynamic_pointer_cast<Linear>(layer))
                    {
                        kaimingSNNInitializer(l);
                        // Scale weights slightly to prevent explosion in deep SNNs
                        for (int i = 0; i < l->weight.rows(); ++i)
                            for (int j = 0; j < l->weight.cols(); ++j) l->weight.at(i, j) *= 0.01f;
                    }
                }
            };
            init_seq(encoder);
            init_seq(decoder);
        }
        else
        {
            cout << "Weights loaded successfully.\n";
        }
    }

   private:
    bool load_weights_from_file(Sequential& seq, const string& file)
    {
        if (!std::filesystem::exists(file)) return false;
        try
        {
            cnpy::npz_t data = cnpy::npz_load(file);
            // Matches index to layer in sequential
            // Seq.layers contains [Linear, Leaky, Linear, Leaky...]
            for (size_t i = 0; i < seq.layers.size(); ++i)
            {
                if (auto lin = dynamic_pointer_cast<Linear>(seq.layers[i]))
                {
                    string key_prefix = std::to_string(i);
                    string w_key = key_prefix + ".weight";
                    string b_key = key_prefix + ".bias";

                    if (data.count(w_key))
                    {
                        auto& arr = data[w_key];
                        float* w_ptr = arr.data<float>();
                        for (int r = 0; r < lin->weight.rows(); ++r)
                            for (int c = 0; c < lin->weight.cols(); ++c)
                                lin->weight.at(r, c) = w_ptr[r * lin->weight.cols() + c];
                    }
                    if (data.count(b_key))
                    {
                        auto& arr = data[b_key];
                        float* b_ptr = arr.data<float>();
                        for (int r = 0; r < lin->bias.rows(); ++r)
                            for (int c = 0; c < lin->bias.cols(); ++c)
                                lin->bias.at(r, c) = b_ptr[r * lin->bias.cols() + c];
                    }
                }
            }
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};

// =============================================================================
// Utility: Gradient Clipping
// =============================================================================
void clip_gradients(vector<nn::Tensor*>& params, float max_norm)
{
    float total_norm_sq = 0.0f;
    for (auto* p : params)
    {
        nn::Tensor g = p->grad();
        if (g.size() > 0)
        {
            // Simple check: iterate to find norm
            for (int i = 0; i < g.rows(); ++i)
                for (int j = 0; j < g.cols(); ++j) total_norm_sq += g.at(i, j) * g.at(i, j);
        }
    }
    float total_norm = std::sqrt(total_norm_sq);

    if (total_norm > max_norm)
    {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto* p : params)
        {
            nn::Tensor g = p->grad();
            for (int i = 0; i < g.rows(); ++i)
                for (int j = 0; j < g.cols(); ++j) g.at(i, j) *= scale;
            p->set_grad(g);
        }
    }
}

// =============================================================================
// Main
// =============================================================================
auto main(int, char*[]) -> int
{
    util::initializeEigenParallel();
    printVectorizationSupport();
    cout << fixed << scientific << setprecision(4);

    // 1. Setup
    ModelConfig config;
    SpikeAutoEncoder model(config);
    model.initialize_weights("weights/encoder_spike_model_weights.npz",
                             "weights/decoder_spike_model_weights.npz");

    auto params = model.params();
    Adam optimizer(config.lr);
    optimizer.attach(params);
    auto criterion = make_shared<MSELoss>();

    // 2. Data Generation
    // Flatten logic: (Samples * Steps, Dim)
    int n_samples = 10;
    auto [input_seq, _] =
        generate_autoencoder_spike_data_of_ones(n_samples, config.input_dim, config.steps);

    nn::Tensor inputs(n_samples * config.steps, config.input_dim);
    for (int t = 0; t < config.steps; ++t)
    {
        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < config.input_dim; ++j)
            {
                inputs.at(t * n_samples + i, j) = input_seq[t].at(i, j);
            }
        }
    }

    // 3. Training Loop
    ofstream log_file("cpp_loss_log.txt");
    cout << "Starting training...\n";

    for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        // Zero Grad
        optimizer.zero_grad(params);
        model.reset_states();

        // Forward
        criterion->set_target(inputs); // Autoencoder target = input
        nn::Tensor recon = model.forward(inputs);
        nn::Tensor loss_val = criterion->forward(recon);

        // Backward
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
