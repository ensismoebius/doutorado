#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "nn/layers/Leaky.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Module.hpp"
#include "nn/tensor/Tensor.hpp"

using nn::Index;
using nn::Tensor;

struct ModeloSNN_Cpp : public Module
{
    int num_inputs;
    int num_outputs;
    int num_ocultos;          // hidden size (matches Python name)
    int num_blocos_residuais; // depth (matches Python name)
    float beta;

    // Match Python attribute names
    float escala_entrada = 1.0F;
    Linear fc_in; // input linear
    Leaky lif_in; // input LIF

    struct ResidualSNNBlock_Cpp
    {
        Linear fc1;
        Leaky lif1;
        Linear fc2;
        Leaky lif2;

        ResidualSNNBlock_Cpp(int dim, float tau, std::mt19937& rng)
            : fc1(dim, dim),
              lif1(1.0F, tau, 1.0F, 1.0F, true, 0.0F),
              fc2(dim, dim),
              lif2(1.0F, tau, 1.0F, 1.0F, true, 0.0F)
        {
            fc1.weight = Tensor::rand((Index) dim, (Index) dim, rng);
            fc1.bias = Tensor::zeros((Index) dim, 1);
            fc2.weight = Tensor::rand((Index) dim, (Index) dim, rng);
            fc2.bias = Tensor::zeros((Index) dim, 1);
        }
    };

    std::vector<ResidualSNNBlock_Cpp> res_blocks;

    Linear fc_out;
    Leaky lif_out;

    ModeloSNN_Cpp(int num_inputs_, int num_outputs_, int profundidade_, int tamanho_oculto_ = 100,
                  float beta_ = 0.9f)
        : num_inputs(num_inputs_),
          num_outputs(num_outputs_),
          num_ocultos(tamanho_oculto_),
          num_blocos_residuais(profundidade_),
          beta(beta_),
          fc_in(num_inputs_, tamanho_oculto_),
          lif_in(1.0F, /*R=*/1.0F, /*C=*/1.0F, /*V_thresh=*/1.0F, true, 0.0F),
          fc_out(tamanho_oculto_, num_outputs_),
          lif_out(1.0F, /*R=*/1.0F, /*C=*/1.0F, /*V_thresh=*/1.0F, true, 0.0F)
    {
        std::mt19937 rng(42);
        const float tau = 1.0F / std::max(1e-6F, -std::log(beta_));
        lif_in.resistance.at(0, 0) = tau;
        lif_out.resistance.at(0, 0) = tau;

        // Initialize weights deterministic
        fc_in.weight = Tensor::rand((Index) num_ocultos, (Index) num_inputs_, rng);
        fc_in.bias = Tensor::zeros((Index) num_ocultos, 1);

        res_blocks.reserve(num_blocos_residuais);
        for (int i = 0; i < num_blocos_residuais; ++i)
        {
            res_blocks.emplace_back(num_ocultos, tau, rng);
        }

        fc_out.weight = Tensor::rand((Index) num_outputs, (Index) num_ocultos, rng);
        fc_out.bias = Tensor::zeros((Index) num_outputs, 1);
    }

    std::string description() const
    {
        std::ostringstream ss;
        ss << "ModeloSNN(num_inputs=" << num_inputs << ", num_outputs=" << num_outputs
           << ", hidden_size=" << num_ocultos << ", depth=" << num_blocos_residuais
           << ", beta=" << beta << ")";
        return ss.str();
    }

    // Initialize state tensors for a given batch size. Order:
    // [mem_in, mem_out, mem_res0_1, mem_res0_2, mem_res1_1, mem_res1_2, ...]
    auto inicializar_estado(Index batch_size) -> std::vector<Tensor>
    {
        std::vector<Tensor> state;
        state.reserve(2 + 2 * num_blocos_residuais);
        state.push_back(Tensor(batch_size, (Index) num_ocultos)); // mem_in
        state.back().setZero();
        state.push_back(Tensor(batch_size, (Index) num_outputs)); // mem_out
        state.back().setZero();
        for (int i = 0; i < num_blocos_residuais; ++i)
        {
            state.push_back(Tensor(batch_size, (Index) num_ocultos));
            state.back().setZero();
            state.push_back(Tensor(batch_size, (Index) num_ocultos));
            state.back().setZero();
        }
        return state;
    }

    // Forward that accepts an explicit mutable state vector. Updates state in-place.
    // Supports 2D input (batch, features) or 3D input (T, batch, features).
    auto forward_with_state(const Tensor& input, std::vector<Tensor>& state,
                            bool /*requires_grad*/ = true) -> Tensor
    {
        if (state.size() != static_cast<size_t>(2 + 2 * num_blocos_residuais))
        {
            state = inicializar_estado(input.rows());
        }

        // Load state into layer v_mem fields
        lif_in.v_mem = state[0];
        lif_out.v_mem = state[1];
        for (int i = 0; i < num_blocos_residuais; ++i)
        {
            res_blocks[i].lif1.v_mem = state[2 + 2 * i];
            res_blocks[i].lif2.v_mem = state[2 + 2 * i + 1];
        }

        auto shp = input.get_shape();
        // Single-step 2D input
        if (shp.size() == 2)
        {
            if (static_cast<int>(input.cols()) != num_inputs)
                throw std::invalid_argument(
                    "Input feature dimension does not match model num_inputs");

            Tensor corrente = input;
            Tensor out = fc_in.forward(corrente);
            Tensor spk = lif_in.forward(out);

            for (int i = 0; i < num_blocos_residuais; ++i)
            {
                Tensor t = res_blocks[i].fc1.forward(spk);
                Tensor s1 = res_blocks[i].lif1.forward(t);
                Tensor t2 = res_blocks[i].fc2.forward(s1);
                Tensor s2 = res_blocks[i].lif2.forward(t2);
                spk = s2 + spk;
            }

            Tensor out2 = fc_out.forward(spk);
            Tensor spk_out = lif_out.forward(out2);

            // Write back state
            state[0] = lif_in.v_mem;
            state[1] = lif_out.v_mem;
            for (int i = 0; i < num_blocos_residuais; ++i)
            {
                state[2 + 2 * i] = res_blocks[i].lif1.v_mem;
                state[2 + 2 * i + 1] = res_blocks[i].lif2.v_mem;
            }
            return spk_out;
        }

        // Sequence input 3D: (T, batch, features)
        if (shp.size() == 3)
        {
            Index T = static_cast<Index>(shp[0]);
            Index batch = static_cast<Index>(shp[1]);
            Index feats = static_cast<Index>(shp[2]);
            if (static_cast<int>(feats) != num_inputs)
                throw std::invalid_argument(
                    "Input feature dimension does not match model num_inputs");

            // Output tensor: (T, batch, num_outputs)
            Tensor out_seq(std::vector<Index>{T, batch, (Index) num_outputs});
            // Iterate time steps
            for (Index t = 0; t < T; ++t)
            {
                // extract step t into a 2D tensor (batch, features)
                Tensor step(batch, feats);
                for (Index b = 0; b < batch; ++b)
                {
                    for (Index f = 0; f < feats; ++f)
                    {
                        step.at(b, f) = input.at(std::vector<Index>{t, b, f});
                    }
                }

                // forward single step (reusing layer v_mem)
                Tensor out = fc_in.forward(step);
                Tensor spk = lif_in.forward(out);
                for (int i = 0; i < num_blocos_residuais; ++i)
                {
                    Tensor t1 = res_blocks[i].fc1.forward(spk);
                    Tensor s1 = res_blocks[i].lif1.forward(t1);
                    Tensor t2 = res_blocks[i].fc2.forward(s1);
                    Tensor s2 = res_blocks[i].lif2.forward(t2);
                    spk = s2 + spk;
                }
                Tensor out2 = fc_out.forward(spk);
                Tensor spk_out = lif_out.forward(out2);

                // write spk_out into out_seq at index t
                for (Index b = 0; b < batch; ++b)
                {
                    for (Index o = 0; o < (Index) num_outputs; ++o)
                    {
                        out_seq.at(std::vector<Index>{t, b, o}) = spk_out.at(b, o);
                    }
                }
            }

            // Write back final state
            state[0] = lif_in.v_mem;
            state[1] = lif_out.v_mem;
            for (int i = 0; i < num_blocos_residuais; ++i)
            {
                state[2 + 2 * i] = res_blocks[i].lif1.v_mem;
                state[2 + 2 * i + 1] = res_blocks[i].lif2.v_mem;
            }
            return out_seq;
        }

        throw std::invalid_argument("Unsupported input shape for forward_with_state");
    }

    // Module interface
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        // For compatibility with Python `forward(x, estado=None)` we provide a
        // simple forward that uses a temporary state if none is provided.
        std::vector<Tensor> tmp_state = inicializar_estado(input.rows());
        return forward_with_state(input, tmp_state, requires_grad);
    }

    auto backward(const Tensor& /*grad_output*/) -> Tensor override
    {
        // No autograd implemented; return zero grad of appropriate shape placeholder.
        return Tensor::zeros(1, 1);
    }

    void reset_state() override
    {
        // Reset internal state (v_mem) for all LIFs
        lif_in.v_mem = Tensor();
        lif_out.v_mem = Tensor();
        for (auto& b : res_blocks)
        {
            b.lif1.v_mem = Tensor();
            b.lif2.v_mem = Tensor();
        }
    }

    auto params() -> std::vector<Tensor*> override
    {
        std::vector<Tensor*> p;
        // fc_in params
        auto v = fc_in.params();
        p.insert(p.end(), v.begin(), v.end());
        // residual params
        for (int i = 0; i < num_blocos_residuais; ++i)
        {
            auto a = res_blocks[i].fc1.params();
            p.insert(p.end(), a.begin(), a.end());
            auto b = res_blocks[i].fc2.params();
            p.insert(p.end(), b.begin(), b.end());
        }
        // fc_out params
        auto z = fc_out.params();
        p.insert(p.end(), z.begin(), z.end());
        return p;
    }
};

extern "C"
{
    void* criar_modelo_snn(int num_inputs, int num_outputs, int profundidade)
    {
        if (num_outputs <= 0) return nullptr;
        if (profundidade <= 0) profundidade = 3;
        ModeloSNN_Cpp* m = new ModeloSNN_Cpp(num_inputs, num_outputs, profundidade, 100, 0.9f);
        return static_cast<void*>(m);
    }

    void destruir_modelo_snn(void* handle)
    {
        if (!handle) return;
        ModeloSNN_Cpp* m = static_cast<ModeloSNN_Cpp*>(handle);
        delete m;
    }

    char* rede_snn_descricao(void* handle)
    {
        if (!handle) return nullptr;
        ModeloSNN_Cpp* m = static_cast<ModeloSNN_Cpp*>(handle);
        std::string desc = m->description();
        char* out = (char*) malloc(desc.size() + 1);
        std::memcpy(out, desc.c_str(), desc.size() + 1);
        return out;
    }

} // extern "C"
