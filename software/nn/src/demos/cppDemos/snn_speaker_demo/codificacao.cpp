/**
 * @file src/demos/cppDemos/snn_speaker_demo/codificacao.cpp
 * @brief Implementation for Codificacao.
 *

 */

#include "codificacao.hpp"

#include <algorithm>
#include <stdexcept>

namespace codificacao
{

float compute_adaptive_max_rate( //
    const nn::Tensor& signal,    //
    float expected_spikes_per_step,
    float min_max_rate,
    float max_max_rate,
    float epsilon //
)
{
    float mean_val = signal.mean();

    if (mean_val < epsilon)
    {
        return max_max_rate;
    }

    float rate = expected_spikes_per_step / (mean_val + epsilon);

    return std::clamp(rate, min_max_rate, max_max_rate);
}

nn::Tensor encode_poisson(   //
    const nn::Tensor& frame, //
    int time_steps,
    std::mt19937& random_engine,
    float max_rate,
    bool adaptive_rate,
    float expected_spikes_per_step //
)
{
    if (time_steps < 1)
    {
        throw std::invalid_argument("passos must be >= 1");
    }

    // x = clamp(frm, 0, 1)
    nn::Tensor x = frame.clamp(0.0f, 1.0f);

    // default frequencia_max
    if (max_rate <= 0.0f)
    {
        max_rate = 0.25f;
    }

    if (adaptive_rate)
    {
        max_rate = compute_adaptive_max_rate( //
            x,                                //
            expected_spikes_per_step,         //
            0.02f,                            //
            0.50f,                            //
            1e-8f                             //
        );
    }

    // Probabilidade por feature por passo.
    // p: probabilidade (por passo) que cada neurônio dispare.
    // p = x * frequencia_max (clamp em [0,1]).
    // Expectativa por passo: E[p] ≈ frequencia_max * mean(x) ≈ qtde_de_spikes_esperada_por_passo
    // quando adaptativo. p = clamp(x * frequencia_max, 0, 1)
    nn::Tensor p = x.multiply_scalar(max_rate).clamp(0.0f, 1.0f);

    // rand = torch.rand((passos,) + p.shape, device=p.device, dtype=p.dtype)
    nn::Tensor rand = nn::Tensor::rand(static_cast<nn::Index>(time_steps), p.cols(), random_engine);

    // Amostragem Bernoulli por passo:
    // spikes[t] = 1 se rand < p.
    nn::Tensor spikes = rand < p;
    return spikes;
}

} // namespace codificacao
