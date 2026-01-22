#include "codificacao.hpp"

#include <algorithm>
#include <stdexcept>

namespace codificacao
{

float calcular_taxa_max_adaptativa(          //
    const nn::Tensor& sinal,                 //
    float qtde_de_spikes_esperada_por_passo, //
    float frequencia_max_min,                //
    float frequencia_max_max,                //
    float eps                                //
)
{
    float mean_val = sinal.mean();

    if (mean_val < eps)
    {
        return frequencia_max_max;
    }

    float frequencia = qtde_de_spikes_esperada_por_passo / (mean_val + eps);

    return std::clamp(frequencia, frequencia_max_min, frequencia_max_max);
}

nn::Tensor codificar_poisson(               //
    const nn::Tensor& sinal,                //
    int passos,                             //
    std::mt19937& rng,                      //
    float frequencia_max,                   //
    bool adaptativo,                        //
    float qtde_de_spikes_esperada_por_passo //
)
{
    if (passos < 1)
    {
        throw std::invalid_argument("passos must be >= 1");
    }

    // x = clamp(frm, 0, 1)
    nn::Tensor x = sinal.clamp(0.0f, 1.0f);

    // default frequencia_max
    if (frequencia_max <= 0.0f)
    {
        frequencia_max = 0.25f;
    }

    if (adaptativo)
    {
        frequencia_max = calcular_taxa_max_adaptativa( //
            x,                                         //
            qtde_de_spikes_esperada_por_passo,         //
            0.02f,                                     //
            0.50f,                                     //
            1e-8f                                      //
        );
    }

    // Probabilidade por feature por passo.
    // p: probabilidade (por passo) que cada neurônio dispare.
    // p = x * frequencia_max (clamp em [0,1]).
    // Expectativa por passo: E[p] ≈ frequencia_max * mean(x) ≈ qtde_de_spikes_esperada_por_passo
    // quando adaptativo. p = clamp(x * frequencia_max, 0, 1)
    nn::Tensor p = x.multiply_scalar(frequencia_max).clamp(0.0f, 1.0f);

    // rand = torch.rand((passos,) + p.shape, device=p.device, dtype=p.dtype)
    nn::Tensor rand = nn::Tensor::rand(static_cast<nn::Index>(passos), p.cols(), rng);

    // Amostragem Bernoulli por passo:
    // spikes[t] = 1 se rand < p.
    nn::Tensor spikes = rand < p;
    return spikes;
}

} // namespace codificacao
