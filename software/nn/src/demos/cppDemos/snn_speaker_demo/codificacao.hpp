/**
 * @file src/demos/cppDemos/snn_speaker_demo/codificacao.hpp
 * @brief Codificacao.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef DEMOS_CPPDEMOS_CODIFICACAO_HPP
#define DEMOS_CPPDEMOS_CODIFICACAO_HPP

#include <random>
#include <vector>

#include "tensor/Tensor.hpp"

namespace codificacao
{

/**
 * Calcula uma frequência máxima (probabilidade) para o codificador Poisson.
 *
 * Ideia:
 * - Se a média das características (em [0,1]) for alta, diminuímos a frequência
 *   máxima para evitar saturação.
 * - Se a média for baixa, aumentamos para evitar neurônios "mortos".
 *
 * Retorna um escalar (float) para uso como `frequencia_max`.
 *
 * Comportamento (resumido):
 * - Calcula a média das features em [0,1].
 * - Define `frequencia = qtde_de_spikes_esperada_por_passo / mean_val` (aproximação)
 *   e limita em [frequencia_max_min, frequencia_max_max].
 * - Objetivo: ajustar a probabilidade por passo de forma que a expectativa
 *   de spikes por neurônio por passo fique próxima a
 *   `qtde_de_spikes_esperada_por_passo`.
 *
 * Exemplo numérico:
 * - `mean_val = 0.5`, `qtde_de_spikes_esperada_por_passo = 0.1` -> frequencia ~= 0.2
 *   (dado p = x * frequencia, a expectativa E[p] ~ frequencia * mean_val =
 *    0.2 * 0.5 = 0.1).
 * - Com `passos = 10`, espera-se ~1 spike por neurônio por janela (10 * 0.1).
 */
float compute_adaptive_max_rate(const nn::Tensor& signal,
    float expected_spikes_per_step = 0.10f,
    float min_max_rate = 0.02f,
    float max_max_rate = 0.50f,
    float epsilon = 1e-8f);

// Backward-compatible alias for existing callers.
inline float calcular_taxa_max_adaptativa(const nn::Tensor& sinal,
    float qtde_de_spikes_esperada_por_passo = 0.10f,
    float frequencia_max_min = 0.02f,
    float frequencia_max_max = 0.50f,
    float eps = 1e-8f)
{
    return compute_adaptive_max_rate(
        sinal, qtde_de_spikes_esperada_por_passo, frequencia_max_min, frequencia_max_max, eps);
}

/**
 * Codifica características contínuas em trens de spikes via Poisson (rate coding).
 *
 * Entrada:
 * - sinal: tensor [lote, num_features] com valores em [0,1]
 *
 * Saída:
 * - spikes: tensor [passos, lote, num_features] com valores {0,1}
 *
 * Por que Poisson aqui?
 * - É robusto e simples.
 * - Funciona bem quando o objetivo é classificação (biometria por voz) e você quer uma
 *   dinâmica temporal interna por janela (passos > 1).
 *
 * Parâmetro `qtde_de_spikes_esperada_por_passo`:
 * - Interpretação: taxa alvo esperada de spikes por neurônio em CADA passo (valor entre 0 e 1).
 *   Por exemplo, `0.10` → 10% de chance média de disparo por neurônio a cada passo.
 * - Quando `adaptativo=True`, o algoritmo tenta escolher `frequencia_max` tal que
 *   E[p] ≈ `qtde_de_spikes_esperada_por_passo` (onde p é a probabilidade por passo após escala).
 * - Resultado prático: com `passos=N` a expectativa de spikes por neurônio por janela é
 *   aproximadamente `N * qtde_de_spikes_esperada_por_passo`.
 *
 * Exemplo completo:
 * - `x` com média 0.5, `qtde_de_spikes_esperada_por_passo=0.1` → `frequencia_max` ≈ 0.2
 *   → p = x * frequencia_max, E[p] ≈ 0.1. Para `passos=10` espera-se ≈ 1 spike/neuronio/janela.
 */
nn::Tensor encode_poisson(const nn::Tensor& frame,
    int time_steps,
    std::mt19937& random_engine,
    float max_rate = -1.0f,
    bool adaptive_rate = true,
    float expected_spikes_per_step = 0.10f);

// Backward-compatible alias for existing callers.
inline nn::Tensor codificar_poisson(const nn::Tensor& frm,
    int passos,
    std::mt19937& rng,
    float frequencia_max = -1.0f,
    bool adaptativo = true,
    float qtde_de_spikes_esperada_por_passo = 0.10f)
{
    return encode_poisson(
        frm, passos, rng, frequencia_max, adaptativo, qtde_de_spikes_esperada_por_passo);
}

} // namespace codificacao

#endif // DEMOS_CPPDEMOS_CODIFICACAO_HPP
