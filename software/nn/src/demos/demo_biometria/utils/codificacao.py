"""Codificação de sinais contínuos em spikes via Poisson (rate coding)."""

import torch


def calcular_taxa_max_adaptativa(
    sinal: torch.Tensor,
    *,
    qtde_de_spikes_esperada_por_passo: float = 0.10,
    frequencia_max_min: float = 0.02,
    frequencia_max_max: float = 0.50,
    eps: float = 1e-8,
) -> float:
    """Calcula uma frequência máxima (probabilidade) para o codificador Poisson.

    Ideia:
    - Se a média das características (em [0,1]) for alta, diminuímos a frequência máxima para evitar saturação.
    - Se a média for baixa, aumentamos para evitar neurônios "mortos".

    Retorna um escalar (float) para uso como `frequencia_max`.

    Comportamento (resumido):
    - Calcula a média das features em [0,1].
    - Define `frequencia = qtde_de_spikes_esperada_por_passo / mean_val` (aproximação) e
      limita em [frequencia_max_min, frequencia_max_max].
    - Objetivo: ajustar a probabilidade por passo de forma que a expectativa
      de spikes por neurônio por passo fique próxima a `qtde_de_spikes_esperada_por_passo`.

    Exemplo numérico:
    - `mean_val = 0.5`, `qtde_de_spikes_esperada_por_passo = 0.1` -> frequencia ~= 0.2
      (dado p = x * frequencia, a expectativa E[p] ~ frequencia * mean_val = 0.2 * 0.5 = 0.1).
    - Com `passos = 10`, espera-se ~1 spike por neurônio por janela (10 * 0.1).
    """

    # Média do vetor em [0, 1] (usa CPU para evitar warnings em logs).
    mean_val = float(torch.mean(sinal).detach().cpu())

    # Esperado: E[p] ~ frequencia_max * mean_val. Ajusta frequencia_max para bater o alvo.
    if mean_val < eps:
        frequencia = frequencia_max_max
    else:
        frequencia = qtde_de_spikes_esperada_por_passo / (mean_val + eps)

    return float(max(frequencia_max_min, min(frequencia_max_max, frequencia)))


def codificar_poisson(
    sinal: torch.Tensor,
    *,
    passos: int,
    frequencia_max: float | None = None,
    adaptativo: bool = True,
    qtde_de_spikes_esperada_por_passo: float = 0.10,
) -> torch.Tensor:
    """Codifica características contínuas em trens de spikes via Poisson (rate coding).

    Entrada:
    - sinal: tensor [lote, num_features] com valores em [0,1]

    Saída:
    - spikes: tensor [passos, lote, num_features] com valores {0,1}

    Por que Poisson aqui?
    - É robusto e simples.
    - Funciona bem quando o objetivo é classificação (biometria por voz) e você quer uma
      dinâmica temporal interna por janela (passos > 1).

    Parâmetro `qtde_de_spikes_esperada_por_passo`:
    - Interpretação: taxa alvo esperada de spikes por neurônio em CADA passo (valor entre 0 e 1).
      Por exemplo, `0.10` → 10% de chance média de disparo por neurônio a cada passo.
    - Quando `adaptativo=True`, o algoritmo tenta escolher `frequencia_max` tal que
      E[p] ≈ `qtde_de_spikes_esperada_por_passo` (onde p é a probabilidade por passo após escala).
    - Resultado prático: com `passos=N` a expectativa de spikes por neurônio por janela é
      aproximadamente `N * qtde_de_spikes_esperada_por_passo`.

    Exemplo completo:
    - `x` com média 0.5, `qtde_de_spikes_esperada_por_passo=0.1` → `frequencia_max` ≈ 0.2
      → p = x * frequencia_max, E[p] ≈ 0.1. Para `passos=10` espera-se ≈ 1 spike/neuronio/janela.
    """

    if passos < 1:
        raise ValueError("`passos` deve ser >= 1")

    # Garante domínio esperado da codificação.
    x = torch.clamp(sinal, 0.0, 1.0)

    # Taxa máxima default caso não haja ajuste adaptativo.
    if frequencia_max is None:
        frequencia_max = 0.25

    if adaptativo:
        frequencia_max = calcular_taxa_max_adaptativa(
            x,
            qtde_de_spikes_esperada_por_passo=qtde_de_spikes_esperada_por_passo,
            frequencia_max_min=0.02,
            frequencia_max_max=0.50,
        )

    # Probabilidade por feature por passo.
    # p: probabilidade (por passo) que cada neurônio dispare.
    # p = x * frequencia_max (clamp em [0,1]).
    # Expectativa por passo: E[p] ≈ frequencia_max * mean(x) ≈ qtde_de_spikes_esperada_por_passo quando adaptativo.
    p = torch.clamp(x * float(frequencia_max), 0.0, 1.0)

    # Amostragem Bernoulli por passo:
    # spikes[t] = 1 se rand < p.
    rand = torch.rand((passos,) + p.shape, device=p.device, dtype=p.dtype)
    spikes = (rand < p.unsqueeze(0)).to(dtype=p.dtype)
    return spikes
