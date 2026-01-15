"""Codificação de sinais contínuos em spikes via Poisson (rate coding)."""

import torch


def calcular_taxa_max_adaptativa(
    caracteristicas_01: torch.Tensor,
    *,
    alvo_spikes_por_passo: float = 0.10,
    taxa_max_min: float = 0.02,
    taxa_max_max: float = 0.50,
    eps: float = 1e-8,
) -> float:
    """Calcula uma taxa máxima (probabilidade) para o codificador Poisson.

    Ideia:
    - Se a média das características (em [0,1]) for alta, diminuímos a taxa máxima para evitar saturação.
    - Se a média for baixa, aumentamos para evitar neurônios "mortos".

        Retorna um escalar (float) para uso como `taxa_max`.

        Comportamento (resumido):
        - Calcula a média das features em [0,1].
        - Define `taxa = alvo_spikes_por_passo / mean_val` (aproximação) e
            limita em [taxa_max_min, taxa_max_max].
        - Objetivo: ajustar a probabilidade por passo de forma que a expectativa
            de spikes por neurônio por passo fique próxima a `alvo_spikes_por_passo`.

        Exemplo numérico:
        - `mean_val = 0.5`, `alvo_spikes_por_passo = 0.1` -> taxa ~= 0.2
            (dado p = x * taxa, a expectativa E[p] ~ taxa * mean_val = 0.2 * 0.5 = 0.1).
        - Com `passos = 10`, espera-se ~1 spike por neurônio por janela (10 * 0.1).
    """

    # Média do vetor em [0, 1] (usa CPU para evitar warnings em logs).
    mean_val = float(torch.mean(caracteristicas_01).detach().cpu())
    # Esperado: E[p] ~ taxa_max * mean_val. Ajusta taxa_max para bater o alvo.
    if mean_val < eps:
        taxa = taxa_max_max
    else:
        taxa = alvo_spikes_por_passo / (mean_val + eps)

    return float(max(taxa_max_min, min(taxa_max_max, taxa)))


def codificar_poisson(
    caracteristicas_01: torch.Tensor,
    *,
    passos: int,
    taxa_max: float | None = None,
    adaptativo: bool = True,
    alvo_spikes_por_passo: float = 0.10,
) -> torch.Tensor:
    """Codifica características contínuas em trens de spikes via Poisson (rate coding).

    Entrada:
    - caracteristicas_01: tensor [lote, num_features] com valores em [0,1]

    Saída:
    - spikes: tensor [passos, lote, num_features] com valores {0,1}

        Por que Poisson aqui?
        - É robusto e simples.
        - Funciona bem quando o objetivo é classificação (biometria por voz) e você quer uma
            dinâmica temporal interna por janela (passos > 1).

        Parâmetro `alvo_spikes_por_passo`:
        - Interpretação: taxa alvo esperada de spikes por neurônio em CADA passo (valor entre 0 e 1).
            Por exemplo, `0.10` → 10% de chance média de disparo por neurônio a cada passo.
        - Quando `adaptativo=True`, o algoritmo tenta escolher `taxa_max` tal que
            E[p] ≈ `alvo_spikes_por_passo` (onde p é a probabilidade por passo após escala).
        - Resultado prático: com `passos=N` a expectativa de spikes por neurônio por janela é
            aproximadamente `N * alvo_spikes_por_passo`.

        Exemplo completo:
        - `x` com média 0.5, `alvo_spikes_por_passo=0.1` → `taxa_max` ≈ 0.2
            → p = x * taxa_max, E[p] ≈ 0.1. Para `passos=10` espera-se ≈ 1 spike/neuronio/janela.
    """

    if passos < 1:
        raise ValueError("`passos` deve ser >= 1")

    # Garante domínio esperado da codificação.
    x = torch.clamp(caracteristicas_01, 0.0, 1.0)

    # Taxa máxima default caso não haja ajuste adaptativo.
    if taxa_max is None:
        taxa_max = 0.25

    if adaptativo:
        taxa_max = calcular_taxa_max_adaptativa(
            x,
            alvo_spikes_por_passo=alvo_spikes_por_passo,
            taxa_max_min=0.02,
            taxa_max_max=0.50,
        )

    # Probabilidade por feature por passo.
    # p: probabilidade (por passo) que cada neurônio dispare.
    # p = x * taxa_max (clamp em [0,1]).
    # Expectativa por passo: E[p] ≈ taxa_max * mean(x) ≈ alvo_spikes_por_passo quando adaptativo.
    p = torch.clamp(x * float(taxa_max), 0.0, 1.0)

    # Amostragem Bernoulli por passo:
    # spikes[t] = 1 se rand < p.
    rand = torch.rand((passos,) + p.shape, device=p.device, dtype=p.dtype)
    spikes = (rand < p.unsqueeze(0)).to(dtype=p.dtype)
    return spikes
