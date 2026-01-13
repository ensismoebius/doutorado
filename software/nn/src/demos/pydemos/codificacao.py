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
    """

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
    - Funciona bem quando o objetivo é classificação (biometria por voz) e você quer uma dinâmica
      temporal interna por janela (passos > 1).
    """

    if passos < 1:
        raise ValueError("`passos` deve ser >= 1")

    x = torch.clamp(caracteristicas_01, 0.0, 1.0)

    if taxa_max is None:
        taxa_max = 0.25

    if adaptativo:
        taxa_max = calcular_taxa_max_adaptativa(
            x,
            alvo_spikes_por_passo=alvo_spikes_por_passo,
            taxa_max_min=0.02,
            taxa_max_max=0.50,
        )

    # Probabilidade por feature por passo
    p = torch.clamp(x * float(taxa_max), 0.0, 1.0)

    # Amostragem Bernoulli por passo
    # spikes[t] = 1 se rand < p
    rand = torch.rand((passos,) + p.shape, device=p.device, dtype=p.dtype)
    spikes = (rand < p.unsqueeze(0)).to(dtype=p.dtype)
    return spikes
