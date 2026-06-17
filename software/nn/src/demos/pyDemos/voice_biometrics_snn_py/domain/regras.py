"""Regras de domínio para decisão/validação de identidade."""


def aplicar_limiar_desconhecido(
    pessoa: str,
    confianca: float,
    *,
    limiar: float,
    rotulo_desconhecido: str = "desconhecido",
) -> str:
    """Se a confiança for baixa, devolve o rótulo de desconhecido."""

    if confianca < limiar:
        return rotulo_desconhecido
    return pessoa
