"""Segmentação de sinais em janelas sobrepostas (com janela de Hann por padrão)."""

import numpy as np


def aplicar_janelamento(
    sinal: np.ndarray,
    tamanho_janela: int,
    tamanho_passo: int,
    funcao_de_janelamento=None,
    **kwargs,
) -> list[np.ndarray]:
    """Segmenta o sinal em janelas sobrepostas com função de janelamento explícita."""

    # Janela de Hann por padrão.
    if funcao_de_janelamento is None:
        funcao_de_janelamento = np.hanning

    # Calcula quantidade de amostras e inicializa a lista de janelas.
    tamanho_do_sinal = len(sinal)

    # Lista de janelas resultantes.
    sinal_janelado = []

    if tamanho_do_sinal < tamanho_janela:
        print("[Janelamento] Aviso: sinal menor que o tamanho da janela.")
        return []

    # Gera os coefficients da janelamento.
    coeficientes_da_janela = funcao_de_janelamento(tamanho_janela)

    # Iterar sobre o sinal com passo fixo (hop_size)
    for inicio in range(0, tamanho_do_sinal - tamanho_janela + 1, tamanho_passo):
        fim = inicio + tamanho_janela

        # Extrai o segmento do sinal
        segmento_do_sinal = sinal[inicio:fim]

        # Aplicar a função de janela (multiplicação de elementos).
        segmento_do_sinal_janelado = segmento_do_sinal * coeficientes_da_janela

        # Adiciona o sinal janelado à lista
        sinal_janelado.append(segmento_do_sinal_janelado)

    print(
        f"[Janelamento] Geradas {len(sinal_janelado)} janelas (tamanho={tamanho_janela}, passo={tamanho_passo})"
    )
    return sinal_janelado
