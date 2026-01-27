"""Cálculo auxiliar de nível WPT e função de compatibilidade legada."""

import numpy as np

try:
    import pywt
except ImportError as e:
    raise ImportError(
        "A biblioteca 'PyWavelets' é necessária. Instale-a com 'pip install PyWavelets' ou via conda."
    ) from e


def calcular_nivel_wpt(
    duracao: float,
    taxa_amostragem: int,
    tamanho_janela: int,
    num_bandas: int,
    wavelet_base: str,
) -> int:
    """Escolhe dinamicamente o nível de WPT usando PyWavelets.

    Regra:
    - Considera o tamanho efetivo disponível: min(tamanho_janela, duracao*taxa_amostragem)
    - Limita pelo máximo estrutural: floor(log2(janela_efetiva))
    - Escolhe o menor nível tal que len(get_level(nível)) >= num_bandas
    """

    # Define a quantidade de amostras efetiva por janela.
    total_amostras = int(duracao * taxa_amostragem)
    janela_efetiva = max(1, min(tamanho_janela, total_amostras))

    # Nível máximo permitido pelo tamanho (2^nivel <= janela_efetiva).
    nivel_max_por_tamanho = int(np.floor(np.log2(janela_efetiva)))
    if nivel_max_por_tamanho < 1:
        return 1

    # Escolha direta: menor nível L tal que 2**L >= num_bandas.
    if num_bandas <= 1:
        nivel_escolhido = 1
    else:
        # Calcula o nível necessário: 2^nivel >= num_bandas
        nivel_necessario = int(np.ceil(np.log2(num_bandas)))

        # Escolhe o nível final considerando o máximo permitido.
        nivel_escolhido = min(nivel_max_por_tamanho, max(1, nivel_necessario))

    return nivel_escolhido


def calcular_wpt_level(
    duration: float,
    sample_rate: int,
    window_size: int,
    num_bands: int,
    wavelet: str,
) -> int:
    """Wrapper (compatibilidade): mantém a API antiga em inglês."""

    return calcular_nivel_wpt(
        duracao=duration,
        taxa_amostragem=sample_rate,
        tamanho_janela=window_size,
        num_bandas=num_bands,
        wavelet_base=wavelet,
    )


def extrair_features_wavelet(janela, num_features=100, wavelet="db4"):
    """(Legado) Extração baseada em DWT + LFCC.

    Este demo atual usa WPT via `caracteristicas.calcular_energia_wpt`.
    Esta função foi mantida apenas para referência; requer um módulo `lfcc.py`.
    """

    try:
        from lfcc import compute_lfcc  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Função legada: requer o módulo 'lfcc'. "
            "Use `caracteristicas.calcular_energia_wpt` (WPT) ou adicione um `lfcc.py`."
        ) from e

    w = pywt.Wavelet(wavelet)  # type: ignore[attr-defined]
    max_level = pywt.dwt_max_level(len(janela), w.dec_len)  # type: ignore[attr-defined]
    coefs = pywt.wavedec(janela, wavelet, level=max_level)  # type: ignore[attr-defined]
    coef_vetor = np.concatenate(coefs)
    return compute_lfcc(coef_vetor, sample_rate=16000, num_ceps=num_features)
