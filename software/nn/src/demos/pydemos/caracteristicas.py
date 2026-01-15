"""Extração de características via energia WPT (Wavelet Packet Transform)."""

import numpy as np

try:
    import pywt
except ImportError:
    raise ImportError(
        "A biblioteca 'PyWavelets' é necessária. Instale-a via pip ou conda."
    )


def calcular_energia_wpt(
    sinal: np.ndarray,
    wavelet_base: str = "db4",
    nivel_maximo: int = 6,
    num_bandas: int = 100,
) -> np.ndarray:
    """Calcula um vetor de energia por banda via Wavelet Packet Transform (WPT).

    Entrada:
    - `sinal`: uma janela de áudio (1D)

    Saída:
    - vetor 1D de tamanho `num_bandas` contendo a energia por banda.

    Observações:
    - A WPT gera 2^nivel_maximo bandas “naturais”. Se `num_bandas` for diferente, fazemos interpolação
      para padronizar o tamanho de saída.
    """

    # Se o sinal for muito curto para o nível solicitado, reduz o nível.
    if len(sinal) < 2**nivel_maximo:
        nivel_maximo = int(np.log2(max(1, len(sinal))))

    # 1) Decomposição Wavelet Packet até o nível `nivel_maximo`.
    wp = pywt.WaveletPacket(
        data=sinal, wavelet=wavelet_base, mode="symmetric", maxlevel=nivel_maximo
    )

    # 2) Obter os nós do nível mais profundo (ordenados por frequência).
    nos = [no.data for no in wp.get_level(nivel_maximo, order="freq")]

    # 3) Energia por nó: E = sum(x^2).
    energias_folhas = np.array([np.sum(n**2) for n in nos])

    if len(energias_folhas) == 0:
        return np.zeros(num_bandas)

    # 4) Ajustar para `num_bandas` via interpolação (padroniza o tamanho de saída).
    if len(energias_folhas) != num_bandas:
        idx_origem = np.linspace(0, 1, len(energias_folhas))
        idx_destino = np.linspace(0, 1, num_bandas)
        energia_final = np.interp(idx_destino, idx_origem, energias_folhas)
    else:
        energia_final = energias_folhas

    return energia_final


def compute_wpt_energy(
    signal: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 6,
    num_bands: int = 100,
) -> np.ndarray:
    """Wrapper (compatibilidade): mantém a API antiga em inglês."""

    return calcular_energia_wpt(
        sinal=signal,
        wavelet_base=wavelet,
        nivel_maximo=max_level,
        num_bandas=num_bands,
    )
