import numpy as np

try:
    import pywt
except ImportError as e:
    raise ImportError(
        "A biblioteca 'PyWavelets' é necessária. Instale-a com 'pip install PyWavelets' ou via conda."
    ) from e


def calcular_wpt_level(
    duration: float,
    sample_rate: int,
    window_size: int,
    num_bands: int,
    wavelet: str,
) -> int:
    """Escolhe dinamicamente o nível de WPT usando PyWavelets.

    Regra:
    - Considera o tamanho efetivo disponível: min(window_size, duration*sample_rate)
    - Limita pelo máximo estrutural: floor(log2(effective_window))
    - Escolhe o menor nível tal que len(get_level(level)) >= num_bands

    Observação:
    - Esse cálculo usa a API do PyWavelets para ficar alinhado ao comportamento do WaveletPacket.
    """

    total_samples = int(duration * sample_rate)
    effective_window = max(1, min(window_size, total_samples))

    max_level_by_size = int(np.floor(np.log2(effective_window)))
    if max_level_by_size < 1:
        return 1

    dummy = np.zeros(effective_window, dtype=np.float32)
    wp = pywt.WaveletPacket(
        data=dummy, wavelet=wavelet, mode="symmetric", maxlevel=max_level_by_size
    )

    chosen_level = max_level_by_size
    for level in range(1, max_level_by_size + 1):
        nodes = wp.get_level(level, order="freq")
        if len(nodes) >= num_bands:
            chosen_level = level
            break

    return chosen_level


def extrair_features_wavelet(janela, num_features=100, wavelet="db4"):
    """(Legado) Extração baseada em DWT + LFCC.

    Este pipeline atual usa WPT via `features.compute_wpt_energy`.
    Esta função foi mantida apenas para referência; requer um módulo `lfcc.py`.
    """

    try:
        from lfcc import compute_lfcc  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Função legada: requer o módulo 'lfcc'. "
            "Use `features.compute_wpt_energy` (WPT) ou adicione um `lfcc.py`."
        ) from e

    w = pywt.Wavelet(wavelet)  # type: ignore[attr-defined]
    max_level = pywt.dwt_max_level(len(janela), w.dec_len)  # type: ignore[attr-defined]
    coefs = pywt.wavedec(janela, wavelet, level=max_level)  # type: ignore[attr-defined]
    coef_vetor = np.concatenate(coefs)
    return compute_lfcc(coef_vetor, sample_rate=16000, num_ceps=num_features)
