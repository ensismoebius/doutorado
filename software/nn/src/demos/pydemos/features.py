import numpy as np

try:
    import pywt
except ImportError:
    raise ImportError("A biblioteca 'PyWavelets' é necessária. Instale-a via pip ou conda.")

def compute_wpt_energy(
    signal: np.ndarray, 
    wavelet: str = 'db4', 
    max_level: int = 6,
    num_bands: int = 100
) -> np.ndarray:
    """
    Calcula a energia por banda utilizando Decomposição Wavelet Packet (WPT).
    
    Args:
        signal (np.ndarray): Janela de sinal de áudio.
        wavelet (str): Nome da wavelet base (e.g., 'db4').
        max_level (int): Nível máximo de decomposição.
        num_bands (int): Número de bandas de frequência desejadas na saída.
        
    Returns:
        np.ndarray: Vetor de energias (dimensão = num_bands).
    """
    # Check if signal is large enough for the requested level
    # pywt usually handles this, but robust check is good
    if len(signal) < 2**max_level:
       max_level = int(np.log2(len(signal)))
       
    # 1. Decomposição Wavelet Packet completa até o nível max_level
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    
    # 2. Obter os nós (bandas) do nível mais profundo
    # 'freq' garante que os nós estejam ordenados por frequência (natural vs gray code)
    nodes = [node.data for node in wp.get_level(max_level, order='freq')]
    
    # 3. Calcular a energia de cada nó: E = sum(x^2)
    # Resultado é vetor de energia por banda natural da wavelet (tamanho 2^max_level)
    leaf_energies = np.array([np.sum(n**2) for n in nodes])
    
    # Validação rápida de integridade da saída
    if len(leaf_energies) == 0:
        return np.zeros(num_bands)

    # 4. Ajustar para num_bands (Interpolation/Resamling)
    # Se decompusemos em 64 bandas (level 6) mas queremos 100, interpolamos linearmente o espectro de energia.
    # Se level 7 (128 bandas) -> 100, também interpolamos.
    if len(leaf_energies) != num_bands:
        original_indices = np.linspace(0, 1, len(leaf_energies))
        target_indices = np.linspace(0, 1, num_bands)
        final_energy = np.interp(target_indices, original_indices, leaf_energies)
    else:
        final_energy = leaf_energies

    # 5. Normalização ou Log-scaling (opcional, mas recomendado para SNNs)
    # A tarefa pede "Output a 100-dimensional feature vector" based on energy.
    # Vamos manter energia pura, SNN lida com magnitude via threshold ou weights
    
    return final_energy
