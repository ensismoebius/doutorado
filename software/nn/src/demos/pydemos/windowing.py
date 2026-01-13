import numpy as np

def aplicar_janelamento(
    signal: np.ndarray, 
    window_size: int, 
    hop_size: int, 
    window_fn=np.hanning
) -> list[np.ndarray]:
    """
    Segmenta o sinal em janelas sobrepostas com função de janelamento explícita.
    
    Args:
        signal (np.ndarray): Sinal de entrada 1D.
        window_size (int): Tamanho da janela em amostras.
        hop_size (int): Passo entre janelas em amostras.
        window_fn (callable): Função geradora da janela (ex: np.hanning, np.hamming).
        
    Returns:
        list[np.ndarray]: Lista de janelas processadas.
    """
    num_samples = len(signal)
    janelas = []
    
    if num_samples < window_size:
        print("[Windowing] Aviso: Sinal menor que o tamanho da janela.")
        return []
        
    # Gerar a janela uma única vez
    window = window_fn(window_size)
    
    # Iterar sobre o sinal com passo fixo (hop_size)
    for start in range(0, num_samples - window_size + 1, hop_size):
        end = start + window_size
        segment = signal[start:end]
        
        # Aplicar a função de janela (multiplicação de elementos)
        windowed_segment = segment * window
        janelas.append(windowed_segment)
        
    print(f"[Windowing] Geradas {len(janelas)} janelas (Size: {window_size}, Hop: {hop_size})")
    return janelas
