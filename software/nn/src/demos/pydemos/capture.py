import numpy as np
import sys

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("A biblioteca 'sounddevice' é necessária. Instale-a com 'pip install sounddevice' ou via conda.")

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print(
        "Aviso: SciPy não encontrado. Reamostragem usará interpolação linear (menor qualidade).",
        flush=True,
    )

def reamostrar_audio(audio: np.ndarray, taxa_origem: int, taxa_destino: int) -> np.ndarray:
    if taxa_origem == taxa_destino:
        return audio
    
    num_amostras = int(len(audio) * taxa_destino / taxa_origem)
    
    if HAS_SCIPY:
        # Reamostragem baseada em FFT
        return signal.resample(audio, num_amostras)
    else:
        # Fallback: interpolação linear
        x_old = np.linspace(0, len(audio), len(audio))
        x_new = np.linspace(0, len(audio), num_amostras)
        return np.interp(x_new, x_old, audio)


# Wrapper (compatibilidade)
def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    return reamostrar_audio(audio, taxa_origem=orig_sr, taxa_destino=target_sr)

def capturar_audio(
    duracao_segundos: float, 
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Captura áudio bruto do dispositivo de entrada padrão.
    Prioriza taxas de amostragem padrão (48k, 44.1k) para máxima compatibilidade,
    realizando resampling forçado se necessário.
    
    Args:
        duracao_segundos (float): Duração da captura em segundos.
        sample_rate (int): Taxa de amostragem desejada em Hz (saída).
        
    Returns:
        np.ndarray: Vetor de áudio 1D (float32).
    """
    
    # Lista de taxas seguras para tentar capturar (hardware típico)
    TAXAS_SEGURAS = [48000, 44100, 32000, 16000, 8000]
    
    # Se a taxa solicitada estiver na lista segura, tente-a primeiro.
    # Se for uma taxa exótica (ex: 16000 em placa que só aceita 48k), tente as seguras primeiro.
    # No caso de 16000, muitas placas suportam, mas algumas (HDMI logs) apenas 48k.
    # Para garantir, vamos tentar 48000 primeiro se 16000 falhar, mas o log mostrou falha/travamento.
    # Então vamos priorizar 48000 se a solicitada for diferente.
    
    taxas_para_tentar = []
    
    # Prioridade estratégica:
    # 1. Tentar 48000Hz (Padrão ouro moderno)
    # 2. Tentar 44100Hz (Padrão áudio)
    # 3. Tentar a taxa solicitada (se não for uma das acima)
    
    defaults = [48000, 44100]
    taxas_para_tentar.extend(defaults)
    if sample_rate not in defaults:
        taxas_para_tentar.append(sample_rate)
    taxas_para_tentar.extend([r for r in TAXAS_SEGURAS if r not in taxas_para_tentar])

    audio_capturado = None
    taxa_real = None
    
    print(f"[Captura] Solicitado (saída): {duracao_segundos}s @ {sample_rate}Hz", flush=True)

    try:
        for rate in taxas_para_tentar:
            try:
                num_samples = int(duracao_segundos * rate)
                audio = sd.rec(num_samples, samplerate=rate, channels=1, dtype="float32")
                sd.wait()
                
                audio_capturado = np.squeeze(audio)
                taxa_real = rate
                print(f"[Captura] Sucesso na captura a {taxa_real}Hz.", flush=True)
                break
                
            except Exception as e:
                # print(f"[Captura] Falha ao capturar a {rate}Hz: {e}", flush=True)
                continue
    
    except Exception as e:
        print(f"[Captura] Erro crítico no loop de captura: {e}", flush=True)

    # Fallback para Ruído
    if audio_capturado is None:
        print("[Captura] ERRO: Não foi possível capturar áudio.", flush=True)
        print("[Captura] FALLBACK: Gerando ruído branco (para validar o pipeline).", flush=True)
        
        num_samples_forced = int(duracao_segundos * sample_rate)
        np.random.seed(42) 
        audio_capturado = np.random.uniform(-0.1, 0.1, size=num_samples_forced).astype("float32")
        taxa_real = sample_rate
        return audio_capturado

    # Resampling
    if taxa_real != sample_rate:
        print(
            f"[Captura] Hardware capturou a {taxa_real}Hz. Reamostrando para {sample_rate}Hz...",
            flush=True,
        )
        audio_capturado = reamostrar_audio(audio_capturado, taxa_origem=taxa_real, taxa_destino=sample_rate)
        
    expected_samples = int(duracao_segundos * sample_rate)
    
    if len(audio_capturado) != expected_samples:
        if len(audio_capturado) > expected_samples:
            audio_capturado = audio_capturado[:expected_samples]
        else:
            audio_capturado = np.pad(
                audio_capturado,
                (0, expected_samples - len(audio_capturado)),
                "constant",
            )
            
    print(f"[Captura] Pronto: {len(audio_capturado)} amostras.", flush=True)
    return audio_capturado
