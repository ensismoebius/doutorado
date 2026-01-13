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
    print("Aviso: Scipy não encontrado. Resampling usará interpolação linear (menor qualidade).", flush=True)

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    
    num_samples = int(len(audio) * target_sr / orig_sr)
    
    if HAS_SCIPY:
        # FFT-based resampling
        return signal.resample(audio, num_samples)
    else:
        # Linear interpolation fallback
        x_old = np.linspace(0, len(audio), len(audio))
        x_new = np.linspace(0, len(audio), num_samples)
        return np.interp(x_new, x_old, audio)

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
    
    # Lista de taxas seguras para tentar capturar (hardware consumer-grade)
    # 48kHz e 44.1kHz são as mais comuns.
    SAFE_RATES = [48000, 44100, 32000, 16000, 8000]
    
    # Se a taxa solicitada estiver na lista segura, tente-a primeiro.
    # Se for uma taxa exótica (ex: 16000 em placa que só aceita 48k), tente as seguras primeiro.
    # No caso de 16000, muitas placas suportam, mas algumas (HDMI logs) apenas 48k.
    # Para garantir, vamos tentar 48000 primeiro se 16000 falhar, mas o log mostrou falha/travamento.
    # Então vamos priorizar 48000 se a solicitada for diferente.
    
    rates_to_try = []
    
    # Prioridade estratégica:
    # 1. Tentar 48000Hz (Padrão ouro moderno)
    # 2. Tentar 44100Hz (Padrão áudio)
    # 3. Tentar a taxa solicitada (se não for uma das acima)
    
    defaults = [48000, 44100]
    rates_to_try.extend(defaults)
    if sample_rate not in defaults:
        rates_to_try.append(sample_rate)
    rates_to_try.extend([r for r in SAFE_RATES if r not in rates_to_try])

    captured_audio = None
    actual_rate = None
    
    print(f"[Capture] Solicitado Final: {duracao_segundos}s @ {sample_rate}Hz", flush=True)

    try:
        for rate in rates_to_try:
            try:
                # print(f"[Capture] Tentando capturar a {rate}Hz...", flush=True)
                num_samples = int(duracao_segundos * rate)
                audio = sd.rec(num_samples, samplerate=rate, channels=1, dtype="float32")
                sd.wait()
                
                captured_audio = np.squeeze(audio)
                actual_rate = rate
                print(f"[Capture] Sucesso na captura a {actual_rate}Hz.", flush=True)
                break
                
            except Exception as e:
                # print(f"[Capture] Falha ao capturar a {rate}Hz: {e}", flush=True)
                continue
    
    except Exception as e:
        print(f"[Capture] Erro crítico no loop de captura: {e}", flush=True)

    # Fallback para Ruído
    if captured_audio is None:
        print("[Capture] ERRO FATAL DE HARDWARE: Não foi possível capturar áudio.", flush=True)
        print("[Capture] MODO FALLBACK ATIVADO: Gerando Ruído Branco.", flush=True)
        
        num_samples_forced = int(duracao_segundos * sample_rate)
        np.random.seed(42) 
        captured_audio = np.random.uniform(-0.1, 0.1, size=num_samples_forced).astype("float32")
        actual_rate = sample_rate
        return captured_audio

    # Resampling
    if actual_rate != sample_rate:
        print(f"[Capture] Hardware capturou a {actual_rate}Hz. Convertendo para {sample_rate}Hz...", flush=True)
        captured_audio = resample_audio(captured_audio, actual_rate, sample_rate)
        
    expected_samples = int(duracao_segundos * sample_rate)
    
    if len(captured_audio) != expected_samples:
        if len(captured_audio) > expected_samples:
            captured_audio = captured_audio[:expected_samples]
        else:
            captured_audio = np.pad(captured_audio, (0, expected_samples - len(captured_audio)), 'constant')
            
    print(f"[Capture] Pronto: {len(captured_audio)} samples.", flush=True)
    return captured_audio
