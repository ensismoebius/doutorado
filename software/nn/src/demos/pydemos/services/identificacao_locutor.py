"""Treinamento e inferência de uma SNN simples para identificação de locutor."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from core.configs import ConfigExtracao, ConfigSNN
from domain.regras import aplicar_limiar_desconhecido
from infra.arquivo_audio import carregar_wav_pcm16
from infra.captura import capturar_audio, reamostrar_audio
from services.conjunto_dados import (
    extrair_janelas_caracteristicas,
    carregar_dataset_janelas,
    listar_amostras_por_pessoa,
)
from services.modelos.rede_snn import criar_modelo_snn
from utils.codificacao import codificar_poisson
from typing import Optional


def treinar_classificador_locutor(
    diretorio_dados: str,
    *,
    cfg_extracao: ConfigExtracao,
    cfg_snn: ConfigSNN,
    epocas: int = 5,
    taxa_aprendizado: float = 1e-3,
    device: str | None = None,
    num_blocos_residuais: int | None = None,
    tamanho_camada_oculta: int = 100,
    loss_mode: str = "rate",
    num_passes: int = 1,
    dataset_override: tuple[Any, Any, Any] | None = None,
    max_samples: int | None = None,
    return_stats: bool = False,
) -> tuple[torch.nn.Module, list[str]] | tuple[torch.nn.Module, list[str], dict]:
    """Treina uma SNN simples para classificar janelas por locutor."""

    # Seleciona dispositivo automaticamente caso não seja informado.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carrega dataset (janelas) já normalizado em [0,1], a menos que seja
    # fornecido um `dataset_override` (útil para executar múltiplos experiments
    # sem recomputar recursos).
    if dataset_override is None:
        X, y, rotulos = carregar_dataset_janelas(diretorio_dados, cfg=cfg_extracao)
    else:
        X, y, rotulos = dataset_override

    # Opcional: limitar número de amostras para execuções rápidas.
    if max_samples is not None and max_samples > 0 and X.shape[0] > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    num_classes = len(rotulos)
    # Modelo SNN simples (snntorch) com saída = número de pessoas.
    rede_neural = criar_modelo_snn(
        numero_de_entradas=cfg_extracao.num_bandas,
        numero_de_saidas=num_classes,
        tamanho_da_camada_escondida=tamanho_camada_oculta,
        qtde_de_blocos_residuais=num_blocos_residuais,
    ).to(device)
    rede_neural.train()

    opcoes = torch.optim.Adam(rede_neural.parameters(), lr=taxa_aprendizado)

    # Converte array para tensores para treinamento.
    entradas = torch.tensor(X, dtype=torch.float32, device=device)
    alvos = torch.tensor(y, dtype=torch.long, device=device)

    # Treino simples em minibatches (para não estourar memória).

    batch_size = 32
    num_samples = entradas.shape[0]

    for ep in range(epocas):
        # Embaralha índices por época.
        perm = torch.randperm(num_samples, device=device)
        total_loss = 0.0
        correct = 0

        for start_idx in range(0, num_samples, batch_size):
            indice_do_lote = perm[start_idx : start_idx + batch_size]
            lote_de_entrada_atual = entradas[indice_do_lote]
            lote_de_alvo_atual = alvos[indice_do_lote]

            # Codifica características contínuas em trens de spikes.
            pulsos_de_entrada = codificar_poisson(
                lote_de_entrada_atual,
                passos=cfg_snn.passos_por_janela,
                adaptativo=True,
                qtde_de_spikes_esperada_por_passo=cfg_snn.alvo_spikes_por_passo,
            )

            # Multi-pass forward (reduz variância estocástica da codificação)
            spk_runs = []
            mem_runs = []
            for _ in range(max(1, num_passes)):
                res = rede_neural(pulsos_de_entrada, None)
                # Model may return (spk, state) or (spk, state, mem_trace)
                if isinstance(res, tuple) and len(res) >= 3:
                    s = res[0]
                    m = res[2]
                elif isinstance(res, tuple) and len(res) == 2:
                    s, m_candidate = res
                    # second return was previously `state` dict; we don't have mem trace
                    if isinstance(m_candidate, dict):
                        m = torch.zeros_like(s)
                    else:
                        m = (
                            m_candidate
                            if m_candidate is not None
                            else torch.zeros_like(s)
                        )
                else:
                    # unexpected return: treat as spikes only
                    s = res
                    m = torch.zeros_like(s)

                spk_runs.append(s)
                mem_runs.append(m)

            # spk_runs: list of [T, B, C] -> stack -> [num_passes, T, B, C]
            spk_out_seq = torch.stack(spk_runs, dim=0).mean(dim=0)
            mem_trace = torch.stack(mem_runs, dim=0).mean(dim=0)

            # Compute loss according to selected mode
            loss = compute_loss(
                loss_mode, spk_out_seq, mem_trace, lote_de_alvo_atual, cfg_snn
            )

            # Training status log (epoch, batch, mode, passes, batch loss)
            batch_idx = start_idx // batch_size + 1
            total_batches = (num_samples + batch_size - 1) // batch_size
            print(
                f"[Treino] Ep {ep+1}/{epocas} | Lote {batch_idx}/{total_batches} | modo={loss_mode} | passes={num_passes} | loss={float(loss.detach().cpu()):.4f}"
            )

            opcoes.zero_grad(set_to_none=True)
            loss.backward()
            opcoes.step()

            total_loss += float(loss.detach().cpu()) * int(
                lote_de_entrada_atual.shape[0]
            )
            # Use spike counts as readout for accuracy reporting (fallback for all modes).
            quantidade_de_pulsos = spk_out_seq.sum(dim=0)
            pred = torch.argmax(quantidade_de_pulsos, dim=1)
            correct += int((pred == lote_de_alvo_atual).sum().detach().cpu())

        acc = correct / num_samples
        print(
            f"[Treino] Época {ep+1}/{epocas} | loss={total_loss/num_samples:.4f} | acc={acc:.3f}"
        )

    final_stats = {"loss": float(total_loss / num_samples), "acc": float(acc)}
    if return_stats:
        return rede_neural, rotulos, final_stats
    return rede_neural, rotulos


def salvar_modelo_e_rotulos(
    rede_neural: torch.nn.Module,
    rotulos: list[str],
    *,
    caminho_modelo: str,
    caminho_rotulos: str,
) -> None:
    # Garante diretório e salva pesos + rótulos em JSON.
    os.makedirs(os.path.dirname(caminho_modelo) or ".", exist_ok=True)
    torch.save(rede_neural.state_dict(), caminho_modelo)
    # Também persiste a profundidade (se disponível) para reconstruir a arquitetura ao recarregar.
    profundidade = getattr(rede_neural, "num_blocos_residuais", None)
    tamanho_camada_oculta = getattr(rede_neural, "num_ocultos", None)
    meta = {"rotulos": rotulos}
    if profundidade is not None:
        meta["profundidade"] = int(profundidade)
    if tamanho_camada_oculta is not None:
        meta["tamanho_camada_oculta"] = int(tamanho_camada_oculta)
    with open(caminho_rotulos, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def carregar_modelo_e_rotulos(
    *,
    caminho_modelo: str,
    caminho_rotulos: str,
    num_inputs: int,
    device: str | None = None,
    num_blocos_residuais: int | None = None,
    tamanho_camada_oculta: int | None = None,
) -> tuple[torch.nn.Module, list[str]]:
    # Usa GPU se disponível, senão CPU.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carrega os rótulos na mesma ordem usada no treino.
    with open(caminho_rotulos, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rotulos = meta["rotulos"]

    # Se a profundidade não foi fornecida explicitamente, tenta ler do arquivo de rótulos.
    profundidade_meta = meta.get("profundidade")
    profundidade_final = (
        num_blocos_residuais if num_blocos_residuais is not None else profundidade_meta
    )
    # Hidden size: prefer explicit argument, else metadata saved at training time.
    tamanho_oculto_meta = meta.get("tamanho_camada_oculta")
    tamanho_oculto_final = (
        tamanho_camada_oculta
        if tamanho_camada_oculta is not None
        else tamanho_oculto_meta
    )

    # Load state dict first so we can infer hidden size if metadata missing.
    sd = torch.load(caminho_modelo, map_location=device)

    # Try to infer hidden size from arguments, metadata, or checkpoint shapes.
    if tamanho_camada_oculta is not None:
        tamanho_para_modelo = int(tamanho_camada_oculta)
    elif tamanho_oculto_meta is not None:
        tamanho_para_modelo = int(tamanho_oculto_meta)
    else:
        # Fallback: inspect checkpoint parameter shapes (fc_in.weight.out_features)
        if "fc_in.weight" in sd:
            tamanho_para_modelo = int(sd["fc_in.weight"].shape[0])
        else:
            tamanho_para_modelo = 100

    # Reconstrói o modelo com a inferência de hidden size.
    rede_neural = criar_modelo_snn(
        numero_de_entradas=num_inputs,
        numero_de_saidas=len(rotulos),
        tamanho_da_camada_escondida=tamanho_para_modelo,
        qtde_de_blocos_residuais=profundidade_final,
    ).to(device)

    # Load weights (will raise if incompatible shapes remain)
    rede_neural.load_state_dict(sd)
    rede_neural.eval()
    return rede_neural, rotulos


@torch.no_grad()
def identificar_locutor_por_microfone(
    rede_neural: torch.nn.Module,
    rotulos: list[str],
    *,
    cfg_extracao: ConfigExtracao,
    cfg_snn: ConfigSNN,
    duracao: float,
    taxa_amostragem: int,
    device: str | None = None,
) -> tuple[str, float, np.ndarray]:
    """Captura áudio, extrai janelas e retorna (predição, confiança, probs)."""

    # Reutiliza o device do modelo se não for informado.
    if device is None:
        device = next(rede_neural.parameters()).device

    # Captura áudio e extrai características por janela.
    audio = capturar_audio(duracao, taxa_amostragem)
    caracs = extrair_janelas_caracteristicas(audio, cfg=cfg_extracao)
    if not caracs:
        raise RuntimeError("Nenhuma janela gerada na captura.")

    # Agrega probabilidades por janela para obter decisão por enunciado.
    probs_utt = torch.zeros((len(rotulos),), device=device)

    for c in caracs:
        xb = torch.tensor(c, dtype=torch.float32, device=device).unsqueeze(0)
        spk_in = codificar_poisson(
            xb,
            passos=cfg_snn.passos_por_janela,
            adaptativo=True,
            qtde_de_spikes_esperada_por_passo=cfg_snn.alvo_spikes_por_passo,
        )
        res = rede_neural(spk_in, None)
        if isinstance(res, tuple) and len(res) >= 1:
            spk_out_seq = res[0]
        else:
            spk_out_seq = res
        contagem = spk_out_seq.sum(dim=0).squeeze(0)
        # Softmax sobre a contagem de spikes para obter "confiança" relativa.
        probs = F.softmax(contagem, dim=0)
        probs_utt += probs

    probs_utt = probs_utt / float(len(caracs))
    idx = int(torch.argmax(probs_utt).detach().cpu())
    return (
        rotulos[idx],
        float(probs_utt[idx].detach().cpu()),
        probs_utt.detach().cpu().numpy(),
    )


@torch.no_grad()
def identificar_locutor_por_wav(
    model: torch.nn.Module,
    rotulos: list[str],
    *,
    caminho_wav: str,
    cfg_extracao: ConfigExtracao,
    cfg_snn: ConfigSNN,
    device: str | None = None,
) -> tuple[str, float, np.ndarray]:
    """Identifica o locutor a partir de um arquivo WAV."""

    # Reutiliza o device do modelo se não for informado.
    if device is None:
        device = next(model.parameters()).device

    # Carrega WAV e reamostra se necessário.
    audio, info = carregar_wav_pcm16(caminho_wav)
    if info.taxa_amostragem != cfg_extracao.taxa_amostragem:
        audio = reamostrar_audio(
            audio,
            taxa_origem=info.taxa_amostragem,
            taxa_destino=cfg_extracao.taxa_amostragem,
        )

    caracs = extrair_janelas_caracteristicas(audio, cfg=cfg_extracao)
    if not caracs:
        raise RuntimeError(f"Nenhuma janela gerada para: {caminho_wav}")

    # Acumula probabilidades por janela para obter decisão global.
    probs_utt = torch.zeros((len(rotulos),), device=device)
    for c in caracs:
        xb = torch.tensor(c, dtype=torch.float32, device=device).unsqueeze(0)
        spk_in = codificar_poisson(
            xb,
            passos=cfg_snn.passos_por_janela,
            adaptativo=True,
            qtde_de_spikes_esperada_por_passo=cfg_snn.alvo_spikes_por_passo,
        )
        res = model(spk_in, None)
        if isinstance(res, tuple) and len(res) >= 1:
            spk_out_seq = res[0]
        else:
            spk_out_seq = res
        contagem = spk_out_seq.sum(dim=0).squeeze(0)
        probs_utt += F.softmax(contagem, dim=0)

    probs_utt = probs_utt / float(len(caracs))
    idx = int(torch.argmax(probs_utt).detach().cpu())
    return (
        rotulos[idx],
        float(probs_utt[idx].detach().cpu()),
        probs_utt.detach().cpu().numpy(),
    )


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    tgt = torch.zeros((target.shape[0], num_classes), device=target.device)
    tgt.scatter_(1, target.view(-1, 1), 1.0)
    return tgt


def apply_psc_kernel(spk: torch.Tensor, tau_steps: int = 5) -> torch.Tensor:
    """Aplica um kernel exponencial causal ao longo da dimensão temporal (dim=0).

    spk: [T, B, C]
    retorna: [T, B, C] (filtrado)
    """
    T = spk.shape[0]
    if T == 0:
        return spk
    alpha = float(torch.exp(torch.tensor(-1.0 / max(1, tau_steps))))
    filt = torch.zeros_like(spk)
    prev = torch.zeros_like(spk[0])
    for t in range(T):
        prev = prev * alpha + spk[t]
        filt[t] = prev
    return filt


def compute_loss(
    loss_mode: str,
    spk_out_seq: torch.Tensor,
    mem_trace: Optional[torch.Tensor],
    target: torch.Tensor,
    cfg_snn: ConfigSNN,
) -> torch.Tensor:
    """Computa a loss a partir do modo selecionado.

    spk_out_seq: [T, B, C]
    mem_trace: [T, B, D] or None
    target: [B] (long)
    """
    T, B = spk_out_seq.shape[0], spk_out_seq.shape[1]
    num_classes = spk_out_seq.shape[2]

    device = spk_out_seq.device

    if loss_mode in ("rate", "monte_carlo"):
        counts = spk_out_seq.sum(dim=0)  # [B, C]
        return F.cross_entropy(counts, target)

    # Construir vetor alvo one-hot para losses contínuas
    target_vec = _one_hot(target.to(device), num_classes)

    if loss_mode == "temporal_pooling":
        num_windows = getattr(cfg_snn, "num_windows", 2)
        window_size = max(1, T // num_windows)
        # Trunca para múltiplo de window_size
        usable_T = window_size * num_windows
        spk = spk_out_seq[:usable_T]
        spk = spk.view(num_windows, window_size, B, num_classes)
        pooled = spk.mean(dim=1)  # [num_windows, B, C]
        readout = pooled.mean(dim=0)  # [B, C]
        return F.cross_entropy(readout, target)

    if loss_mode == "van_rossum":
        tau_steps = getattr(cfg_snn, "tau_psc_steps", 5)
        filtered = apply_psc_kernel(spk_out_seq, tau_steps=tau_steps)
        readout = filtered.sum(dim=0)  # [B, C]
        return F.mse_loss(readout, target_vec)

    if loss_mode == "membrane":
        # Prefer membrane-based readout, but only if mem_trace is a tensor
        # that participates in the autograd graph. Otherwise fallback to
        # spike-count readout to keep gradients flowing.
        if not torch.is_tensor(mem_trace) or not getattr(
            mem_trace, "requires_grad", False
        ):
            print(
                f"[WARN] membrane readout unavailable or not differentiable; falling back to spike-count loss"
            )
            readout = spk_out_seq.sum(dim=0)
            return F.cross_entropy(readout, target)
        else:
            readout = mem_trace.sum(dim=0)
            return F.mse_loss(readout, target_vec)

    if loss_mode == "cosine":
        if mem_trace is None:
            readout = spk_out_seq.mean(dim=0)
        else:
            readout = mem_trace.mean(dim=0)
        # If readout isn't differentiable, fallback to spike-count readout
        if not torch.is_tensor(readout) or not getattr(readout, "requires_grad", False):
            counts = spk_out_seq.sum(dim=0)
            cos = F.cosine_similarity(counts, target_vec, dim=1)
            return 1.0 - cos.mean()
        # cosine similarity per-sample over classes
        cos = F.cosine_similarity(readout, target_vec, dim=1)
        return 1.0 - cos.mean()

    if loss_mode == "mse_vector":
        if mem_trace is None:
            readout = spk_out_seq.mean(dim=0)
        else:
            readout = mem_trace.mean(dim=0)
        if not torch.is_tensor(readout) or not getattr(readout, "requires_grad", False):
            counts = spk_out_seq.sum(dim=0)
            return F.mse_loss(counts, target_vec)
        return F.mse_loss(readout, target_vec)

    # Default fallback
    counts = spk_out_seq.sum(dim=0)
    return F.cross_entropy(counts, target)
