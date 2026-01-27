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


def treinar_classificador_locutor(
    diretorio_dados: str,
    *,
    cfg_extracao: ConfigExtracao,
    cfg_snn: ConfigSNN,
    epocas: int = 5,
    taxa_aprendizado: float = 1e-3,
    device: str | None = None,
    num_blocos_residuais: int | None = None,
) -> tuple[torch.nn.Module, list[str]]:
    """Treina uma SNN simples para classificar janelas por locutor."""

    # Seleciona dispositivo automaticamente caso não seja informado.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carrega dataset (janelas) já normalizado em [0,1].
    X, y, rotulos = carregar_dataset_janelas(diretorio_dados, cfg=cfg_extracao)

    num_classes = len(rotulos)
    # Modelo SNN simples (snntorch) com saída = número de pessoas.
    model = criar_modelo_snn(
        numero_de_entradas=cfg_extracao.num_bandas,
        numero_de_saidas=num_classes,
        qtde_de_blocos_residuais=num_blocos_residuais,
    ).to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=taxa_aprendizado)

    # Converte para tensores para treinamento.
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    # Treino simples em minibatches (para não estourar memória).
    batch = 128
    n = X_t.shape[0]

    for ep in range(epocas):
        # Embaralha índices por época.
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        correct = 0

        for i0 in range(0, n, batch):
            idx = perm[i0 : i0 + batch]
            xb = X_t[idx]
            yb = y_t[idx]

            # Codifica características contínuas em trens de spikes.
            spk_in = codificar_poisson(
                xb,
                passos=cfg_snn.passos_por_janela,
                adaptativo=True,
                qtde_de_spikes_esperada_por_passo=cfg_snn.alvo_spikes_por_passo,
            )

            # Conta spikes por neurônio de saída (soma no tempo).
            spk_out_seq, _ = model(spk_in, None)
            contagem = spk_out_seq.sum(dim=0)  # [lote, classes]

            # Perda de classificação sobre contagens de spikes.
            loss = F.cross_entropy(contagem, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu()) * int(xb.shape[0])
            pred = torch.argmax(contagem, dim=1)
            correct += int((pred == yb).sum().detach().cpu())

        acc = correct / n
        print(
            f"[Treino] Época {ep+1}/{epocas} | loss={total_loss/n:.4f} | acc={acc:.3f}"
        )

    return model, rotulos


def salvar_modelo_e_rotulos(
    model: torch.nn.Module,
    rotulos: list[str],
    *,
    caminho_modelo: str,
    caminho_rotulos: str,
) -> None:
    # Garante diretório e salva pesos + rótulos em JSON.
    os.makedirs(os.path.dirname(caminho_modelo) or ".", exist_ok=True)
    torch.save(model.state_dict(), caminho_modelo)
    # Também persiste a profundidade (se disponível) para reconstruir a arquitetura ao recarregar.
    profundidade = getattr(model, "num_blocos_residuais", None)
    meta = {"rotulos": rotulos}
    if profundidade is not None:
        meta["profundidade"] = int(profundidade)
    with open(caminho_rotulos, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def carregar_modelo_e_rotulos(
    *,
    caminho_modelo: str,
    caminho_rotulos: str,
    num_inputs: int,
    device: str | None = None,
    num_blocos_residuais: int | None = None,
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

    # Reconstrói o modelo com a mesma dimensão de saída e profundidade.
    model = criar_modelo_snn(
        numero_de_entradas=num_inputs,
        numero_de_saidas=len(rotulos),
        qtde_de_blocos_residuais=profundidade_final,
    ).to(device)
    sd = torch.load(caminho_modelo, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, rotulos


@torch.no_grad()
def identificar_locutor_por_microfone(
    model: torch.nn.Module,
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
        device = next(model.parameters()).device

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
        spk_out_seq, _ = model(spk_in, None)
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
        spk_out_seq, _ = model(spk_in, None)
        contagem = spk_out_seq.sum(dim=0).squeeze(0)
        probs_utt += F.softmax(contagem, dim=0)

    probs_utt = probs_utt / float(len(caracs))
    idx = int(torch.argmax(probs_utt).detach().cpu())
    return (
        rotulos[idx],
        float(probs_utt[idx].detach().cpu()),
        probs_utt.detach().cpu().numpy(),
    )


__all__ = [
    "ConfigSNN",
    "treinar_classificador_locutor",
    "salvar_modelo_e_rotulos",
    "carregar_modelo_e_rotulos",
    "identificar_locutor_por_microfone",
    "identificar_locutor_por_wav",
    "aplicar_limiar_desconhecido",
]
