"""Run selected configs with an 80/20 validation split and report results.

Writes results to `experiments/hp_with_val.csv`.
"""

from __future__ import annotations

import csv
import os
import pathlib
import random
import sys
import time
from typing import Any

# ensure package root
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

import torch
import numpy as np
from core.configs import ConfigExtracao, ConfigSNN
from services.identificacao_locutor import (
    treinar_classificador_locutor,
)
from services.conjunto_dados import carregar_dataset_janelas
from utils.codificacao import codificar_poisson
import torch.nn.functional as F

DATA_DIR = "dados/vozes"
RESULTS_CSV = "experiments/hp_with_val.csv"
EPOCHS = 10
SEED = 42
TRAIN_FRAC = 0.8

# Top 3 configs (from previous quick search)
configs = [
    {"lr": 1e-3, "hidden": 32, "loss_mode": "van_rossum", "num_passes": 1},
    {"lr": 1e-3, "hidden": 64, "loss_mode": "van_rossum", "num_passes": 1},
    {"lr": 1e-3, "hidden": 64, "loss_mode": "membrane", "num_passes": 1},
]

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# Preload dataset
cfg_extr = ConfigExtracao(
    taxa_amostragem=44100,
    tamanho_janela=512,
    tamanho_passo=256,
    wavelet_base="db4",
    num_bandas=100,
)
cfg_snn = ConfigSNN(passos_por_janela=5, alvo_spikes_por_passo=0.1)
print("Loading full dataset (may take a moment)...")
X, y, rotulos = carregar_dataset_janelas(DATA_DIR, cfg=cfg_extr)
print(f"Dataset loaded: {X.shape[0]} samples, {len(rotulos)} labels")

# create train/val split
n = X.shape[0]
indices = list(range(n))
random.seed(SEED)
random.shuffle(indices)
cut = int(n * TRAIN_FRAC)
train_idx = indices[:cut]
val_idx = indices[cut:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")


def evaluate(model, X_eval, y_eval, cfg_extr, cfg_snn, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    n = X_eval.shape[0]
    batch_size = 32
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.tensor(
                X_eval[start : start + batch_size], dtype=torch.float32, device=device
            )
            yb = torch.tensor(
                y_eval[start : start + batch_size], dtype=torch.long, device=device
            )
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
            counts = spk_out_seq.sum(dim=0)  # [B, C]
            pred = torch.argmax(counts, dim=1)
            correct += int((pred == yb).sum().cpu())
    return correct / float(n)


# Run each config
header = [
    "lr",
    "hidden",
    "loss_mode",
    "num_passes",
    "time_s",
    "rotulos_len",
    "train_loss",
    "train_acc",
    "val_acc",
]
with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    for cfg in configs:
        print("Running:", cfg)
        start = time.time()
        model, labels, stats = treinar_classificador_locutor(
            DATA_DIR,
            cfg_extracao=cfg_extr,
            cfg_snn=cfg_snn,
            epocas=EPOCHS,
            taxa_aprendizado=cfg["lr"],
            num_blocos_residuais=1,
            tamanho_camada_oculta=cfg["hidden"],
            loss_mode=cfg["loss_mode"],
            num_passes=cfg["num_passes"],
            dataset_override=(X_train, y_train, rotulos),
            max_samples=None,
            return_stats=True,
        )
        elapsed = time.time() - start
        train_loss = stats.get("loss")
        train_acc = stats.get("acc")
        val_acc = evaluate(model, X_val, y_val, cfg_extr, cfg_snn)
        row = {
            "lr": cfg["lr"],
            "hidden": cfg["hidden"],
            "loss_mode": cfg["loss_mode"],
            "num_passes": cfg["num_passes"],
            "time_s": round(elapsed, 2),
            "rotulos_len": len(labels),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        writer.writerow(row)
        f.flush()
        print("Done:", row)

print("All runs finished. Results in:", RESULTS_CSV)
