"""Long training for best candidate and k-fold cross-validation for top candidates.

Outputs:
- experiments/long_train_curve.csv  (epoch, train_loss, train_acc, val_acc)
- experiments/cv_results.csv (candidate, fold, train_loss, train_acc, val_acc, time_s)

Be careful: this runs full training loops and may take time.
"""

from __future__ import annotations

import csv
import os
import pathlib
import random
import sys
import time
from typing import Any, List, Tuple

# ensure package root
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "voice_biometrics_snn_py"
sys.path.insert(0, str(PKG_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from core.configs import ConfigExtracao, ConfigSNN
from services.modelos.rede_snn import criar_modelo_snn
from services.identificacao_locutor import compute_loss
from services.conjunto_dados import carregar_dataset_janelas
from utils.codificacao import codificar_poisson

DATA_DIR = "dados/vozes"

# Long training params (best candidate)
BEST_CFG = {"lr": 1e-3, "hidden": 32, "loss_mode": "van_rossum", "num_passes": 1}
EPOCHS_LONG = 30
BATCH_SIZE = 32

# CV params
TOP_CANDIDATES = [
    {"lr": 1e-3, "hidden": 32, "loss_mode": "van_rossum", "num_passes": 1},
    {"lr": 1e-3, "hidden": 64, "loss_mode": "van_rossum", "num_passes": 1},
    {"lr": 1e-3, "hidden": 64, "loss_mode": "membrane", "num_passes": 1},
]
FOLDS = 5
EPOCHS_CV = 10
SEED = 42

os.makedirs("experiments", exist_ok=True)

# preload dataset
cfg_extr = ConfigExtracao(
    taxa_amostragem=44100,
    tamanho_janela=512,
    tamanho_passo=256,
    wavelet_base="db4",
    num_bandas=100,
)
cfg_snn = ConfigSNN(passos_por_janela=5, alvo_spikes_por_passo=0.1)
print("Loading dataset...")
X, y, rotulos = carregar_dataset_janelas(DATA_DIR, cfg=cfg_extr)
print(f"Loaded {X.shape[0]} samples, {len(rotulos)} classes")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(
    model,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    cfg_extr,
    cfg_snn,
    epochs,
    batch_size,
    loss_mode,
    num_passes,
    device=DEVICE,
):
    model.to(device)
    model.train()
    n = X_train.shape[0]
    history = []
    for ep in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        correct = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            spk_in = codificar_poisson(
                xb,
                passos=cfg_snn.passos_por_janela,
                adaptativo=True,
                qtde_de_spikes_esperada_por_passo=cfg_snn.alvo_spikes_por_passo,
            )

            spk_runs = []
            mem_runs = []
            for _ in range(max(1, num_passes)):
                res = model(spk_in, None)
                if isinstance(res, tuple) and len(res) >= 3:
                    s = res[0]
                    m = res[2]
                elif isinstance(res, tuple) and len(res) == 2:
                    s, m_candidate = res
                    if isinstance(m_candidate, dict):
                        m = torch.zeros_like(s)
                    else:
                        m = (
                            m_candidate
                            if m_candidate is not None
                            else torch.zeros_like(s)
                        )
                else:
                    s = res
                    m = torch.zeros_like(s)
                spk_runs.append(s)
                mem_runs.append(m)

            spk_out_seq = torch.stack(spk_runs, dim=0).mean(dim=0)
            mem_trace = torch.stack(mem_runs, dim=0).mean(dim=0)

            loss = compute_loss(loss_mode, spk_out_seq, mem_trace, yb, cfg_snn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * int(xb.shape[0])
            counts = spk_out_seq.sum(dim=0)
            pred = torch.argmax(counts, dim=1)
            correct += int((pred == yb).sum().detach().cpu())

        train_loss = total_loss / n
        train_acc = correct / n
        val_acc = evaluate(model, X_val, y_val, cfg_extr, cfg_snn, device)
        print(
            f"[Train] Ep {ep+1}/{epochs} | loss={train_loss:.4f} | acc={train_acc:.3f} | val_acc={val_acc:.3f}"
        )
        history.append((train_loss, train_acc, val_acc))
    return history


def evaluate(model, X_eval, y_eval, cfg_extr, cfg_snn, device=DEVICE):
    model.eval()
    n = X_eval.shape[0]
    correct = 0
    bs = BATCH_SIZE
    with torch.no_grad():
        for start in range(0, n, bs):
            xb = torch.tensor(
                X_eval[start : start + bs], dtype=torch.float32, device=device
            )
            yb = torch.tensor(
                y_eval[start : start + bs], dtype=torch.long, device=device
            )
            spk_in = codificar_poisson(
                xb,
                passos=cfg_snn.passos_por_janela,
                adaptativo=True,
                qtde_de_spikes_esperada_por_passo=cfg_snn.alvo_spikes_por_passo,
            )
            res = model(spk_in, None)
            spk_out_seq = res[0] if isinstance(res, tuple) else res
            counts = spk_out_seq.sum(dim=0)
            pred = torch.argmax(counts, dim=1)
            correct += int((pred == yb).sum().cpu())
    return correct / float(n)


# Long training for best candidate
print("Starting long training for best candidate:", BEST_CFG)
# train/val split 80/20
n = X.shape[0]
indices = list(range(n))
random.seed(SEED)
random.shuffle(indices)
cut = int(0.8 * n)
train_idx = indices[:cut]
val_idx = indices[cut:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

model = criar_modelo_snn(
    numero_de_entradas=cfg_extr.num_bandas,
    numero_de_saidas=len(rotulos),
    tamanho_da_camada_escondida=BEST_CFG["hidden"],
    qtde_de_blocos_residuais=1,
).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=BEST_CFG["lr"])
start = time.time()
history = train_loop(
    model,
    opt,
    X_train,
    y_train,
    X_val,
    y_val,
    cfg_extr,
    cfg_snn,
    EPOCHS_LONG,
    BATCH_SIZE,
    BEST_CFG["loss_mode"],
    BEST_CFG["num_passes"],
    device=DEVICE,
)
elapsed = time.time() - start
# write curve
with open("experiments/long_train_curve.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "train_acc", "val_acc"])
    for i, (tl, ta, va) in enumerate(history, 1):
        w.writerow([i, tl, ta, va])
print(
    "Long training finished in",
    round(elapsed, 2),
    "s -> experiments/long_train_curve.csv",
)

# k-fold CV for top candidates
print("Starting k-fold CV for top candidates")
results = []
K = FOLDS
n = X.shape[0]
indices = list(range(n))
random.seed(SEED)
random.shuffle(indices)
fold_sizes = [n // K + (1 if i < n % K else 0) for i in range(K)]
cur = 0
folds = []
for fs in fold_sizes:
    folds.append(indices[cur : cur + fs])
    cur += fs

for cand in TOP_CANDIDATES:
    for fi in range(K):
        # build train/val for this fold
        val_idx = folds[fi]
        train_idx = [i for j, fold in enumerate(folds) if j != fi for i in fold]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        model = criar_modelo_snn(
            numero_de_entradas=cfg_extr.num_bandas,
            numero_de_saidas=len(rotulos),
            tamanho_da_camada_escondida=cand["hidden"],
            qtde_de_blocos_residuais=1,
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cand["lr"])
        print(f"CV candidate {cand} fold {fi+1}/{K}")
        start = time.time()
        history = train_loop(
            model,
            opt,
            X_tr,
            y_tr,
            X_va,
            y_va,
            cfg_extr,
            cfg_snn,
            EPOCHS_CV,
            BATCH_SIZE,
            cand["loss_mode"],
            cand["num_passes"],
            device=DEVICE,
        )
        elapsed = time.time() - start
        tr_loss, tr_acc, val_acc = history[-1]
        results.append(
            {
                "lr": cand["lr"],
                "hidden": cand["hidden"],
                "loss_mode": cand["loss_mode"],
                "num_passes": cand["num_passes"],
                "fold": fi + 1,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_acc": val_acc,
                "time_s": round(elapsed, 2),
            }
        )
        # flush intermediate results
        with open("experiments/cv_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "lr",
                    "hidden",
                    "loss_mode",
                    "num_passes",
                    "fold",
                    "train_loss",
                    "train_acc",
                    "val_acc",
                    "time_s",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)

print("CV finished. Results in experiments/cv_results.csv")
