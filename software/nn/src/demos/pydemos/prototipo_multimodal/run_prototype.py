"""Runner do protótipo multimodal EEG+Áudio.

Executa:
1) Ingestão e pré-processamento com janela de 100 ms
2) Treino de autoencoder (denso ou spiking)
3) Extração de features AE e wavelet
4) Classificador simples para speaker_id
5) Métricas paraconsistentes
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import PrototypeConfig
from .data_io import (
    RawRecord,
    load_records_from_mat,
    load_records_from_npz,
    load_records_from_wav_csv,
)
from .datasets import MultimodalWindowDataset, split_train_val_by_sample
from .models import build_autoencoder
from .paraconsistent import (
    certainty_and_contradiction,
    mu_lambda_from_probabilities,
    summarize_paraconsistent,
)
from .preprocess import WindowedRecord, build_windowed_records
from .wavelet_features import extract_multimodal_wavelet_features


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_records(cfg: PrototypeConfig) -> list[RawRecord]:
    if cfg.data.format == "npz":
        candidates = sorted(cfg.data.data_root.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"nenhum .npz encontrado em {cfg.data.data_root}")
        return load_records_from_npz(candidates[0])
    if cfg.data.format == "mat":
        candidates = sorted(cfg.data.data_root.glob("*.mat"))
        if not candidates:
            raise FileNotFoundError(f"nenhum .mat encontrado em {cfg.data.data_root}")
        return load_records_from_mat(
            candidates[0], audio_var=cfg.data.audio_key, eeg_var=cfg.data.eeg_key
        )
    if cfg.data.format == "wav_csv":
        csv_path = cfg.data.data_root / "metadata.csv"
        return load_records_from_wav_csv(cfg.data.data_root, csv_path)
    raise ValueError(f"formato inválido: {cfg.data.format}")


def _train_autoencoder(
    model: nn.Module, loader: DataLoader, epochs: int, lr: float, wd: float, device: str
) -> list[float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    history: list[float] = []

    model.train()
    for _ in range(epochs):
        losses: list[float] = []
        for batch in loader:
            x = batch["x"].to(device).float()
            opt.zero_grad()
            recon, _ = model(x)
            loss = crit(recon, x)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)) if losses else 0.0)
    return history


def _encode(
    model: nn.Module, records: list[WindowedRecord], batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    ds = MultimodalWindowDataset(records)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device).float()
            _, z = model(x)
            zs.append(z.cpu().numpy().astype(np.float32))
            ys.append(np.asarray(batch["speaker_id"], dtype=np.int64))
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def _wavelet_matrix(
    records: list[WindowedRecord], family: str, max_level: int
) -> tuple[np.ndarray, np.ndarray]:
    xw: list[np.ndarray] = []
    y: list[int] = []
    for r in records:
        xw.append(
            extract_multimodal_wavelet_features(
                r.audio_win, r.eeg_win, family=family, max_level=max_level
            )
        )
        y.append(r.speaker_id)
    return np.vstack(xw).astype(np.float32), np.asarray(y, dtype=np.int64)


def _train_eval_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
) -> tuple[np.ndarray, float]:
    classes = np.unique(y_train)
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}
    y_train_idx = np.asarray([class_to_idx[c] for c in y_train], dtype=np.int64)
    y_val_idx = np.asarray([class_to_idx.get(c, 0) for c in y_val], dtype=np.int64)

    model = LinearClassifier(in_dim=x_train.shape[1], num_classes=len(classes)).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()

    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(y_train_idx).to(device)

    model.train()
    for _ in range(100):
        opt.zero_grad()
        logits = model(xt)
        loss = ce(logits, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(x_val).to(device)
        logits = model(xv)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_val_idx))
    return probs, acc


def run(cfg: PrototypeConfig, audio_orig_sr: int, eeg_orig_sr: int) -> dict[str, Any]:
    _set_seed(cfg.train.seed)
    out_dir = cfg.output.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_records = _load_records(cfg)
    windowed = build_windowed_records(
        raw_records,
        audio_orig_sr=audio_orig_sr,
        eeg_orig_sr=eeg_orig_sr,
        target_audio_sr=cfg.preprocess.target_audio_sr,
        target_eeg_sr=cfg.preprocess.target_eeg_sr,
        window_sec=cfg.preprocess.window_sec,
        overlap=cfg.preprocess.overlap,
        zscore_per_window=cfg.preprocess.zscore_per_window,
    )
    if not windowed:
        raise RuntimeError("nenhuma janela foi gerada")

    train_records, val_records = split_train_val_by_sample(windowed)

    input_dim = int(train_records[0].x_concat.shape[0])
    ae = build_autoencoder(
        model_type=cfg.ae.model_type,
        input_dim=input_dim,
        latent_dim=cfg.ae.latent_dim,
        hidden_dims=cfg.ae.hidden_dims,
        snn_time_steps=cfg.ae.snn_time_steps,
        snn_dt=cfg.ae.snn_dt,
        snn_resistance=cfg.ae.snn_resistance,
        snn_capacitance=cfg.ae.snn_capacitance,
        snn_threshold=cfg.ae.snn_threshold,
        snn_surrogate_sharpness=cfg.ae.snn_surrogate_sharpness,
    )

    train_ds = MultimodalWindowDataset(train_records)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)

    loss_hist = _train_autoencoder(
        model=ae,
        loader=train_dl,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        wd=cfg.train.weight_decay,
        device=cfg.train.device,
    )

    z_train, y_train = _encode(
        ae, train_records, cfg.train.batch_size, cfg.train.device
    )
    z_val, y_val = _encode(ae, val_records, cfg.train.batch_size, cfg.train.device)

    wav_train, _ = _wavelet_matrix(
        train_records, cfg.wavelet.family, cfg.wavelet.max_level
    )
    wav_val, _ = _wavelet_matrix(val_records, cfg.wavelet.family, cfg.wavelet.max_level)

    comb_train = np.concatenate([z_train, wav_train], axis=1)
    comb_val = np.concatenate([z_val, wav_val], axis=1)

    probs_ae, acc_ae = _train_eval_classifier(
        z_train, y_train, z_val, y_val, cfg.train.device
    )
    probs_wav, acc_wav = _train_eval_classifier(
        wav_train, y_train, wav_val, y_val, cfg.train.device
    )
    probs_comb, acc_comb = _train_eval_classifier(
        comb_train, y_train, comb_val, y_val, cfg.train.device
    )

    y_ref = y_val.astype(np.int64)
    # mapeia rótulos originais para índices 0..K-1 de forma consistente com probs
    classes = np.unique(y_train)
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}
    y_ref_idx = np.asarray([class_to_idx.get(c, 0) for c in y_ref], dtype=np.int64)

    mu_ae, lam_ae = mu_lambda_from_probabilities(probs_ae, y_ref_idx)
    mu_w, lam_w = mu_lambda_from_probabilities(probs_wav, y_ref_idx)
    mu_c, lam_c = mu_lambda_from_probabilities(probs_comb, y_ref_idx)

    gc_ae, gct_ae = certainty_and_contradiction(mu_ae, lam_ae)
    gc_w, gct_w = certainty_and_contradiction(mu_w, lam_w)
    gc_c, gct_c = certainty_and_contradiction(mu_c, lam_c)

    summary = {
        "train_windows": len(train_records),
        "val_windows": len(val_records),
        "input_dim": input_dim,
        "ae_loss_history": loss_hist,
        "accuracy": {"ae": acc_ae, "wavelet": acc_wav, "combined": acc_comb},
        "paraconsistent": {
            "ae": summarize_paraconsistent(gc_ae, gct_ae),
            "wavelet": summarize_paraconsistent(gc_w, gct_w),
            "combined": summarize_paraconsistent(gc_c, gct_c),
        },
    }

    if cfg.output.save_features:
        np.savez(out_dir / "features_ae.npz", x=z_val, y=y_val)
        np.savez(out_dir / "features_wavelet.npz", x=wav_val, y=y_val)
        np.savez(out_dir / "features_combined.npz", x=comb_val, y=y_val)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    raise SystemExit(
        "Use scripts/run_multimodal_prototype.py para executar este protótipo."
    )
