"""Runner do protótipo multimodal EEG+Áudio.

Executa:
1) Ingestão e pré-processamento com janela de 100 ms
2) Treino de autoencoder (denso ou spiking)
3) Extração de features AE e wavelet
4) Classificador simples para speaker_id
5) Métricas paraconsistentes
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    # Allow direct execution (python run_prototype.py) while preserving relative imports.
    script_path = Path(__file__).resolve()
    for ancestor_dir in script_path.parents:
        if ancestor_dir.name == "src":
            sys.path.insert(0, str(ancestor_dir))
            relative_path = script_path.relative_to(ancestor_dir)
            __package__ = ".".join(relative_path.parent.parts)
            break

from .config import DataConfig, OutputConfig, PrototypeConfig, TrainConfig
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


class SpeakerLinearClassifier(nn.Module):
    """Minimal linear head used to classify speaker IDs from feature vectors."""

    def __init__(self, input_dimension: int, class_count: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dimension, class_count))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for each class for a batch of feature vectors."""
        return self.classifier(features)


def _set_random_seeds(seed_value: int) -> None:
    """Set Python, NumPy and PyTorch seeds for deterministic prototype runs."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def _load_raw_records_from_config(config: PrototypeConfig) -> list[RawRecord]:
    """Load multimodal records according to the configured input format."""
    if config.data.format == "npz":
        candidate_files = sorted(config.data.data_root.glob("*.npz"))
        if not candidate_files:
            raise FileNotFoundError(
                f"nenhum .npz encontrado em {config.data.data_root}"
            )
        return load_records_from_npz(candidate_files[0])
    if config.data.format == "mat":
        candidate_files = sorted(config.data.data_root.glob("*.mat"))
        if not candidate_files:
            raise FileNotFoundError(
                f"nenhum .mat encontrado em {config.data.data_root}"
            )
        return load_records_from_mat(
            candidate_files[0],
            audio_var=config.data.audio_key,
            eeg_var=config.data.eeg_key,
        )
    if config.data.format == "wav_csv":
        metadata_csv_path = config.data.data_root / "metadata.csv"
        return load_records_from_wav_csv(config.data.data_root, metadata_csv_path)
    raise ValueError(f"formato inválido: {config.data.format}")


def _train_autoencoder_reconstruction(
    autoencoder_model: nn.Module,
    training_loader: DataLoader,
    epoch_count: int,
    learning_rate: float,
    weight_decay: float,
    device_name: str,
) -> list[float]:
    """Train AE reconstruction with MSE and return mean loss per epoch."""
    autoencoder_model.to(device_name)
    optimizer = torch.optim.Adam(
        autoencoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    reconstruction_loss = nn.MSELoss()
    epoch_loss_history: list[float] = []

    autoencoder_model.train()
    for _ in range(epoch_count):
        mini_batch_losses: list[float] = []
        for batch in training_loader:
            input_tensor = batch["x"].to(device_name).float()
            optimizer.zero_grad()
            reconstructed_tensor, _ = autoencoder_model(input_tensor)
            loss = reconstruction_loss(reconstructed_tensor, input_tensor)
            loss.backward()
            optimizer.step()
            mini_batch_losses.append(float(loss.item()))
        epoch_loss_history.append(
            float(np.mean(mini_batch_losses)) if mini_batch_losses else 0.0
        )
    return epoch_loss_history


def _encode_records_with_autoencoder(
    autoencoder_model: nn.Module,
    windowed_records: list[WindowedRecord],
    batch_size: int,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode records into latent vectors and return `(latent_matrix, speaker_ids)`."""
    dataset = MultimodalWindowDataset(windowed_records)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    latent_batches: list[np.ndarray] = []
    speaker_batches: list[np.ndarray] = []
    autoencoder_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_tensor = batch["x"].to(device_name).float()
            _, latent_tensor = autoencoder_model(input_tensor)
            latent_batches.append(latent_tensor.cpu().numpy().astype(np.float32))
            speaker_batches.append(np.asarray(batch["speaker_id"], dtype=np.int64))
    return np.concatenate(latent_batches, axis=0), np.concatenate(speaker_batches, axis=0)


def _extract_wavelet_feature_matrix(
    windowed_records: list[WindowedRecord], wavelet_family: str, max_level: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract wavelet features from each record and return `(features, labels)`."""
    wavelet_feature_rows: list[np.ndarray] = []
    speaker_labels: list[int] = []
    for record in windowed_records:
        wavelet_feature_rows.append(
            extract_multimodal_wavelet_features(
                record.audio_win,
                record.eeg_win,
                family=wavelet_family,
                max_level=max_level,
            )
        )
        speaker_labels.append(record.speaker_id)
    return np.vstack(wavelet_feature_rows).astype(np.float32), np.asarray(
        speaker_labels, dtype=np.int64
    )


def _train_and_evaluate_linear_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    device_name: str,
) -> tuple[np.ndarray, float]:
    """Train a simple linear classifier and return `(val_probabilities, val_accuracy)`."""
    unique_classes = np.unique(train_labels)
    class_to_index = {label: idx for idx, label in enumerate(unique_classes.tolist())}
    indexed_train_labels = np.asarray(
        [class_to_index[label] for label in train_labels], dtype=np.int64
    )
    indexed_validation_labels = np.asarray(
        [class_to_index.get(label, 0) for label in validation_labels], dtype=np.int64
    )

    classifier_model = SpeakerLinearClassifier(
        input_dimension=train_features.shape[1], class_count=len(unique_classes)
    ).to(device_name)
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-2)
    classification_loss = nn.CrossEntropyLoss()

    train_feature_tensor = torch.from_numpy(train_features).to(device_name)
    train_label_tensor = torch.from_numpy(indexed_train_labels).to(device_name)

    classifier_model.train()
    for _ in range(100):
        optimizer.zero_grad()
        logits = classifier_model(train_feature_tensor)
        loss = classification_loss(logits, train_label_tensor)
        loss.backward()
        optimizer.step()

    classifier_model.eval()
    with torch.no_grad():
        validation_feature_tensor = torch.from_numpy(validation_features).to(device_name)
        logits = classifier_model(validation_feature_tensor)
        class_probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        predicted_classes = np.argmax(class_probabilities, axis=1)
        validation_accuracy = float(
            np.mean(predicted_classes == indexed_validation_labels)
        )
    return class_probabilities, validation_accuracy


def execute_prototype_pipeline(
    config: PrototypeConfig,
    audio_original_sample_rate: int,
    eeg_original_sample_rate: int,
) -> dict[str, Any]:
    """Run the full multimodal prototype pipeline and return a summary dictionary."""
    _set_random_seeds(config.train.seed)
    output_directory = config.output.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_records = _load_raw_records_from_config(config)
    windowed_records = build_windowed_records(
        raw_records,
        audio_orig_sr=audio_original_sample_rate,
        eeg_orig_sr=eeg_original_sample_rate,
        target_audio_sr=config.preprocess.target_audio_sr,
        target_eeg_sr=config.preprocess.target_eeg_sr,
        window_sec=config.preprocess.window_sec,
        overlap=config.preprocess.overlap,
        zscore_per_window=config.preprocess.zscore_per_window,
    )
    if not windowed_records:
        raise RuntimeError("nenhuma janela foi gerada")

    train_records, validation_records = split_train_val_by_sample(windowed_records)

    input_dimension = int(train_records[0].x_concat.shape[0])
    autoencoder_model = build_autoencoder(
        model_type=config.ae.model_type,
        input_dim=input_dimension,
        latent_dim=config.ae.latent_dim,
        hidden_dims=config.ae.hidden_dims,
        snn_time_steps=config.ae.snn_time_steps,
        snn_dt=config.ae.snn_dt,
        snn_resistance=config.ae.snn_resistance,
        snn_capacitance=config.ae.snn_capacitance,
        snn_threshold=config.ae.snn_threshold,
        snn_surrogate_sharpness=config.ae.snn_surrogate_sharpness,
    )

    train_dataset = MultimodalWindowDataset(train_records)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True
    )

    autoencoder_loss_history = _train_autoencoder_reconstruction(
        autoencoder_model=autoencoder_model,
        training_loader=train_dataloader,
        epoch_count=config.train.epochs,
        learning_rate=config.train.lr,
        weight_decay=config.train.weight_decay,
        device_name=config.train.device,
    )

    train_latent_features, train_speaker_ids = _encode_records_with_autoencoder(
        autoencoder_model, train_records, config.train.batch_size, config.train.device
    )
    validation_latent_features, validation_speaker_ids = _encode_records_with_autoencoder(
        autoencoder_model,
        validation_records,
        config.train.batch_size,
        config.train.device,
    )

    train_wavelet_features, _ = _extract_wavelet_feature_matrix(
        train_records, config.wavelet.family, config.wavelet.max_level
    )
    validation_wavelet_features, _ = _extract_wavelet_feature_matrix(
        validation_records,
        config.wavelet.family,
        config.wavelet.max_level,
    )

    train_combined_features = np.concatenate(
        [train_latent_features, train_wavelet_features], axis=1
    )
    validation_combined_features = np.concatenate(
        [validation_latent_features, validation_wavelet_features], axis=1
    )

    latent_probabilities, latent_accuracy = _train_and_evaluate_linear_classifier(
        train_latent_features,
        train_speaker_ids,
        validation_latent_features,
        validation_speaker_ids,
        config.train.device,
    )
    wavelet_probabilities, wavelet_accuracy = _train_and_evaluate_linear_classifier(
        train_wavelet_features,
        train_speaker_ids,
        validation_wavelet_features,
        validation_speaker_ids,
        config.train.device,
    )
    combined_probabilities, combined_accuracy = _train_and_evaluate_linear_classifier(
        train_combined_features,
        train_speaker_ids,
        validation_combined_features,
        validation_speaker_ids,
        config.train.device,
    )

    validation_reference_labels = validation_speaker_ids.astype(np.int64)
    # Map original labels to the same compact index space used by class probabilities.
    unique_classes = np.unique(train_speaker_ids)
    class_to_index = {label: idx for idx, label in enumerate(unique_classes.tolist())}
    validation_reference_indices = np.asarray(
        [class_to_index.get(label, 0) for label in validation_reference_labels],
        dtype=np.int64,
    )

    latent_mu, latent_lambda = mu_lambda_from_probabilities(
        latent_probabilities, validation_reference_indices
    )
    wavelet_mu, wavelet_lambda = mu_lambda_from_probabilities(
        wavelet_probabilities, validation_reference_indices
    )
    combined_mu, combined_lambda = mu_lambda_from_probabilities(
        combined_probabilities, validation_reference_indices
    )

    latent_certainty, latent_contradiction = certainty_and_contradiction(
        latent_mu, latent_lambda
    )
    wavelet_certainty, wavelet_contradiction = certainty_and_contradiction(
        wavelet_mu, wavelet_lambda
    )
    combined_certainty, combined_contradiction = certainty_and_contradiction(
        combined_mu, combined_lambda
    )

    summary = {
        "train_windows": len(train_records),
        "val_windows": len(validation_records),
        "input_dim": input_dimension,
        "ae_loss_history": autoencoder_loss_history,
        "accuracy": {
            "ae": latent_accuracy,
            "wavelet": wavelet_accuracy,
            "combined": combined_accuracy,
        },
        "paraconsistent": {
            "ae": summarize_paraconsistent(latent_certainty, latent_contradiction),
            "wavelet": summarize_paraconsistent(
                wavelet_certainty, wavelet_contradiction
            ),
            "combined": summarize_paraconsistent(
                combined_certainty, combined_contradiction
            ),
        },
    }

    if config.output.save_features:
        # Save validation-set features for external analyses and plotting.
        np.savez(
            output_directory / "features_ae.npz",
            x=validation_latent_features,
            y=validation_speaker_ids,
        )
        np.savez(
            output_directory / "features_wavelet.npz",
            x=validation_wavelet_features,
            y=validation_speaker_ids,
        )
        np.savez(
            output_directory / "features_combined.npz",
            x=validation_combined_features,
            y=validation_speaker_ids,
        )

    with (output_directory / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    with (output_directory / "config_used.json").open("w", encoding="utf-8") as config_file:
        json.dump(asdict(config), config_file, indent=2, default=str)

    return summary


def run(
    cfg: PrototypeConfig, audio_orig_sr: int, eeg_orig_sr: int
) -> dict[str, Any]:
    """Backward-compatible entrypoint kept for existing imports/callers."""
    return execute_prototype_pipeline(
        config=cfg,
        audio_original_sample_rate=audio_orig_sr,
        eeg_original_sample_rate=eeg_orig_sr,
    )


def parse_cli_arguments() -> argparse.Namespace:
    """Parse command-line arguments used to run the prototype from terminal/debug."""
    argument_parser = argparse.ArgumentParser(
        description="Executa o prototipo multimodal EEG+Audio"
    )
    argument_parser.add_argument("--data-root", type=Path, required=True)
    argument_parser.add_argument(
        "--format", choices=["mat", "npz", "wav_csv"], default="npz"
    )
    argument_parser.add_argument("--audio-orig-sr", type=int, default=44100)
    argument_parser.add_argument("--eeg-orig-sr", type=int, default=1000)
    argument_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/prototipo_multimodal")
    )
    argument_parser.add_argument("--device", type=str, default="cpu")
    argument_parser.add_argument("--epochs", type=int, default=5)
    argument_parser.add_argument("--batch-size", type=int, default=32)
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--lr", type=float, default=1e-3)
    argument_parser.add_argument("--weight-decay", type=float, default=1e-5)
    argument_parser.add_argument("--no-save-features", action="store_true")
    return argument_parser.parse_args()


def build_prototype_config_from_cli(args: argparse.Namespace) -> PrototypeConfig:
    """Build strongly-typed `PrototypeConfig` from CLI args."""
    data_config = DataConfig(
        data_root=args.data_root,
        format=args.format,
    )
    train_config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    output_config = OutputConfig(
        output_dir=args.output_dir,
        save_features=not args.no_save_features,
    )
    return PrototypeConfig(data=data_config, train=train_config, output=output_config)


def main() -> None:
    """CLI entrypoint for local execution and debugger launches."""
    cli_args = parse_cli_arguments()
    runtime_config = build_prototype_config_from_cli(cli_args)
    summary = execute_prototype_pipeline(
        config=runtime_config,
        audio_original_sample_rate=cli_args.audio_orig_sr,
        eeg_original_sample_rate=cli_args.eeg_orig_sr,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
