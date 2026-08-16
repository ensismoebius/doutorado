#!/usr/bin/env python3
"""
Generate comprehensive SNN autoencoder test profiles for parameter grid search.
Explores various hyperparameter combinations to find optimal settings for convergence.
"""

import json
import os
import argparse


DEFAULT_OUTPUT_DIR = "src/experiments/autoencoderRunner/profiles/snnAutoEncodersProfiles"

# Base template - audio-window-snn (most stable variant)
BASE_PROFILE = {
    "window_audio_config": {
        "overlap": 0.5,
        "sample_rate": 44100,
        "window_size": 11025
    },
    "window_eeg_config": {
        "overlap": 0.5,
        "sample_rate": 1024,
        "window_size": 256
    },
    "neural_network_audio_features": 11025,
    "neural_network_eeg_features": 0,
    "neural_network_depth": 2,
    "neural_network_hidden_size": 64,
    "neural_network_input_features": 0,
    "neural_network_latent_size": 32,
    "neural_network_layer_sizes": [64, 64],
    "neural_network_type": "audio-window-snn",
    "dataset_type": "audio-window",
    "program_prefetch_lookahead": 5,
    "dataset_subject_filter_regex": "^S(\\d+)$",
    "dataset_root_path": "~/database.sqlite",
    "training_optimizer_type": "adam",
    "training_normalize_inputs": True,
    "validation_modality_diagnostics_enabled": False,
    "kfold_enabled": False,
    "sampler_shuffle_seed": 42,
    "neural_network_layer": [
        "encoder:linear:64:leaky",
        "encoder:linear:64:leaky",
        "encoder:linear:latent:leaky",
        "decoder:linear:64:leaky",
        "decoder:linear:64:leaky",
        "decoder:linear:output:leaky_integrator"
    ]
}

# Parameter exploration grids
param_grids = {
    "eeg-window-snn": {
        "learning_rates": [0.0001, 0.0003, 0.0005, 0.001, 0.01],
        "batch_sizes": [8, 16, 32, 64, 128, 256],
        "hidden_sizes": [16, 32, 64, 128],
        "latent_sizes": [8, 16, 32, 64],
        "loss_types": ["mse", "mae"],
        "depths": [1, 2, 3, 4, 5],
        "epochs": [2, 5, 10, 15],
        "max_batches_per_epoch": [16, 32, 64, 128],
        "lr_plateau_enabled": [True, False],
    },
    "audio-window-snn": {
        "learning_rates": [0.0001, 0.0003, 0.0005, 0.001, 0.01],
        "batch_sizes": [8, 16, 32, 64, 128, 256],
        "hidden_sizes": [16, 32, 64, 128],
        "latent_sizes": [8, 16, 32, 64],
        "loss_types": ["mse", "mae"],
        "depths": [1, 2, 3, 4, 5],
        "epochs": [2, 5, 10, 15],
        "max_batches_per_epoch": [16, 32, 64, 128],
        "lr_plateau_enabled": [True, False],
    },
    "fused-window-snn": {
        "learning_rates": [0.0001, 0.0003, 0.0005, 0.001, 0.01],
        "batch_sizes": [4, 8, 16, 32, 64, 128, 256],
        "hidden_sizes": [16, 32, 64, 128],
        "latent_sizes": [8, 16, 32, 64],
        "loss_types": ["mse", "mae"],
        "depths": [1, 2, 3, 4, 5],
        "epochs": [2, 5, 10, 15],
        "max_batches_per_epoch": [16, 32, 64, 128],
        "lr_plateau_enabled": [True, False],
    }
}

def create_audio_window_profiles(output_dir: str):
    """Generate audio-window-snn profiles with parameter variations."""
    os.makedirs(output_dir, exist_ok=True)
    
    params = param_grids["audio-window-snn"]
    profile_idx = 0

    # Create an exaustive grid of profiles for audio-window-snn iterating over all parameter combinations
    for lr in params["learning_rates"]:
        for bs in params["batch_sizes"]:
            for hs in params["hidden_sizes"]:
                for ls in params["latent_sizes"]:
                    for loss in params["loss_types"]:
                        for max_batches in params["max_batches_per_epoch"]:
                            for epochs in params["epochs"]:
                                for lr_plateau_enabled in params["lr_plateau_enabled"]:
                                    profile = BASE_PROFILE.copy()
                                    profile.update({
                                        "training_batch_size": bs,
                                        "training_learning_rate": lr,
                                        "neural_network_hidden_size": hs,
                                        "neural_network_latent_size": ls,
                                        "training_loss_type": loss,
                                        "training_max_batches_per_epoch": max_batches,
                                        "training_epochs": epochs,
                                        "training_lr_plateau_enabled": lr_plateau_enabled,
                                        "training_lr_plateau_factor": 0.5,
                                        "training_lr_plateau_patience": 2,
                                    })
                            
                                    variant_str = (
                                        f"lr{lr:.4f}_bs{bs}_hs{hs}_ls{ls}_{loss}"
                                        f"_mb{max_batches}_ep{epochs}_lp{int(lr_plateau_enabled)}"
                                    )
                                    filename = f"{output_dir}/audio-window-snn_grid_{profile_idx:03d}_{variant_str}.json"
                                    
                                    with open(filename, 'w') as f:
                                        json.dump(profile, f, indent=2)
                                    
                                    profile_idx += 1
                                    print(f"Created {filename}")    
 


def create_fused_window_profiles(output_dir: str):
    """Generate fused-window-snn profiles with parameter variations."""
    os.makedirs(output_dir, exist_ok=True)
    
    fused_base = BASE_PROFILE.copy()
    fused_base.update({
        "neural_network_audio_features": 11025,
        "neural_network_eeg_features": 256,
        "neural_network_type": "fused-window-snn",
        "dataset_type": "fused-window",
    })
    
    params = param_grids["fused-window-snn"]
    profile_idx = 0

    for lr in params["learning_rates"]:
        for bs in params["batch_sizes"]:
            for hs in params["hidden_sizes"]:
                for ls in params["latent_sizes"]:
                    for loss in params["loss_types"]:
                        for max_batches in params["max_batches_per_epoch"]:
                            for epochs in params["epochs"]:
                                for lr_plateau_enabled in params["lr_plateau_enabled"]:
                                    profile = fused_base.copy()
                                    profile.update({
                                        "training_batch_size": bs,
                                        "training_learning_rate": lr,
                                        "neural_network_hidden_size": hs,
                                        "neural_network_latent_size": ls,
                                        "training_loss_type": loss,
                                        "training_max_batches_per_epoch": max_batches,
                                        "training_epochs": epochs,
                                        "training_lr_plateau_enabled": lr_plateau_enabled,
                                        "training_lr_plateau_factor": 0.5,
                                        "training_lr_plateau_patience": 2,
                                    })

                                    variant_str = (
                                        f"lr{lr:.4f}_bs{bs}_hs{hs}_ls{ls}_{loss}"
                                        f"_mb{max_batches}_ep{epochs}_lp{int(lr_plateau_enabled)}"
                                    )
                                    filename = f"{output_dir}/fused-window-snn_grid_{profile_idx:03d}_{variant_str}.json"

                                    with open(filename, 'w') as f:
                                        json.dump(profile, f, indent=2)

                                    profile_idx += 1
                                    print(f"Created {filename}")


def create_eeg_window_profiles(output_dir: str):
    """Generate eeg-window-snn profiles with parameter variations."""
    os.makedirs(output_dir, exist_ok=True)
    
    eeg_base = BASE_PROFILE.copy()
    eeg_base.update({
        "neural_network_audio_features": 0,
        "neural_network_eeg_features": 256,
        "neural_network_type": "eeg-window-snn",
        "dataset_type": "eeg-window",
    })
    
    params = param_grids["eeg-window-snn"]
    profile_idx = 0
    for lr in params["learning_rates"]:
        for bs in params["batch_sizes"]:
            for hs in params["hidden_sizes"]:
                for ls in params["latent_sizes"]:
                    for loss in params["loss_types"]:
                        for max_batches in params["max_batches_per_epoch"]:
                            for epochs in params["epochs"]:
                                for lr_plateau_enabled in params["lr_plateau_enabled"]:
                                    profile = eeg_base.copy()
                                    profile.update({
                                        "training_batch_size": bs,
                                        "training_learning_rate": lr,
                                        "neural_network_hidden_size": hs,
                                        "neural_network_latent_size": ls,
                                        "training_loss_type": loss,
                                        "training_max_batches_per_epoch": max_batches,
                                        "training_epochs": epochs,
                                        "training_lr_plateau_enabled": lr_plateau_enabled,
                                        "training_lr_plateau_factor": 0.5,
                                        "training_lr_plateau_patience": 2,
                                    })

                                    variant_str = (
                                        f"lr{lr:.4f}_bs{bs}_hs{hs}_ls{ls}_{loss}"
                                        f"_mb{max_batches}_ep{epochs}_lp{int(lr_plateau_enabled)}"
                                    )
                                    filename = f"{output_dir}/eeg-window-snn_grid_{profile_idx:03d}_{variant_str}.json"

                                    with open(filename, 'w') as f:
                                        json.dump(profile, f, indent=2)

                                    profile_idx += 1
                                    print(f"Created {filename}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for profile generation."""
    parser = argparse.ArgumentParser(description="Generate SNN autoencoder test profiles")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where generated JSON profiles are written (default: {DEFAULT_OUTPUT_DIR})",
    )
    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir

    print("Generating SNN autoencoder test profiles...")
    create_audio_window_profiles(output_dir)
    print("\n" + "="*60 + "\n")
    create_fused_window_profiles(output_dir)
    print("\n" + "="*60 + "\n")
    create_eeg_window_profiles(output_dir)
    
    # Count total profiles
    profile_count = len([f for f in os.listdir(output_dir) if f.endswith('.json')])
    print(f"\nTotal profiles created: {profile_count}")
