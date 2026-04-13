#!/usr/bin/env python3
"""
Generate comprehensive SNN autoencoder test profiles for parameter grid search.
Explores various hyperparameter combinations to find optimal settings for convergence.
"""

import json
import os
from itertools import product

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
    "dataset_root_path": "/home/ensismoebius/Documentos/UNESP/doutorado/databases/BaseDeDatosHablaImaginada",
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
    "audio-window-snn": {
        "learning_rates": [0.0001, 0.0003, 0.0005, 0.001],
        "batch_sizes": [8, 16, 32, 64],
        "hidden_sizes": [32, 64, 128],
        "latent_sizes": [16, 32, 64],
        "loss_types": ["mse", "mae"],
        "depths": [1, 2, 3],
        "epochs": [5, 10, 15],
        "max_batches_per_epoch": [20, 40, 60],
        "lr_plateau_enabled": [True, False],
    },
    "fused-window-snn": {
        "learning_rates": [0.0001, 0.0003, 0.0005],
        "batch_sizes": [4, 8, 16, 32],
        "hidden_sizes": [64, 128],
        "latent_sizes": [32, 64],
        "loss_types": ["mse", "mae"],
        "depths": [2, 3],
        "epochs": [5, 10],
        "max_batches_per_epoch": [20, 40],
        "lr_plateau_enabled": [True],
    }
}

def create_audio_window_profiles():
    """Generate audio-window-snn profiles with parameter variations."""
    output_dir = "src/experiments/03/profiles/tests"
    os.makedirs(output_dir, exist_ok=True)
    
    params = param_grids["audio-window-snn"]
    profile_idx = 0
    
    # Select subsets to avoid explosion of combinations
    # Main effects: learning_rate × batch_size × hidden_size × latent_size × loss_type
    for lr in params["learning_rates"]:
        for bs in params["batch_sizes"]:
            for hs in params["hidden_sizes"]:
                for ls in params["latent_sizes"]:
                    for loss in params["loss_types"]:
                        profile = BASE_PROFILE.copy()
                        profile.update({
                            "training_batch_size": bs,
                            "training_learning_rate": lr,
                            "neural_network_hidden_size": hs,
                            "neural_network_latent_size": ls,
                            "training_loss_type": loss,
                            "training_max_batches_per_epoch": 30,
                            "training_epochs": 8,
                            "training_lr_plateau_enabled": lr > 0.0003,  # Enable for higher LRs
                            "training_lr_plateau_factor": 0.5,
                            "training_lr_plateau_patience": 2,
                            "training_lr_plateau_min_delta": 0.000001,
                        })
                        
                        variant_str = f"lr{lr:.4f}_bs{bs}_hs{hs}_ls{ls}_{loss}"
                        filename = f"{output_dir}/audio-window-snn_grid_{profile_idx:03d}_{variant_str}.json"
                        
                        with open(filename, 'w') as f:
                            json.dump(profile, f, indent=2)
                        
                        profile_idx += 1
                        print(f"Created {filename}")


def create_fused_window_profiles():
    """Generate fused-window-snn profiles with parameter variations."""
    output_dir = "src/experiments/03/profiles/tests"
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
                        profile = fused_base.copy()
                        profile.update({
                            "training_batch_size": bs,
                            "training_learning_rate": lr,
                            "neural_network_hidden_size": hs,
                            "neural_network_latent_size": ls,
                            "training_loss_type": loss,
                            "training_max_batches_per_epoch": 25,
                            "training_epochs": 8,
                            "training_lr_plateau_enabled": True,
                            "training_lr_plateau_factor": 0.5,
                            "training_lr_plateau_patience": 2,
                        })
                        
                        variant_str = f"lr{lr:.4f}_bs{bs}_hs{hs}_ls{ls}_{loss}"
                        filename = f"{output_dir}/fused-window-snn_grid_{profile_idx:03d}_{variant_str}.json"
                        
                        with open(filename, 'w') as f:
                            json.dump(profile, f, indent=2)
                        
                        profile_idx += 1
                        print(f"Created {filename}")


def create_eeg_window_profiles():
    """Generate eeg-window-snn profiles with parameter variations."""
    output_dir = "src/experiments/03/profiles/tests"
    os.makedirs(output_dir, exist_ok=True)
    
    eeg_base = BASE_PROFILE.copy()
    eeg_base.update({
        "neural_network_audio_features": 0,
        "neural_network_eeg_features": 256,
        "neural_network_type": "eeg-window-snn",
        "dataset_type": "eeg-window",
    })
    
    # Simpler grid for single-modality
    lrs = [0.0001, 0.0003, 0.0005]
    bss = [8, 16, 32]
    hs = 64
    ls = 32
    losses = ["mse", "mae"]
    
    profile_idx = 0
    for lr in lrs:
        for bs in bss:
            for loss in losses:
                profile = eeg_base.copy()
                profile.update({
                    "training_batch_size": bs,
                    "training_learning_rate": lr,
                    "neural_network_hidden_size": hs,
                    "neural_network_latent_size": ls,
                    "training_loss_type": loss,
                    "training_max_batches_per_epoch": 30,
                    "training_epochs": 8,
                    "training_lr_plateau_enabled": True,
                    "training_lr_plateau_factor": 0.5,
                    "training_lr_plateau_patience": 2,
                })
                
                variant_str = f"lr{lr:.4f}_bs{bs}_{loss}"
                filename = f"{output_dir}/eeg-window-snn_grid_{profile_idx:03d}_{variant_str}.json"
                
                with open(filename, 'w') as f:
                    json.dump(profile, f, indent=2)
                
                profile_idx += 1
                print(f"Created {filename}")


if __name__ == "__main__":
    print("Generating SNN autoencoder test profiles...")
    create_audio_window_profiles()
    print("\n" + "="*60 + "\n")
    create_fused_window_profiles()
    print("\n" + "="*60 + "\n")
    create_eeg_window_profiles()
    
    # Count total profiles
    output_dir = "src/experiments/03/profiles/tests"
    profile_count = len([f for f in os.listdir(output_dir) if f.endswith('.json')])
    print(f"\nTotal profiles created: {profile_count}")
