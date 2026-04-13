#!/usr/bin/env python3
"""
Analyze SNN autoencoder hyperparameter grid search results.
Aggregates JSON output from experiment03 runs and identifies patterns/optima.
"""

import json
import os
import sys
from collections import defaultdict
import pandas as pd


def load_results(results_dir="src/experiments/03/results"):
    """Load all JSON result files from experiment03 runs."""
    results = []
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Only include grid search profiles (from tests/)
                if not data.get("profile", "").startswith("audio-window-snn_grid") and \
                   not data.get("profile", "").startswith("fused-window-snn_grid") and \
                   not data.get("profile", "").startswith("eeg-window-snn_grid"):
                    continue
                
                # Extract metrics
                result = {
                    "profile": data.get("profile", ""),
                    "timestamp": filename[:13],
                    "dataset_type": data.get("dataset_type", ""),
                    "autoencoder_type": data.get("autoencoder_type", ""),
                    "exit_code": data.get("exit_code", -1),
                    "processed_samples": data.get("processed_samples", 0),
                    "seen_batches": data.get("seen_batches", 0),
                }
                
                # Loss metrics
                epoch_losses = data.get("epoch_mean_losses", [])
                if epoch_losses:
                    result["initial_loss"] = epoch_losses[0]
                    result["final_loss"] = epoch_losses[-1]
                    result["min_loss"] = min(epoch_losses)
                    result["loss_trend"] = (epoch_losses[-1] - epoch_losses[0]) / max(abs(epoch_losses[0]), 0.001)
                else:
                    result["initial_loss"] = None
                    result["final_loss"] = None
                    result["min_loss"] = None
                    result["loss_trend"] = None
                
                # Validation loss
                result["val_loss"] = data.get("kfold", {}).get("mean_val_loss", None) or data.get("mean_val_loss")
                
                # Learning rate
                result["final_lr"] = data.get("optimizer", {}).get("final_learning_rate", 
                                             data.get("training_learning_rate"))
                
                # Error message
                result["error"] = data.get("error", "")
                
                results.append(result)
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)
            continue
    
    return pd.DataFrame(results)


def extract_params(profile_name):
    """Extract hyperparameters from profile name."""
    parts = profile_name.split('_')
    params = {}
    
    try:
        # Name format: {type}_grid_{idx}_{lr}_{bs}_{hs}_{ls}_{loss}
        if len(parts) >= 6:
            modality = "_".join(parts[:2])  # e.g., "audio-window-snn"
            grid_idx = parts[2]
            
            # Parse parameter string
            param_str = "_".join(parts[3:])
            
            # Extract using regex or simple parsing
            remaining = parts[3:]
            
            for p in remaining:
                if p.startswith("lr"):
                    params["lr"] = float(p[2:])
                elif p.startswith("bs"):
                    params["bs"] = int(p[2:])
                elif p.startswith("hs"):
                    params["hs"] = int(p[2:])
                elif p.startswith("ls"):
                    params["ls"] = int(p[2:])
                elif p in ["mse", "mae"]:
                    params["loss"] = p
            
            params["modality"] = modality
            params["grid_idx"] = grid_idx
    except:
        pass
    
    return params


def analyze(df):
    """Perform analysis on results dataframe."""
    
    print("\n" + "="*80)
    print("SNN AUTOENCODER GRID SEARCH ANALYSIS")
    print("="*80)
    
    # Filter for successful runs
    successful = df[df["exit_code"] == 0].copy()
    failed = df[df["exit_code"] != 0].copy()
    
    print(f"\nTotal profiles analyzed: {len(df)}")
    print(f"Successful runs: {len(successful)} ({100*len(successful)/len(df):.1f}%)")
    print(f"Failed runs: {len(failed)} ({100*len(failed)/len(df):.1f}%)")
    
    if len(failed) > 0:
        print(f"\nCommon errors in failed runs:")
        for error, count in failed["error"].value_counts().head(3).items():
            print(f"  - {error[:80]} ({count} runs)")
    
    if len(successful) == 0:
        print("\nNo successful runs yet!")
        return
    
    # Basic stats
    print(f"\n--- Loss Statistics (Successful Runs) ---")
    print(f"Initial loss:    mean={successful['initial_loss'].mean():.6f}, "
          f"median={successful['initial_loss'].median():.6f}")
    print(f"Final loss:      mean={successful['final_loss'].mean():.6f}, "
          f"median={successful['final_loss'].median():.6f}")
    print(f"Loss trend:      mean={successful['loss_trend'].mean():.4f}, "
          f"median={successful['loss_trend'].median():.4f}")
    print(f"Convergence:     {100 * (successful['loss_trend'] < 0).sum() / len(successful):.1f}% converging")
    
    # Top performers
    print(f"\n--- Top 10 Configurations ---")
    top = successful.nsmallest(10, "final_loss")
    for idx, row in top.iterrows():
        params = extract_params(row["profile"])
        params_str = f"lr={params.get('lr')}, bs={params.get('bs')}, hs={params.get('hs')}, " \
                     f"ls={params.get('ls')}, loss={params.get('loss')}"
        print(f"{row['final_loss']:.6f}  {row['profile']}")
        print(f"             {params_str}")
    
    # Modality comparison
    print(f"\n--- By Modality ---")
    for modality in successful["autoencoder_type"].unique():
        if pd.isna(modality):
            continue
        subset = successful[successful["autoencoder_type"] == modality]
        print(f"\n{modality}:")
        print(f"  Count: {len(subset)}")
        print(f"  Final loss (mean): {subset['final_loss'].mean():.6f}")
        print(f"  Final loss (min): {subset['final_loss'].min():.6f}")
    
    # Loss function comparison
    print(f"\n--- By Loss Function ---")
    for dtype in successful["dataset_type"].unique():
        if pd.isna(dtype):
            continue
        for profile_group in successful["profile"].str[:20].unique():
            subset = successful[successful["profile"].str.startswith(profile_group)]
            
            mse_rows = subset[subset["profile"].str.contains("_mse")]
            mae_rows = subset[subset["profile"].str.contains("_mae")]
            
            if len(mse_rows) > 0 and len(mae_rows) > 0:
                print(f"\n{profile_group}:")
                print(f"  MSE: mean={mse_rows['final_loss'].mean():.6f}, "
                      f"min={mse_rows['final_loss'].min():.6f}, count={len(mse_rows)}")
                print(f"  MAE: mean={mae_rows['final_loss'].mean():.6f}, "
                      f"min={mae_rows['final_loss'].min():.6f}, count={len(mae_rows)}")
    
    # Learning rate sensitivity
    print(f"\n--- Learning Rate Sensitivity ---")
    if "final_loss" in successful.columns:
        # Extract LR from profile names
        lrs = []
        losses = []
        for _, row in successful.iterrows():
            params = extract_params(row["profile"])
            if "lr" in params:
                lrs.append(params["lr"])
                losses.append(row["final_loss"])
        
        if lrs:
            lr_df = pd.DataFrame({"lr": lrs, "loss": losses})
            print(lr_df.groupby("lr")["loss"].agg(["count", "mean", "min", "std"]).to_string())
    
    # Output CSV
    output_file = "snn_grid_search_results.csv"
    successful_with_params = successful.copy()
    for idx, row in successful_with_params.iterrows():
        params = extract_params(row["profile"])
        for key, val in params.items():
            successful_with_params.at[idx, key] = val
    
    successful_with_params.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    print("Loading results...")
    df = load_results()
    
    if len(df) == 0:
        print("No results found yet!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} profile results")
    analyze(df)
