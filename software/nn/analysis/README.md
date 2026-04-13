# SNN Hyperparameter Grid Search - Analysis Results

This directory contains comprehensive analysis of the SNN autoencoder hyperparameter grid search.

## Files Overview

### Primary Results
- **`master_comparison.csv`** — Complete ranking of all 402 profiles by validation loss
  - Ranked from best to worst mean validation loss
  - Includes all hyperparameters (LR, BS, HS, LS), loss metrics, and status
  - **Use this for:** Finding top-performing configurations across all modalities

### Modality-Specific Rankings
- **`comparison_audio-window.csv`** — Best 288 audio-basis-only configurations
- **`comparison_fused-window.csv`** — Best 96 audio+EEG fusion configurations  
- **`comparison_eeg-window.csv`** — Best 18 EEG-only configurations
- **Use these for:** Understanding which configurations excel in each input modality

### Hyperparameter Sensitivity Analysis
- **`sensitivity_audio-window_mae.csv`** — Impact of each hyperparameter (audio + MAE loss)
- **`sensitivity_audio-window_mse.csv`** — Impact of each hyperparameter (audio + MSE loss)
- **`sensitivity_fused-window_mae.csv`** — Impact of each hyperparameter (fusion + MAE)
- **`sensitivity_fused-window_mse.csv`** — Impact of each hyperparameter (fusion + MSE)
- **`sensitivity_eeg-window_mae.csv`** — Impact of each hyperparameter (EEG + MAE)
- **`sensitivity_eeg-window_mse.csv`** — Impact of each hyperparameter (EEG + MSE)
- **Use these for:** Understanding which parameter values work best (e.g., optimal learning rate range)

### Summary Statistics
- **`summary_statistics.csv`** — Project-level key metrics
  - Total profiles and success rate
  - Best/worst/average/median validation losses
  - Per-modality performance summaries
  - **Use this for:** High-level project overview

## How to Use These Files

### Quick Start
```bash
# View top 20 best configurations
head -21 master_comparison.csv | column -t -s,

# Find best audio configuration
head -6 comparison_audio-window.csv | tail -5
```

### In Excel/LibreOffice
```bash
# Open any CSV in spreadsheet application
libreoffice master_comparison.csv
```

### Data Analysis
```bash
# Extract all MAE-based results
grep ",mae," master_comparison.csv

# Find configurations with specific batch size
grep ",8," master_comparison.csv

# Get top 10 by validation loss
head -11 master_comparison.csv | tail -10
```

### Python Analysis
```python
import pandas as pd

# Load master results
df = pd.read_csv('master_comparison.csv')

# Top 10 configurations
print(df.head(10))

# Best by modality
for modality in df['Modality'].unique():
    print(f"\n{modality}:")
    print(df[df['Modality'] == modality].head(3))

# Learning rate impact (for audio + MAE)
lr_effect = df[(df['Modality'] == 'audio-window') & (df['Loss Type'] == 'mae')].groupby('LR')['Mean Val Loss'].agg(['mean', 'min', 'count'])
print(lr_effect)
```

## Column Explanations

### master_comparison.csv
| Column | Meaning |
|--------|---------|
| Rank | Position in overall ranking (1 = best) |
| Profile | Configuration name (e.g., audio-window_grid_001) |
| Modality | Input type: audio-window, fused-window, eeg-window |
| Loss Type | mae or mse |
| LR | Learning rate (0.0001 to 0.001) |
| BS | Batch size (4 to 64) |
| HS | Hidden layer size (32 to 128) |
| LS | Latent size (16 to 64) |
| Final Train Loss | Training loss at last epoch |
| Final Val Loss | Validation loss at last training epoch |
| Mean Val Loss | Average across K-fold splits (primary metric) |
| Best Val Loss | Best validation loss during training |
| Epochs | Number of training epochs |
| Status | Success or failure code |

### sensitivity_*.csv
| Column | Meaning |
|--------|---------|
| Hyperparameter | name (Learning Rate, Batch Size, Hidden Size, Latent Size) |
| Value | Specific value tested |
| Avg Val Loss | Average validation loss for this value across other configs |
| Best Val Loss | Best validation loss achieved with this value |
| Count | How many configurations tested with this value |

## Key Findings

### From Completed Results
- **MAE vs MSE:** MAE loss typically 0.22 lower than MSE on same architecture
- **Success Rate:** 100% of profiles converged successfully
- **Best Architecture:** Audio-window with MAE loss
- **Typical Runtime:** 240-310 seconds per profile

### Hyperparameter Insights (From Sensitivity Tables)
Look for V-shaped patterns in sensitivity tables:
- Lowest Avg Val Loss indicates optimal range for each parameter
- Count column shows how many configurations were tested
- Best Val Loss shows potential (lower bound) for that parameter

## Regenerating These Files

If you need to regenerate these analysis files:

```bash
# Go to project root
cd /path/to/nn/

# Run the analysis script
python3 scripts/analyze_grid_results.py

# Or use the interactive wrapper
bash scripts/run_analysis.sh
```

The script looks for result files in `src/experiments/03/results/` and generates all CSV files in `analysis/`.

## Grid Parameters Tested

Total combinations: 402 configurations across:
- **Modalities:** 3 (audio-window, fused-window, eeg-window)
- **Learning Rates:** 3 (0.0001, 0.0005, 0.001)
- **Batch Sizes:** 5 (4, 8, 16, 32, 64)
- **Hidden Sizes:** 3 (32, 64, 128)
- **Latent Sizes:** 3 (16, 32, 64)
- **Loss Types:** 2 (mae, mse)

*Note: Not all theoretical combinations (2,700) were tested; 402 represents carefully selected experimental subset.*

## Archiving Results

To save these results for future reference:
```bash
# Create timestamped backup
tar -czf snn_grid_analysis_$(date +%Y%m%d_%H%M%S).tar.gz analysis/

# Or copy to version control
git add analysis/
git commit -m "SNN grid search analysis: 402 profiles"
```

---

Generated by `scripts/analyze_grid_results.py`  
For questions or modifications, refer to `ANALYSIS_QUICKREF.md` and `GRID_RERUN_SUMMARY.md`
