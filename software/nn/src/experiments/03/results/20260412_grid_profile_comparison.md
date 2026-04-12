# Experiment03 ANN Grid Comparison (2026-04-12)

## Scope
This sweep was built from:
- Findings from `/tmp/FINAL_TUNING_REPORT.md` (latent impact small, plateau early, prioritize loss/scheduler/diagnostics).
- Existing in-repo profile schema under `src/experiments/03/profiles`.
- External guidance from official and canonical sources:
  - PyTorch optimization loop: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
  - PyTorch ReduceLROnPlateau: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
  - Denoising autoencoders (Vincent et al. 2010): https://www.jmlr.org/papers/v11/vincent10a.html
  - Multimodal fusion taxonomy (Baltrusaitis et al. 2017): https://arxiv.org/abs/1705.09406
  - beta-VAE capacity schedule (Burgess et al. 2018): https://arxiv.org/abs/1804.03599

## Runtime Budget (constant for all runs)
- Controlled via profile fields: `kfold_enabled = false`, `training_epochs = 2`, `training_max_batches_per_epoch = 50`
- binary: `out/build/max-performance/src/experiments/03/experiment03`

## Profiles Created
- `src/experiments/03/profiles/fused-window-ann-grid-a-baseline.json`
- `src/experiments/03/profiles/fused-window-ann-grid-b-deeper.json`
- `src/experiments/03/profiles/fused-window-ann-grid-c-verydeep.json`
- `src/experiments/03/profiles/fused-window-ann-grid-d-mae.json`
- `src/experiments/03/profiles/fused-window-ann-grid-e-higherlr.json`
- `src/experiments/03/profiles/fused-window-ann-grid-f-batch32.json`

## Raw Comparison (mixed metrics)
Note: MAE and MSE are not directly comparable by absolute value.

| Rank | Profile | Status | Loss | LR | Final LR | Last Train | Last Val | Last Val EEG | Last Val Audio | Seen Batches | Processed Samples | Result JSON |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | fused-window-ann-grid-d-mae | ok | mae | 0.0001 | 0.0001 | 1.42509 | 1.42477 | 90.2946 | 19.8867 | 100 | 1052 | src/experiments/03/results/20260412_091529_fused-window-ann-grid-d-mae.json |
| 2 | fused-window-ann-grid-f-batch32 | ok | mse | 0.0001 | 0.0001 | 26.6637 | 26.6554 | 28.5735 | 26.3883 | 100 | 3100 | src/experiments/03/results/20260412_091552_fused-window-ann-grid-f-batch32.json |
| 3 | fused-window-ann-grid-e-higherlr | ok | mse | 0.0003 | 0.0003 | 28.0462 | 27.0018 | 87.045 | 18.6367 | 100 | 1052 | src/experiments/03/results/20260412_091539_fused-window-ann-grid-e-higherlr.json |
| 4 | fused-window-ann-grid-a-baseline | ok | mse | 0.0001 | 0.0001 | 28.4773 | 28.4239 | 90.0537 | 19.8377 | 100 | 1052 | src/experiments/03/results/20260412_091454_fused-window-ann-grid-a-baseline.json |
| 5 | fused-window-ann-grid-b-deeper | ok | mse | 0.0001 | 0.0001 | 28.4866 | 28.4606 | 90.1346 | 19.8683 | 100 | 1052 | src/experiments/03/results/20260412_091503_fused-window-ann-grid-b-deeper.json |
| 6 | fused-window-ann-grid-c-verydeep | ok | mse | 0.0001 | 0.0001 | 28.4915 | 28.4651 | 90.2384 | 19.8589 | 100 | 1052 | src/experiments/03/results/20260412_091518_fused-window-ann-grid-c-verydeep.json |

## Fair Ranking (MSE-only)
| MSE Rank | Profile | Last Val (MSE) | Last Train (MSE) | Val EEG (MSE) | Val Audio (MSE) |
|---:|---|---:|---:|---:|---:|
| 1 | fused-window-ann-grid-f-batch32 | 26.6554 | 26.6637 | 28.5735 | 26.3883 |
| 2 | fused-window-ann-grid-e-higherlr | 27.0018 | 28.0462 | 87.0450 | 18.6367 |
| 3 | fused-window-ann-grid-a-baseline | 28.4239 | 28.4773 | 90.0537 | 19.8377 |
| 4 | fused-window-ann-grid-b-deeper | 28.4606 | 28.4866 | 90.1346 | 19.8683 |
| 5 | fused-window-ann-grid-c-verydeep | 28.4651 | 28.4915 | 90.2384 | 19.8589 |

## Main Findings
1. Deeper was not better in this budget: `d3` and `d4` trailed baseline at same LR and batch.
2. Batch size 32 gave the best MSE in this short regime.
3. Higher LR (3e-4) improved overall MSE versus baseline, but showed a large EEG/Audio imbalance.
4. MAE run achieved low MAE (as expected), but modality diagnostics are still MSE-like, so direct ranking against MSE runs is invalid.

## Practical Tips Applied to Next Round
1. Keep scheduler stepping on validation metric after validation (PyTorch RLROP guidance).
2. Compare only like-for-like metrics (MSE with MSE; MAE with MAE).
3. Since latent size already showed weak sensitivity in prior report, prioritize modality balancing and LR policy before adding depth.
4. Run a longer Stage-D style follow-up for top MSE profiles to confirm stability:
   - `fused-window-ann-grid-f-batch32`
   - `fused-window-ann-grid-e-higherlr`
