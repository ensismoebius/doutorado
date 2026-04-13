# Final Grid Findings (Deduplicated)

Source tables:
- analysis/master_comparison.csv (raw runs, includes duplicates)
- analysis/master_comparison_dedup_by_profile.csv (one best run per profile signature)

## Coverage
- Unique profiles: 402
- Successful: 306
- Failed: 96
- Success rate: 76.12%
- Modality distribution: audio-window=288, fused-window=96, eeg-window=18
- Loss distribution: mae=201, mse=201

## Overall Performance (successful only)
- Best mean validation loss: 0.704623
- Worst mean validation loss: 1.015340
- Average mean validation loss: 0.855541
- Median mean validation loss: 0.868075
- Stddev mean validation loss: 0.128858

## Best Overall Configuration
- Profile: audio-window-snn_grid_257_lr0_0010_bs32_hs32_ls64_mae
- Modality: audio-window
- Loss: mae
- Hyperparameters: LR=0.0010, BS=32, HS=32, LS=64
- Mean val loss: 0.704623

## Best in audio-window
- Profile: audio-window-snn_grid_257_lr0_0010_bs32_hs32_ls64_mae
- Mean val loss: 0.704623
- LR=0.0010, BS=32, HS=32, LS=64

## Best in eeg-window
- Profile: eeg-window-snn_grid_005_lr0_0001_bs32_mae
- Mean val loss: 0.753174
- LR=0.0001, BS=32, HS=32, LS=16

## Best in mae
- Profile: audio-window-snn_grid_257_lr0_0010_bs32_hs32_ls64_mae
- Mean val loss: 0.704623
- LR=0.0010, BS=32, HS=32, LS=64

## Best in mse
- Profile: audio-window-snn_grid_038_lr0_0001_bs32_hs32_ls32_mse
- Mean val loss: 0.954851
- LR=0.0001, BS=32, HS=32, LS=32

## Top 10 (deduplicated)

| Rank | Profile Signature | Modality | Loss | LR | BS | HS | LS | Mean Val Loss |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | audio-window-snn_grid_257_lr0_0010_bs32_hs32_ls64_mae | audio-window | mae | 0.0010 | 32 | 32 | 64 | 0.704623 |
| 2 | audio-window-snn_grid_263_lr0_0010_bs32_hs64_ls64_mae | audio-window | mae | 0.0010 | 32 | 64 | 64 | 0.704623 |
| 3 | audio-window-snn_grid_269_lr0_0010_bs32_hs128_ls64_mae | audio-window | mae | 0.0010 | 32 | 128 | 64 | 0.704623 |
| 4 | audio-window-snn_grid_275_lr0_0010_bs64_hs32_ls64_mae | audio-window | mae | 0.0010 | 64 | 32 | 64 | 0.704623 |
| 5 | audio-window-snn_grid_281_lr0_0010_bs64_hs64_ls64_mae | audio-window | mae | 0.0010 | 64 | 64 | 64 | 0.704623 |
| 6 | audio-window-snn_grid_287_lr0_0010_bs64_hs128_ls64_mae | audio-window | mae | 0.0010 | 64 | 128 | 64 | 0.704623 |
| 7 | audio-window-snn_grid_235_lr0_0010_bs16_hs32_ls16_mae | audio-window | mae | 0.0010 | 16 | 32 | 16 | 0.705212 |
| 8 | audio-window-snn_grid_241_lr0_0010_bs16_hs64_ls16_mae | audio-window | mae | 0.0010 | 16 | 64 | 16 | 0.705212 |
| 9 | audio-window-snn_grid_247_lr0_0010_bs16_hs128_ls16_mae | audio-window | mae | 0.0010 | 16 | 128 | 16 | 0.705212 |
| 10 | audio-window-snn_grid_163_lr0_0005_bs16_hs32_ls16_mae | audio-window | mae | 0.0005 | 16 | 32 | 16 | 0.706852 |