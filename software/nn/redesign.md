# Experiment03 Autoencoder Redesign Plan

## Goal

Replace the current scaffold-like MLP autoencoders in `experiment03` with modality-aware architectures for protocol, EEG-window, audio-window, and fused EEG+audio data. The redesign should favor realistic signal-processing inductive bias over cosmetic hyperparameter changes.

## Principles

- EEG and audio are temporal signals and should not be treated as i.i.d. flat vectors by default.
- Multimodal fused inputs should use modality-specific branches rather than a single monolithic encoder.
- SNN decoders should reconstruct from membrane-space outputs, not from spike-only outputs.
- Public training flow in `experiment03` should remain stable while architecture selection becomes more expressive.

## Planned Phases

### Phase 1: Architecture Surface

1. Expand `AutoencoderConfig` with architecture-family, branch-width, fusion-width, residual-depth, and modality split hints.
2. Extend CLI/config plumbing so experiment03 can select redesigned architectures from the command line.
3. Centralize reusable builder logic so the eight autoencoder variants do not duplicate nearly identical stacks.

### Phase 2: ANN Redesign

1. EEG ANN: tapered residual dense autoencoder as an immediate upgrade path, then temporal convolutional encoder-decoder when framework primitives are ready.
2. Audio ANN: deeper tapered residual dense autoencoder now, temporal convolutional autoencoder in the next primitive-expansion phase.
3. Fused ANN: dual-branch encoder and decoder with a shared fusion bottleneck.
4. Protocol ANN: use structured multimodal handling when the representation is manageable; otherwise preserve a strong dense fallback until Conv1d/downsampling primitives exist.

### Phase 3: SNN Redesign

1. Keep membrane reconstruction in the decoder.
2. Mirror unimodal ANN topology with spiking hidden dynamics where practical.
3. Mirror fused multimodal ANN topology with branch-specific spiking encoders and membrane-readout decoders.

### Phase 4: Training and Validation

1. Preserve MSE reconstruction as the baseline objective.
2. Add architecture smoke tests, encode/decode shape checks, branch split checks, and reset-state checks.
3. Rebuild and run targeted tests after each architecture slice lands.

## Immediate Implementation Slice

This repository pass implements the foundation and the first concrete redesign slice:

1. Add architecture/config surface.
2. Refactor unimodal builders to shared tapered/residual helpers.
3. Redesign fused ANN and fused SNN autoencoders into branch-aware multimodal models.
4. Preserve protocol models as dense fallbacks for now because the full-trial stacked representation still needs temporal downsampling primitives for a credible full redesign.

## Files Expected To Change

- `src/experiments/03/lib/include/AutoencoderConfig.hpp`
- `src/experiments/03/lib/include/cli.hpp`
- `src/experiments/03/lib/src/cli.cpp`
- `src/experiments/03/lib/src/experiment03.cpp`
- `src/experiments/03/lib/include/AutoencoderBuilders.hpp`
- `src/experiments/03/lib/src/*Autoencoder.cpp`
- `src/experiments/03/lib/include/*Autoencoder.hpp`

## Deferred Work

- True Conv1d or a clean temporal-convolution wrapper
- Normalization and dropout primitives in the core framework
- Protocol-specific temporal encoder over the stacked 4-second full-trial representation
- Latent regularization beyond plain reconstruction MSE