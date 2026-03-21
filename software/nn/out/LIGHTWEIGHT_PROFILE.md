# Lightweight Autoencoder Profile for Raspberry Pi (1GB RAM)

## Overview
To run the EEG+Audio autoencoders on memory-constrained devices like Raspberry Pi 4B with 1GB RAM, use the `--profile lightweight` flag.

## Profile Settings

### Lightweight Profile (RPi 1GB target)
```bash
--profile lightweight
```

Configuration applied:
- **Hidden size**: 16 (vs default 64) — reduces layer widths 4x
- **Latent size**: 8 (vs default 32) — reduces bottleneck 4x  
- **Depth**: 1 (vs default 2) — single hidden layer per side
- **Batch size**: 4 (vs default 32) — memory-friendly mini-batch
- **Max batches per epoch**: 20 (vs default 100) — limits iterations

**Estimated model size**: ~50KB (vs ~500KB for default)

## Quick Start

### ANN Lightweight (Raspberry Pi)
```bash
./experiment03 --profile lightweight \
  --dataset-type fused-window \
  --autoencoder fused-window-ann \
  --epochs 2 \
  --max-batches 10
```

### SNN Lightweight (Raspberry Pi)
```bash
./experiment03 --profile lightweight \
  --dataset-type fused-window \
  --autoencoder fused-window-snn \
  --epochs 2 \
  --max-batches 10
```

## Desktop/Server Profile (Default)
```bash
--profile default  # or omit (default is applied automatically)
```

Configuration:
- Hidden size: 64 | Latent size: 32 | Depth: 2 | Batch size: 32

## Memory Usage Estimates

| Parameter | ANN | SNN |
|-----------|-----|-----|
| Hidden → Latent layers | 16→8, 8→16 | +state vars (1x1 per neuron) |
| Learnable params (lightweight) | ~5K | ~10K |
| Activations (batch_size=4) | ~64KB | ~200KB |
| Gradients (same as activations) | ~64KB | ~200KB |
| **Total per epoch** | **~250KB** | **~500KB** |
| Peak forward+backward | **~500KB** | **~1MB** |

## CLI Override
Profile settings can be overridden by explicit flags; later flags override profile:
```bash
# Apply lightweight profile but use larger hidden size
./experiment03 --profile lightweight --ae-hidden-size 32
```

## Testing on Raspberry Pi
Before deploying to RPi, ensure:
1. Build native ARM binaries (cross-compile or build on RPi directly)
2. Start with `--max-batches 1` to verify compilation
3. Monitor `/proc/meminfo` during first training run  
4. Adjust batch_size further if needed (default lightweight is batch=4)

## Additional Optimization Tips
- Use `--no-shuffle` to reduce RNG overhead
- Set `--lookahead 1` (default) to avoid prefetch memory overhead
- On very constrained RPi (512MB), reduce further:
  ```bash
  ./experiment03 --profile lightweight \
    --ae-hidden-size 8 \
    --ae-latent-size 4 \
    --batch-size 2 \
    --max-batches 5
  ```

