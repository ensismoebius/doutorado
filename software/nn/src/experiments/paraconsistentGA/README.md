# paraconsistentGA — NSGA-II search over autoencoder architectures

Searches the same architecture axes as the thesis Phase 00 autoencoders (layers, widths,
encodings, losses), but automatically with NSGA-II. Each candidate is ranked by the
paraconsistent `d_penalized` of its latent features under a latency constraint. It reuses the
thesis code (`thesis_lib`: data loading, feature extraction, scoring) and reimplements none of
it.

- Design and rationale: [`.wiki/Experiments/ParaconsistentGA-Design.md`](../../../.wiki/Experiments/ParaconsistentGA-Design.md)
- How to run it and read its outputs: [`.wiki/Experiments/ParaconsistentGA.md`](../../../.wiki/Experiments/ParaconsistentGA.md)
- What was reused, confirmed against the tree: [`PHASE0.md`](PHASE0.md)

| Path | Role |
|---|---|
| `paraconsistentGA.cpp` | entry point: `paraconsistentGA --config <profile.json>` |
| `lib/*/GaConfig.*` | the profile schema (strict, `Required<T>` like the thesis) |
| `lib/*/GaGenome.*`, `DiploidGenome.hpp`, `Gamete.hpp`, `GenomeBounds.hpp` | the genome, diploid reproduction, and its bounds |
| `lib/*/GaFitness.*` | genome → autoencoder config → `thesis::extract_features` → `d_penalized` + latency |
| `lib/*/GaNsga2.*` | the NSGA-II loop (non-dominated sorting from `include/ga/Nsga2Core.hpp`) |
| `lib/*/GaCheckpoint.*`, `GaModelSnapshot.*` | resumable checkpoints and saved models |
| `lib/*/GaOutput.*` | Pareto fronts and per-generation output under `results/paraconsistentGA/` |
| `profiles/` | the 12 population profiles (`pga_{ann,snn}_{eeg,voice,fused_early,fused_late}` + SNN loss variants); `profiles/smoke/pga_debug.json` |
| `tests/` | `paraconsistent_ga_gtest` |

```bash
# from software/nn
cmake --build out/build/max-performance --target paraconsistentGA paraconsistent_ga_gtest -j$(nproc)
./out/build/max-performance/src/experiments/paraconsistentGA/paraconsistentGA \
    --config src/experiments/paraconsistentGA/profiles/smoke/pga_debug.json
./scripts/pipeline/paraconsistentGA/01_paraconsistentGA_run_all_profiles.sh   # all 12 (heavy)
```
