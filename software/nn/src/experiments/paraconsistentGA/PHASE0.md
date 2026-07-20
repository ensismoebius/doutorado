# paraconsistentGA — Phase 0 discovery report

Deliverable #1 of `ga.md` (§2). Confirms every reused component against the current
tree and records the scope decisions. **Reuse rule honored: no existing component is
reimplemented — the GA only assembles and orchestrates.**

## Reused components (confirmed present)

| Need | Symbol | Source | Confirmed |
|------|--------|--------|-----------|
| Paraconsistent α/β/G1/G2 | `calculate_alpha/beta/certainty_degree_g1/contradiction_degree_g2` | `include/paraconsistent/paraconsistent.hpp` | code |
| `d_penalized` ranking | `thesis::score_feature_set`, `ParaconsistentScore{alpha,beta,g1,g2,d_truth,d_penalized}`, `kContradictionPenalty = 2-√2` | `thesis/lib/include/ThesisParaconsistent.hpp` | code |
| AE train + latent extraction | `thesis::extract_features(view, feature_extraction, training, modality, fusion_mode, seed)` → `vector<FeatureSet>` | `thesis/lib/include/ThesisFeatureExtraction.hpp` | code |
| ANN-AE / SNN-AE networks | `ProtocolAutoencoder` / `ProtocolSpikingAutoencoder` (driven via `AutoencoderConfig`) | `autoencoderRunner/lib/include/autoencoder/` (linked through `thesis_lib`) | code |
| Config schema + JSON parse | `thesis::ThesisConfig::from_json` incl. `AutoencoderConfig{model,encoder_layer_spec,decoder_layer_spec,encoding,time_steps,voltage_threshold,firing_rate_*}` | `thesis/lib/include/ThesisConfig.hpp` | code |
| Dataset load | `thesis::load_dataset(dataset_cfg)` → `ThesisDatasetView{samples,...}` | `thesis/lib/include/ThesisDataset.hpp` | code |
| Output dir helper | `thesis::ensure_dir` | `thesis/lib/include/ThesisOutput.hpp` | code |
| SNN neurons / surrogate | `Lif`, `LifBPTT`, surrogate gradients | `include/layers/spiking/` | code |

Fitness is therefore a thin wrapper: genome → `AutoencoderConfig` → `extract_features`
(trains the AE, returns latent vectors) → `score_feature_set` (returns `d_penalized`).

## Known gaps and how they are handled

1. **NSGA-II — absent.** Implemented new in `GaNsga2.{hpp,cpp}` (fast non-dominated
   sort, crowding distance, constrained dominance). This is the only genuinely new
   algorithm.
2. **Reconstruction error not exposed.** `extract_features` returns only latent
   vectors, not decoder loss. Rather than modify `thesis_lib` (reuse rule), the
   §3.3 sanity filter is implemented as a **latent-collapse guard**: per-dimension
   latent std must exceed `tau_rec`. This directly detects the exact degeneracy the
   reconstruction filter defends against (α=β=1 constant latent), so it is a faithful
   proxy. A true reconstruction-MSE filter would require `extract_features` to also
   return the trainer's final loss — logged as future work, not done here.
3. **Van Rossum / Victor–Purpura losses — absent.** Out of scope for v1. The SNN-AE
   already trains under MSE on spike frames (`ae_cfg.loss_type="mse"`); the exotic
   spike-distance metrics are not required for the paraconsistent objective. Recorded
   as future work per `ga.md` §5.2. (`SpikeCountLoss`/`SpikeTimeLoss` do exist.)
4. **Latency on target hardware — no target confirmed** (`ga.md` §4). Secondary
   objective uses a **deterministic structural proxy** (encoder MAC count × time_steps),
   which is what §4 itself recommends for the cheap pre-training screen. The proxy→ms
   conversion (`ns_per_mac`) is uncalibrated and logged as such; real on-hardware
   calibration is future work.

## Scope decisions recorded (per ga.md)

- **Genome = phase00 AE axes only** (§5.1): `hidden` width, `latent` dim, and for the
  SNN population `encoding` ∈ {direct,latency,poisson}. `model` and `modality` are
  fixed per run (population-defining), exactly as phase00 splits its profiles.
- **`time_steps`/`voltage_threshold`: coupled to `encoding`** as in phase00
  (direct→1/1.0, latency|poisson→16/0.2). Not independent genes. Everything else
  phase00 holds constant (depth=2, leaky→identity, firing-rate reg 0.5/0.1/0.8) stays
  fixed.
- **Loss fixed per population** (§5.1): MSE. Not evolved.
- **Populations evolved separately** (§5.3), one `--config` run per (model, modality).
- **n_seeds** averaging with mean/std of `d_penalized` (§5.5).

## Note fed back to ga.md

`ga.md` §7 acceptance said `(α,β)=(1,1) → ≈ 2.4142`. That contradicts §3.1 ("the three
degenerate vertices … evaluate to exactly 2") and the implementation
(`kContradictionPenalty = 2-√2` is chosen precisely so those vertices score 2.0):
g1=0, g2=1, d_truth=√2≈1.4142, d_penalized=√2+(2-√2)·1 = **2.0000**. ga.md §7 corrected
to 2.0000; the `(0.92,0.075) → 0.1580` case is correct and kept.
