# experiment03 - Neural Network training runner

Overview
--------
`experiment03` trains ANN or SNN neural networks against the 10.1117 imagined-speech EEG+Audio dataset.
It is profile-driven and supports two layers of profile configuration:

- A base profile JSON loaded from `src/experiments/03/profiles/default.json`.
- A selected profile JSON loaded from `src/experiments/03/profiles/*.json`.

Configuration precedence
------------------------
Runtime configuration is resolved in this order:

1. `default` profile in `src/experiments/03/profiles/default.json`
2. `--profile <name>` or `--profile <path/to/file.json>`

All runtime behavior must be defined by profile fields. CLI accepts only `--profile` (plus help flags).

Use `--help` to inspect the full option list without starting the experiment or touching the logger pipeline.

Determinism policy
------------------
- `sampler_shuffle_seed` must be defined by profile, otherwise startup fails.
- If `kfold_enabled=true` and `kfold_shuffle=true`, `kfold_seed` must be defined, otherwise startup fails.

How to run
----------
Run with a profile name (from `src/experiments/03/profiles`) or an explicit JSON path:

```bash
./src/experiments/03/experiment03 --profile default
```

```bash
./src/experiments/03/experiment03 --profile src/experiments/03/profiles/sample-training-flow.json
```

How to think about batches and epochs
-------------------------------------
If you want genuine training runs instead of capped smoke tests, the most important profile fields are:

- `training_epochs`: increase this above `1` for repeated training passes.
- `training_batch_size`: controls optimization granularity and throughput.
- `training_max_batches_per_epoch = 0`: removes the artificial epoch cap and lets each epoch consume the full dataset.

Typical patterns:

Smoke test profile example:
```bash
./src/experiments/03/experiment03 --profile lightweight
```

Full training epoch over the dataset:
```bash
./src/experiments/03/experiment03 --profile sample-training-flow
```

Longer SNN training run:
```bash
./src/experiments/03/experiment03 --profile fused-window-snn-default
```

Profiles
--------
Profiles live in `src/experiments/03/profiles/` and act as reusable starting points.

The profile loader accepts ordinary JSON plus `_comment_*` metadata fields. Any other unknown key is rejected at load time. This repo includes a loadable example of inline documentation in `sample-training-flow.json`.

Useful built-in examples:
- `default.json`: generic baseline defaults.
- `lightweight.json`: smaller/faster smoke-test baseline.
- `fused-window-snn-default.json`: fused SNN baseline.
- `protocol-ann-default.json`: protocol ANN baseline.
- `protocol-snn-default.json`: protocol SNN baseline.
- `sample-training-flow.json`: commented, loadable example that shows how to move from smoke tests to full training.

Profiles may set:
- dataset type
- neural network type
- explicit encoder/decoder layer specs
- explicit branch/fusion layer specs for fused neural networks
- batch size
- epoch count
- optimizer type
- loss type
- learning rate
- window configs
- neural network widths/depth
- prefetch settings

Declarative profile fields
-------------------------
Profiles are now the authoritative place to describe neural-network structure and training policy.

Core architecture field:
- `neural_network_layer`: ordered declarative stages using `section:layer_spec` entries.

Supported sections:
- `encoder`
- `decoder`
- `branch_encoder`
- `branch_decoder`
- `fusion_encoder`
- `fusion_decoder`

Supported layer spec tokens:
- modules: `linear`, `residual`, `residual_block`, activation-only entries
- width tokens: integer width, `latent`, `output`, `branch_hidden`, `fusion_hidden`
- ANN activations: `relu`, `leaky_relu`, `identity`
- SNN activations: `leaky`, `leaky_integrator`, `identity`

Supported grammar:
- `linear:<width>:<activation>`
- Broader module grammar:
  - `linear:<width>`
  - `<activation>` (activation-only stage)
  - `residual` or `residual:<N>` (repeat residual block `N` times)

Examples:
- `"linear:64:relu"`
- `"linear:64"`, `"relu"`, `"residual:2"`, `"linear:latent"`, `"identity"`

Core training fields:
- `training_optimizer_type`: currently `adam` or `sgd`
- `training_loss_type`: currently `mse` or `mae`
- `training_learning_rate`
- `training_lr_plateau_enabled`
- `training_lr_plateau_factor`
- `training_lr_plateau_patience`
- `training_lr_plateau_min_delta`
- `validation_modality_diagnostics_enabled`

Minimal ANN example:
```json
{
  "neural_network_type": "fused-window-ann",
  "training_optimizer_type": "adam",
  "training_loss_type": "mse",
  "neural_network_layer": [
    "encoder:linear:64:relu",
    "encoder:linear:64:relu",
    "encoder:linear:latent:relu",
    "decoder:linear:64:relu",
    "decoder:linear:64:relu",
    "decoder:linear:output:identity"
  ]
}
```

Profiles are authoritative; there are no per-parameter CLI overrides.

Sample commented profile flow
-----------------------------
Use `sample-training-flow.json` when you want a self-documented baseline.

Recommended progression:

1. Start with the profile as-is for a short smoke run.
2. Increase `training_epochs` and set `training_max_batches_per_epoch` to `0` once the pipeline is stable.
3. Increase `training_batch_size` only after checking RAM and throughput.
4. Change dataset/neural-network families after the training loop is already behaving correctly.

Example:
```bash
./src/experiments/03/experiment03 --profile sample-training-flow
```

Default profile baseline
------------------------
When `--profile` is omitted, the launcher resolves to the `default` profile.
Baseline behavior should be edited in `src/experiments/03/profiles/default.json`.

- For reproducibility, create a dedicated profile file for each sweep or study.

Data layout expected by the runner
----------------------------------
The dataset root is expected to contain subject folders such as:

```text
dataset_root/
  S01/
    S01_EEG.mat
    S01_Audio.mat
  S02/
    S02_EEG.mat
    S02_Audio.mat
```

The protocol dataset path uses synchronized audio and EEG rows. Windowed dataset types derive sliding windows from those source signals using the configured EEG/audio window specs.

Operational notes
-----------------
- The prefetcher uses a single producer thread to avoid unsafe concurrent MAT I/O.
- Progress reporting uses the effective batch limit, so capped runs still report correct percentages.
- Run summaries are written as JSON and include profile, dataset type, neural-network type, exit code, processed samples, seen batches, and epoch mean losses.

Troubleshooting
---------------
- If training ends too quickly, check whether `training_max_batches_per_epoch` is capped to a small number.
- If you expected multiple epochs, verify `training_epochs` is greater than `1`.
- If no batches are produced, verify `dataset_root_path`, `dataset_subject_filter_regex`, and dataset integrity.
- If I/O becomes unstable during debugging, temporarily reduce `program_prefetch_lookahead` to `1`.

Code pointers
-------------
- Launcher defaults: `src/experiments/03/experiment03.cpp`
- CLI contract and `Config` fields: `src/experiments/03/lib/include/cli.hpp`
- CLI parsing and overrides: `src/experiments/03/lib/src/cli.cpp`
- Training loop: `src/experiments/03/lib/src/experiment03.cpp`
- Profile loading: `src/experiments/03/lib/src/ProfileLoader.cpp`

References
----------
This experiment builds upon established autoencoder and multimodal learning literature:

### Foundational Autoencoder Theory
- **[Vincent et al., 2010]** Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol.
  "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion."
  *Journal of Machine Learning Research*, vol. 11, pp. 3371–3408.
  https://www.jmlr.org/papers/v11/vincent10a.html
  - Key insight: Denoising criterion (input corrupted, output clean) guides learning of robust, transferable representations.

### Variational Autoencoders (VAE)
- **[Kingma & Welling, 2013]** Diederik P. Kingma and Max Welling.
  "Auto-Encoding Variational Bayes."
  *arXiv*:1312.6114, 2013.
  https://arxiv.org/abs/1312.6114
  - Key insight: Reparameterization trick enables efficient variational inference; KL divergence regularizes latent space.

- **[Burgess et al., 2018]** Christopher P. Burgess, Irina Higgins, Anurag Saxe, and Alexander A. Lerch.
  "Understanding Disentangling in β-VAE."
  *arXiv*:1804.03599, 2018.
  https://arxiv.org/abs/1804.03599
  - Key insight: β-annealing enables progressive information capacity growth, preventing KL collapse while learning disentangled representations.

### Multimodal Learning
- **[Baltrušaitis et al., 2017]** Tadas Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency.
  "Multimodal Machine Learning: A Survey and Taxonomy."
  *arXiv*:1705.09406, 2017.
  https://arxiv.org/abs/1705.09406
  - Key findings: Early fusion (feature-level) and late fusion (decision-level) strategies; co-learning across modalities improves generalization.

### Optimization Best Practices
- **[PyTorch Team, 2024]** "Optimization Loop."
  PyTorch Official Tutorials.
  https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
  - Key practices: Adam optimizer with learning-rate scheduling (ReduceLROnPlateau), early stopping, per-epoch validation loops.

### Recommended Configuration Strategy
When tuning autoencoders on multimodal EEG+Audio data:

1. **Loss function:** Plain MSE is baseline; combine 0.7×MSE + 0.3×L1 for robustness and reduced overfitting.
2. **Denoising:** Add input corruption (EEG: Gaussian noise σ∈[0.01,0.03], Audio: temporal masking) to harden reconstruction.
3. **Latent dimensionality:** Sweep {16, 24, 32} to find compression sweet spot; larger bottlenecks improve reconstruction, risk overfitting.
4. **Regularization:** For VAE objectives, use β-annealing; start β=0.1, progressively increase to 1.0 over first 5–10 epochs.
5. **Multimodal strategy:** Prefer dual-branch (per-modality encoder + late fusion) over naive concatenation; enables independent scaling.
6. **Optimization:** Adam (lr=1e-4), ReduceLROnPlateau (factor=0.5, patience=3–5), early stopping (patience 8–10 on validation loss).
7. **Tracking:** Report train/validation per epoch; decompose losses by modality to diagnose imbalance.
