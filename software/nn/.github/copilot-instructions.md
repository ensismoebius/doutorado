# Neural Network Framework Guidelines (updated)

This C++ neural network framework focuses on spiking neural networks (SNNs) and small autoencoder examples. The project uses Eigen for linear algebra, GoogleTest for unit tests, and CMake for builds. This document outlines up-to-date patterns, conventions and where to find common components in the repository.

## Project layout (high level)

```
nn/
├── src/
│   ├── dataLoaders/      # Dataset, DataLoader, MatFile, tests
│   ├── initializers/     # Weight initialization strategies
│   ├── layers/           # Neural network layers (Linear, Leaky, ...)
│   ├── optimizers/       # Optimization algorithms (Adam, SGD, ...)
│   ├── tensor/           # Tensor wrapper around Eigen
│   └── util/             # Utility functions and helpers
├── lib/                  # Third-party libs (cnpy, imgui, implot, ...)
├── .github/              # repo guidelines & Copilot instructions
├── build/                # CMake build artifacts (created at configure-time)
├── Flags.cmake           # Compiler/flag policy used by top-level CMake
├── Main.cmake            # High-level project wiring (includes modular .cmake files)
└── README_ACTION_HISTORY.md (generated)  # optional auto-generated action history
```

## Key components and current conventions

### Tensor system

- `Tensor` is implemented in `src/tensor/Tensor.hpp` and wraps Eigen matrices.
- A Tensor contains `data` (Eigen::MatrixXf) and `grad` where applicable.

### Layers and Modules

- Layers inherit from a `Module` base and implement `forward()` and `backward()`.
- Common layers: `Linear`, `Leaky` (LIF-like), `ReLU`, `LeakyReLU` etc.
- Composition: use `Sequential` (`src/layers/Sequential.hpp`) to chain layers.

### Data loading

- Dataset API: abstract `Dataset` in `src/dataLoaders/Dataset.h` with `get_item`, `collate`, `size()`.
- `TensorDataset` in `src/dataLoaders/TensorDataset.h` is a concrete dataset that wraps `Tensor` inputs/targets and implements `get_item` and `collate`.
- `DataLoader` lives in `src/dataLoaders/DataLoader.h/.cpp` and provides iteration, optional shuffling, deterministic seeding, and batching. Use `for (const auto &batch : loader)` to iterate.
- `MatFile` utilities: `src/dataLoaders/MatFile.cpp/h` read/write MATLAB `.mat` v5 files and are used by the demo.

Notes on DataLoader/Dataset
- The code was refactored to be PyTorch-like: `Dataset` + `DataLoader` + `collate` semantics.
- `TensorDataset` uses `Tensor::slice()` to build per-sample batches. `Dataset::collate` provides a default implementation that allocates and fills matrices.

### Demo: `dataLoader_demo`

- Demo source: `src/dataLoaderTest.cpp`.
- Behavior: reads the first numeric variable from a `.mat` file (via `MatFile`), trims to safe dimensions, wraps data into `TensorDataset`, constructs a tiny autoencoder (Linear -> Leaky -> Linear), trains with `MSELoss` + `Adam`, and writes `reconstructed.mat`.
- Defensive features: sample-size caps (`max_features`, `max_elements`) and try/catch around large Eigen allocations to avoid OOMs when users point to large MAT files.

### Initializers & optimizers

- Initializers: `xavierInitializer`, `kaimingSNNInitializer` in `src/initializers/`.
- Optimizers: `Adam`, `SGD`, `SGDMinimal` implementing `attach(params)`, `step()`, `zero_grad()`.

## Build & CMake layout

- Top-level CMakeLists includes `Flags.cmake` to set compiler flags and then includes `Main.cmake`.
- I reorganized `Main.cmake` to include modular files so each logical group lives in its own `.cmake` file at the repository root, e.g.:
   - `AutoEncoderTargets.cmake` — auto-encoder related targets
   - `PlotTarget.cmake` — plotSpikingNetwork and ImGui/ImPlot wiring
   - `LoadingDataTarget.cmake` — `loadingData` target
- `Flags.cmake` contains compiler flags and policy settings (C++20, diagnostics, debug/release flags) and is included by the top-level `CMakeLists.txt`.

Quick configure & build

```bash
cmake -S . -B build
cmake --build build -- -j <cpus>
```

## Tests

- Unit tests: GoogleTest (gtest). Tests are grouped and discovered via `gtest_discover_tests`.
- Consolidated test target: `dataLoaders_gtest` (covers MatFile tests and data loader tests). Other component gtest targets (layers_gtest, optimizers_gtest, tensor_gtest, etc.) are found under `src/*/CMakeLists.txt`.
- Run tests (from repo root):

```bash
cmake --build build --target dataLoaders_gtest
ctest --test-dir build --output-on-failure -j 4
```

## Tests added recently (high level)
- `src/dataLoaders/dataLoader_gtest.cpp` — deterministic shuffle, small/empty dataset behavior.
- `src/dataLoaders/dataLoader_more_gtest.cpp` — collate correctness, mismatched input/target columns, deterministic seed checks, and a lightweight concurrency smoke test.
- CMake updated to create a single `dataLoaders_gtest` target that links the loader implementation so tests link cleanly.

## Conventions & coding guidelines (current)

1. Code style

    - Use `auto` with trailing-return type when helpful for clarity.
    - Member variables use snake_case; functions use camelCase.
    - Prefer descriptive local variable names in demos and tests.

2. Memory & ownership

    - Use `std::shared_ptr` for layer ownership and dataset sharing.
    - Tensor slices/copied views are used for safety; avoid exposing raw pointers for internal tensors.

3. Error handling

    - Use `assert` for internal invariants.
    - For I/O and external input (MAT files) use explicit checks and guarded allocations.

4. Performance

    - Use Eigen efficiently (avoid unnecessary copies where possible).
    - Configure `Eigen` parallel behavior via `EigenParallel.cmake` and `configure_eigen_parallel_target()`.
    - Batch-level parallelism uses OpenMP; targets are linked with `${OpenMP_CXX_LIBRARIES}`.

## Common tasks (examples)

1) Add a new layer

```cpp
struct NewLayer : public Module {
   auto forward(const Tensor &input) -> Tensor override;
   auto backward(const Tensor &grad_output) -> Tensor override;
};
```

2) Small training loop sketch

```cpp
optimizer.zero_grad(params);
Tensor out = model.forward(input);
Tensor loss = loss_fn.forward(out);
Tensor grad = loss_fn.backward(out);
model.backward(grad);
optimizer.step(params);
```

## Helpful pointers

- `src/dataLoaders/TensorDataset.h` — example of a minimal concrete Dataset.
- `src/dataLoaders/DataLoader.{h,cpp}` — iteration/shuffle/batching behavior (iterator-based API).
- `src/dataLoaderTest.cpp` — a runnable demo that trains an autoencoder on a MATLAB variable and writes `reconstructed.mat`.
- `Flags.cmake` — common compiler flags and standard settings used across the project.

## Next steps and optional improvements

- Add a committed small `.mat` test file for CI reproducibility (currently demo creates `/tmp/dataLoader_demo_default.mat` at runtime).
- Add sanitizer-enabled CMake configurations (`ASAN`, `TSAN`) and a CI job to run concurrency-related tests under ThreadSanitizer.
- Split `cmake/` helpers into a small helper directory (`cmake/utils.cmake`) and document available helper functions.

If you want, I can:
- generate a `README_ACTION_HISTORY.md` from the git log (I can create that now),
- add sanitizer targets, or
- commit a small deterministic `.mat` file into `tests/data/` and add a CTest that runs the demo on it.

---

Small note: this file is intended as a living, developer-facing guide. When you change build wiring (CMake modularization, new targets, or DataLoader API) please update this document so it reflects the current project conventions.

## Copilot & linter guidance (how to generate edits safely)

This section tells autocomplete/code-generation tools (and contributors using them) how to produce edits that play nicely with the project's linters, CMake, and build flags. Follow these concrete rules when generating or editing code in this repository.

- Language & standard
    - Target C++20. The project sets `CMAKE_CXX_STANDARD 20` in `Flags.cmake`.

- Compiler flags and checks
    - Respect the compiler flags in `Flags.cmake` (e.g., `-Wall -Wpedantic -Wshadow`). Avoid constructs that intentionally silence those warnings.
    - Prefer explicit casts where needed to avoid narrowing or signed/unsigned comparison warnings.

- Formatting & static checks
    - Run `clang-format` on edits when adding or changing multiple lines of C++ code. Keep the existing style (four-space indentation, no mixed tabs).
    - Run `clang-tidy` fixes where appropriate (do not enable new `clang-tidy` checks that will fail CI unless the code is updated to satisfy them).

- Small patterns that cause linter warnings (avoid these)
    - Do not declare multiple variables in one statement. Instead prefer one declaration per line:

        // avoid
        std::vector<int> a, b;

        // prefer
        std::vector<int> a;
        std::vector<int> b;

    - Use uppercase unsigned suffixes for integer literals when needed (e.g., `42U` not `42u`) to match the style used in this project.
    - Always use braces for single-line control statements (loops/ifs) to avoid clang-tidy/clang-format complaints and accidental mistakes:

        // prefer
        for (auto &x : xs) {
            do_something(x);
        }

    - Avoid calling `emplace_back` in a loop without reserving capacity first. Either call `reserve()` or populate a pre-sized container.

- CMake editing rules
    - When editing CMake files: don't leave stray text fragments or unmatched parentheses — CMake errors can be introduced by accidental copy/paste.
    - Prefer extracting logical blocks into separate `.cmake` files and `include()` them from `Main.cmake` (the repository already uses this pattern).
    - When adding new top-level targets, ensure any source files needed by tests are linked into the test target so link-time errors do not occur.

- Tests and CI
    - Add or update tests using GoogleTest where relevant. Use `gtest_discover_tests()` to register test binaries with CTest.
    - After edits that touch CMake, run a local configure step (`cmake -S . -B build`) and run `ctest --test-dir build --output-on-failure -j 4` to validate behavior.

- Commit and PR hygiene
    - Keep generated changes minimal and targeted. When a copilot suggestion spans multiple files, re-run formatting and the test suite before committing.



# Speaker Identification Pipeline: EEG + Audio with Spiking Neural Networks

## TODO

- Implementar regularização L1 e L2 no auto-encoder? (é citada na tese, acho que vou tirar)
  - Acho que não precisa pois os neurônios de pulso aparentemente não requerem (eu acho).
- Implementar denoising auto-encoder? (é citada na tese, acho que vou tirar tb)
  - Se tudo der certo será um auto-encoder de pulso.
- Definir o tamanho da janela de tempo (time window) a ser utilizada na extração de features
  - Verificar se a biblioteca de wavelets já não implementa as janelas de alguma forma.
- Definir o número de features a serem extraídas pelo auto-encoder
- Definir a arquitetura do auto-encoder (número de camadas, número de neurônios por camada, tipo de camadas, etc.)

- Implementar a etapa de pré-processamento dos dados:
  - Seguindo o padrão do mestrado nomealizar os sinais entre -1 e 1.
  - Redimensionar o tamanho do sinal janelado para uma potência de 2 para que a transformada wavelet seja feita sem maiores problemas.
- Definir os parâmetros de treinamento do auto-encoder (número de épocas, taxa de aprendizado, tamanho do batch, etc.)

## Executive Summary

For the task of speaker identification using synchronized EEG and audio data, the recommended starting point is a **1.5-second window with a 50% overlap**. This duration is long enough to capture prosodic and intonational cues fundamental to speaker identity, which are often more discriminative than short-term phonetic features (Snyder et al., 2018). The 50% overlap ensures a good trade-off between temporal resolution and computational efficiency, generating a sufficient number of samples for training deep learning models without excessive redundancy.

---

## 1. Executive Recommendation

- **Optimal Window Length:** 1.5 seconds.
- **Optimal Overlap:** 50% (or 0.75 seconds).
- **Rationale:** Speaker identification relies heavily on suprasegmental features like pitch, rhythm, and intonation, which unfold over longer time scales than phonemes. While phonetic information can be captured in windows as short as 100-200 ms, these are often insufficient for robust, text-independent speaker recognition. A 1.5s window provides a strong balance, capturing these critical prosodic contours. The 50% overlap ensures that features at the edges of windows are not lost and doubles the number of training examples from the same data, which is beneficial for data-hungry deep learning models.

## 2. Candidate Grid to Evaluate

The following grid should be evaluated to empirically determine the best configuration.

- **Window Lengths (seconds):** `[0.2, 0.5, 1.0, 1.5, 2.0, 3.0]`
- **Overlaps (percentage):** `[0.25, 0.5, 0.75]`
- **Justification:**
  - **Short Windows (0.2s, 0.5s):** These will primarily capture phonetic and formant information. They are computationally cheaper but may be less robust for text-independent tasks as they don't capture the broader speaking style. They are worth testing to establish a baseline and understand the contribution of fine-grained acoustic features.
    - **Medium Windows (1.0s, 1.5s, 2.0s):** This range is the most promising. It's long enough to capture prosodic information, intonation patterns, and co-articulation effects that are highly speaker-specific (García-Perera et al., 2021).
    - **Long Windows (3.0s):** These windows capture significant contextual and rhythmic information. However, they may smear short-term temporal details, increase computational load, and reduce the number of available training samples. They are included to test the upper bound of performance.
    - **Overlaps:** Testing 25%, 50%, and 75% overlap allows for a systematic evaluation of the trade-off between data augmentation, computational cost, and model performance. Higher overlap can lead to better performance at the cost of longer training times.

## 3. Detailed Preprocessing Pipeline

### Audio Preprocessing

1. **Downsampling:** Resample the audio from 44.1 kHz to 16 kHz. This is standard for speech processing, as most speaker-relevant information is below 8 kHz.
2. **Band-Pass Filtering:** Apply a Butterworth band-pass filter between 80 Hz and 7600 Hz to remove DC offset, low-frequency noise, and high-frequency hiss.
3. **Normalization:** Apply Z-score normalization (subtract mean, divide by standard deviation) to the entire audio recording before windowing.

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt

def preprocess_audio(audio_data, original_sr=44100, target_sr=16000, lowcut=80, highcut=7600):
    # Downsample (assuming you have a resampling function)
    # For simplicity, we'll use scipy's resample, but for long signals, polyphase is better.
    from scipy.signal import resample
    num_samples = int(len(audio_data) * target_sr / original_sr)
    resampled_audio = resample(audio_data, num_samples)

    # Band-pass filter
    sos = butter(5, [lowcut, highcut], btype='band', fs=target_sr, output='sos')
    filtered_audio = sosfiltfilt(sos, resampled_audio)

    # Z-score normalization
    mean = np.mean(filtered_audio)
    std = np.std(filtered_audio)
    normalized_audio = (filtered_audio - mean) / std
    
    return normalized_audio, target_sr
```

### EEG Preprocessing

1. **Band-Pass Filtering:** Apply a Butterworth band-pass filter between 1 Hz and 40 Hz. This range captures the most relevant EEG rhythms (delta, theta, alpha, beta, low gamma) while removing DC drift and muscle artifacts (EMG).
2. **Notch Filtering:** Apply a notch filter at the power-line frequency (60 Hz or 50 Hz) and its harmonics to remove electrical interference.
3. **Normalization:** Apply Z-score normalization per channel to the entire EEG recording before windowing.

```python
def preprocess_eeg(eeg_data, sr=1024, lowcut=1.0, highcut=40.0, notch_freq=60.0):
    # Band-pass filter
    sos_band = butter(5, [lowcut, highcut], btype='band', fs=sr, output='sos')
    filtered_eeg = sosfiltfilt(sos_band, eeg_data, axis=0)

    # Notch filter
    from scipy.signal import iirnotch
    b_notch, a_notch = iirnotch(notch_freq, 30.0, sr)
    notched_eeg = sosfiltfilt(np.array([b_notch, a_notch]), filtered_eeg, axis=0)

    # Z-score normalization (per channel)
    mean = np.mean(notched_eeg, axis=0)
    std = np.std(notched_eeg, axis=0)
    normalized_eeg = (notched_eeg - mean) / std
    
    return normalized_eeg, sr
```

## 4. Spike-Encoding Methods

After windowing and preprocessing, the analog signals in each window must be converted to spikes.

### A. Poisson Rate Coding

Encodes signal amplitude as the firing rate of a Poisson process. Higher amplitude means more spikes.

- **Pseudo-math:** `P(spike at t) = λ * amplitude(t) * Δt` where `λ` is a scaling factor.
- **Hyperparameters:** `scaling_factor` (λ): controls the overall spike rate. Sweep `[0.1, 0.5, 1.0, 2.0]`.
- **Trade-offs:** Simple and robust but can be inefficient as it requires many spikes to represent a signal. It loses precise timing information.

```python
def rate_encode(window, scaling_factor=1.0, duration_ms=100):
    # Assuming window values are normalized and non-negative (e.g., after abs())
    spike_probs = window * scaling_factor / len(window)
    return (np.random.rand(*window.shape) < spike_probs).astype(np.float32)
```

### B. Time-to-First-Spike (TTFS) Coding

Encodes amplitude as the latency of a single spike. Higher amplitude means an earlier spike.

- **Pseudo-math:** `t_spike = T_max - (amplitude * T_max)` where `T_max` is the maximum simulation time (e.g., the window duration in ms).
- **Hyperparameters:** `T_max`: The simulation time window. This is usually fixed to the segment length.
- **Trade-offs:** Very efficient (one spike per neuron). Captures timing information precisely. Sensitive to noise and requires a reset mechanism. We can handle bipolar signals by using two channels per input neuron: one for positive and one for negative values.

```python
def ttfs_encode(window, T_max):
    # Assumes window values are in [0, 1]
    spike_times = T_max - (window * T_max)
    spike_times[window <= 0] = T_max # No spike for non-positive values
    return spike_times
```

### C. Ben's Spiking Algorithm (BSA) / Threshold-Based Encoding

Fires a spike when the signal's value crosses a threshold. The signal is then modulated to prevent immediate re-firing.

- **Pseudo-math:**
    1. `v(t) = input(t)`
    2. If `v(t) > threshold`: fire spike, `v(t) = v(t) - reset_value`.
- **Hyperparameters:** `threshold`, `reset_value`. Sweep `threshold` in `[0.5, 1.0, 1.5]` (assuming z-scored data). `reset_value` can be fixed to `threshold`.
- **Trade-offs:** More biologically plausible than rate coding. Efficient and captures temporal features. Performance is highly dependent on the `threshold`.

```python
def bsa_encode(window, threshold=1.0, reset_value=1.0):
    spikes = np.zeros_like(window)
    potential = 0
    for t in range(len(window)):
        potential += window[t]
        if potential > threshold:
            spikes[t] = 1
            potential -= reset_value
    return spikes
```

## 5. Sliding-Window Implementation

This function synchronizes the audio and EEG data streams and extracts windows.

```python
import numpy as np

def synchronized_windowing(eeg_data, audio_data, eeg_sr, audio_sr, window_sec, overlap_perc):
    """
    Extracts synchronized windows from EEG and audio data.
    
    Args:
        eeg_data (np.ndarray): Preprocessed EEG data (samples, channels).
        audio_data (np.ndarray): Preprocessed audio data (samples,).
        eeg_sr (int): Sampling rate of EEG.
        audio_sr (int): Sampling rate of audio.
        window_sec (float): Window length in seconds.
        overlap_perc (float): Overlap percentage (0.0 to 1.0).
        
    Returns:
        list of tuples: Each tuple contains (eeg_window, audio_window).
    """
    
    window_samples_eeg = int(window_sec * eeg_sr)
    window_samples_audio = int(window_sec * audio_sr)
    
    step_sec = window_sec * (1 - overlap_perc)
    step_samples_eeg = int(step_sec * eeg_sr)
    step_samples_audio = int(step_sec * audio_sr)
    
    windows = []
    
    num_samples_eeg = eeg_data.shape[0]
    num_samples_audio = len(audio_data)
    
    start_eeg, start_audio = 0, 0
    
    while (start_eeg + window_samples_eeg < num_samples_eeg) and \
          (start_audio + window_samples_audio < num_samples_audio):
        
        end_eeg = start_eeg + window_samples_eeg
        end_audio = start_audio + window_samples_audio
        
        eeg_window = eeg_data[start_eeg:end_eeg, :]
        audio_window = audio_data[start_audio:end_audio]
        
        windows.append((eeg_window, audio_window))
        
        start_eeg += step_samples_eeg
        start_audio += step_samples_audio
        
    return windows

# Example usage:
# eeg_data, eeg_sr = preprocess_eeg(...)
# audio_data, audio_sr = preprocess_audio(...)
# windows = synchronized_windowing(eeg_data, audio_data, eeg_sr, audio_sr, 1.5, 0.5)
```

## 6. Spiking Autoencoder Training Plan

- **Architecture:** A fully-connected feed-forward SNN.
  - **Input Layer:** Size matches the concatenated, flattened, and spike-encoded EEG and audio window data.
  - **Encoder:** 2-3 spiking layers (e.g., 1024 -> 512 neurons) with Leaky Integrate-and-Fire (LIF) neurons.
  - **Embedding Layer:** A bottleneck spiking layer of size `D` (e.g., `D=128` or `256`). This layer's spike output (or integrated membrane potential) will be the embedding.
  - **Decoder:** Symmetrical to the encoder (e.g., 512 -> 1024 -> Input Size).
- **Loss Function:** A combination of:
    1. **Reconstruction Loss:** Van Rossum distance or MSE between the original and reconstructed spike trains (after low-pass filtering).
    2. **Sparsity Loss:** L1/L2 regularization on the firing rates of the embedding layer to encourage sparse representations.
- **Training:**
  - **Batch Size:** 32 or 64.
    - **Optimizer:** Adam with a learning rate sweep `[1e-4, 5e-4, 1e-3]`.
    - **Surrogate Gradient:** Use a surrogate gradient function (e.g., `atan` or a fast sigmoid) to enable backpropagation through the spiking non-linearity.
    - **Early Stopping:** Monitor the validation reconstruction loss and stop if it doesn't improve for 10-15 epochs.
- **Outputting Embeddings:** After training, pass a window through the encoder and use the spike train, average firing rate, or mean membrane potential of the embedding layer as the fixed-length embedding vector.

## 7. Downstream ResNet-SNN Usage

- **Input Formatting:** The `D`-dimensional embedding vector from the SAE is treated as a static feature vector.
- **Architecture:** A Spiking ResNet (e.g., Spiking ResNet-18). The first layer of the ResNet will take the `D`-dimensional vector as input. The rest of the architecture follows the standard ResNet structure, but with all ReLU activations replaced by LIF neurons.
- **Training Protocol:**
  - This is a standard supervised classification task.
  - **Loss Function:** Cross-entropy loss on the output of the final layer (summed membrane potentials over time).
  - **Optimizer:** Adam or SGD with momentum.
  - **Evaluation:** Use a held-out test set of speakers (not seen during SAE or ResNet training).

## 8. Evaluation Metrics and Statistical Tests

- **Primary Metrics:**
  - **Speaker-ID Accuracy:** Top-1 classification accuracy.
  - **AUC/mAP:** Area Under the ROC Curve / mean Average Precision, especially for one-vs-rest evaluations.
- **Secondary (Ablation) Metrics:**
  - **SAE Reconstruction Loss:** To ensure the autoencoder is learning meaningful representations.
  - **Embedding Separability:** Use t-SNE to visualize embeddings and calculate the silhouette score to quantify cluster separation.
  - **Spike Budget:** Measure the average number of spikes per inference to evaluate computational efficiency.
- **Statistical Tests:**
  - Use **nested cross-validation**. The outer loop splits speakers into training/testing sets. The inner loop tunes hyperparameters (window size, overlap, etc.) on the training set.
  - Report `mean ± std` of the primary metrics across the outer folds.
  - Use the **Wilcoxon signed-rank test** to compare the performance of the top 2-3 pipeline configurations (e.g., 1.5s window vs 2.0s window) to see if the difference is statistically significant.

## 9. Final Recommended Pipeline (Conceptual)

```bash
# 1. Preprocess all data
python preprocess.py --audio_dir /path/to/audio --eeg_dir /path/to/eeg --output_dir /path/to/processed

# 2. Run hyperparameter sweep for windowing and encoding
for window in 1.0 1.5 2.0; do
  for overlap in 0.5 0.75; do
    for encoder in "rate" "bsa"; do
      
      # 3. Train Spiking Autoencoder
      python train_sae.py \
        --data_dir /path/to/processed \
        --window_sec $window \
        --overlap_perc $overlap \
        --spike_encoder $encoder \
        --embedding_dim 128 \
        --learning_rate 1e-4 \
        --output_model_path "/models/sae_w${window}_o${overlap}_e${encoder}.pt"

      # 4. Extract Embeddings
      python extract_embeddings.py \
        --sae_model "/models/sae_w${window}_o${overlap}_e${encoder}.pt" \
        --data_dir /path/to/processed \
        --output_embedding_path "/embeddings/w${window}_o${overlap}_e${encoder}/"

      # 5. Train and Evaluate Downstream Classifier
      python train_resnet_snn.py \
        --embedding_path "/embeddings/w${window}_o${overlap}_e${encoder}/" \
        --num_speakers N \
        --log_file "/results/results.csv"
        
    done
  done
done

# 6. Analyze results and run statistical tests
python analyze_results.py --results_file /results/results.csv
```

## 10. Concise Summary of Assumptions and Budget

- **Assumptions / Required Information:**
    1. **Number of Speakers & Data Balance:** The experimental design (especially cross-validation folds) depends on the number of speakers and the amount of recording time per speaker.
    2. **Signal Quality:** The preprocessing pipeline assumes moderately clean signals. If signals are extremely noisy (e.g., high motion artifacts in EEG), more advanced artifact removal (e.g., ICA) may be needed.
    3. **Synchronization Accuracy:** The pipeline assumes the EEG and audio streams are accurately synchronized with minimal, constant drift.
- **Recommended Computational Budget:**
  - **Hardware:** A modern GPU with at least 16 GB of VRAM is recommended to accommodate the SNN simulations and hyperparameter sweep. Neuromorphic hardware (e.g., Loihi) would be ideal but is not assumed.
  - **Expected Runtimes:** The full hyperparameter sweep will be computationally expensive. Training a single SAE model could take several hours to a day, depending on the dataset size. The entire grid search could take several days to weeks. It is advisable to start with a smaller subset of the data and a reduced grid to get initial estimates.

---

## JSON Output

```json
{
  "best_window": 1.5,
  "overlap": 0.5,
  "grid": {
    "window_lengths_sec": [0.2, 0.5, 1.0, 1.5, 2.0, 3.0],
    "overlaps_perc": [0.25, 0.5, 0.75]
  },
  "preprocessing": {
    "audio": "Downsample to 16kHz, band-pass 80-7600Hz, Z-score normalization.",
    "eeg": "Band-pass 1-40Hz, notch filter at 60/50Hz, Z-score normalization per channel."
  },
  "spike_encoders": {
    "poisson": "Rate coding, hyperparameter: scaling_factor.",
    "ttfs": "Time-to-first-spike, hyperparameter: T_max (fixed to window).",
    "bsa": "Threshold-based, hyperparameters: threshold, reset_value."
  },
  "windowing_code": "def synchronized_windowing(eeg_data, audio_data, eeg_sr, audio_sr, window_sec, overlap_perc): ...",
  "autoencoder_plan": {
    "architecture": "Feed-forward SNN with LIF neurons and bottleneck embedding layer.",
    "loss": "Reconstruction loss (Van Rossum or MSE) + Sparsity regularization.",
    "training": "Surrogate gradients (atan), Adam optimizer, early stopping."
  },
  "downstream_plan": {
    "model": "Spiking ResNet-18.",
    "input": "Static D-dimensional embedding vector from SAE.",
    "training": "Supervised classification with cross-entropy loss."
  },
  "evaluation": {
    "metrics": ["Speaker-ID Accuracy", "AUC", "SAE Reconstruction Loss", "Silhouette Score"],
    "protocol": "Nested cross-validation with outer loop for speakers.",
    "statistical_test": "Wilcoxon signed-rank test for comparing top pipelines."
  },
  "assumptions": {
    "required_info": ["Number of speakers", "Data balance per speaker", "Signal quality/SNR"],
    "compute_budget": "GPU with >16GB VRAM recommended. Full grid search may take several days."
  }
}
```

---

## Experiment Checklist

1. `run_experiment --window 1.5 --overlap 0.5 --encoder bsa --lr 1e-4`
2. `run_experiment --window 2.0 --overlap 0.75 --encoder bsa --lr 1e-4`
3. `run_experiment --window 1.5 --overlap 0.5 --encoder rate --lambda 0.5 --lr 5e-4`

---

### References

- García-Perera, L. P., et al. (2021). *A review on deep learning for speaker recognition*. Expert Systems with Applications.
- Snyder, D., et al. (2018). *X-vectors: Robust d-vector embeddings for speaker recognition*. IEEE ICASSP.


Following these rules will keep Copilot (or any code-generation assistant) from producing code that triggers the project's compiler warnings or CMake parse errors. If you have an automated flow that edits many files, run the full build and tests locally after generation and fix any linter or build failures before pushing.

