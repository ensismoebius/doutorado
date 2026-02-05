# Voice Biometrics Spike Pipeline — Architecture and Portability Guide

## A. System Overview
- Purpose: end-to-end speaker identification/verification demo built from audio capture through wavelet features, spike encoding, and a residual spiking neural network (SNN). Implements CLI-driven flows: demo visualization, enrollment (capture), training, identification, verification with unknown threshold, and offline evaluation.
- Functional goals: produce per-speaker classification with adjustable depth and hidden size, expose data capture and persistence, generate plots for didactic inspection, and support hyperparameter experiments.
- Execution lifecycle (typical train+infer): capture/gather WAVs → resample → window audio → compute Wavelet Packet Transform (WPT) energies → log+normalize to [0,1] → Poisson spike encoding over multiple time steps per window → residual SNN forward → spike count/membrane readout → loss or decision → optional confidence threshold for unknown.
- Inputs: mono audio streams (live microphone or WAV PCM 16/8/32), configuration parameters (sampling rate, window/hop, wavelet, bands, SNN depth/hidden, loss mode), labeled speaker directories. Outputs: trained weights (torch state dict), label metadata (JSON with labels, depth, hidden), plots (PNG), confusion matrix text, experiment CSVs.
- Side-effects: writes WAVs per person under dados/vozes/<id>, saves model/label files, writes experiment CSV/PNG, prints progress logs; uses deterministic seeding where defined (e.g., torch.manual_seed in model factory) but Poisson coding remains stochastic unless seeds set externally.

## B. Architecture Diagram (Textual)
- Modules: CLI (app/main) → Command handlers (app/comandos) → Services (services/*) → Infra (captura, arquivo_audio, visualizacao) → Utils (caracteristicas, preprocessamento, codificacao, janelamento, ondaletas) → Domain rules (regras) → Core configs (dataclasses) → Model (services/modelos/rede_snn).
- Data flow: Microphone/WAV → captura.resample → janelamento.aplicar_janelamento → utils.ondaletas.calcular_nivel_wpt → utils.caracteristicas.calcular_energia_wpt → utils.preprocessamento.preprocessar_energia_wpt_para_snn → utils.codificacao.codificar_poisson (T steps) → rede_snn forward (stateful) → loss/readout in identificacao_locutor.compute_loss → decision/threshold → visualization or persistence.
- Control flow: CLI selects subcommand; demo/identificar/verificar/avaliar call capture or dataset loader then run inference; treinar/experiments loop over epochs, batches, and optional multi-pass per batch; experiments orchestrate grids and validation splits.
- Core ↔ Vendor dependency graph: pydemos uses NumPy, PyWavelets, matplotlib, sounddevice (PortAudio), SciPy (optional), torch + snnTorch. Project-level vendor CMake supplies Eigen, FFTW3, NFFT3, cnpy, yaml-cpp, matio/matio-cpp, argparse, imgui/implot, matplotlib-cpp, GoogleTest. When porting, map these to equivalent numerical kernels, I/O, plotting, CLI, and testing stacks.
- Extension points: configurable SNN depth/hidden, loss modes (rate/MC/temporal_pooling/van_rossum/membrane/cosine/mse_vector), spike target rate, dataset overrides for experiments, plotting outputs, threshold for unknown.

## C. Core Library Mapping (src/core — MUST REUSE)
- tensor: Tensor wrapper with Eigen backend; provides shapes, math ops, gradients; reuse for numerical storage instead of ad-hoc arrays. In other languages, bind to equivalent dense tensor (e.g., Eigen, ndarray, ArrayFire) preserving contiguous layout and gradient hooks; do not replace unless missing feature proven.
- layers: Module interface (`forward`, `backward`, `train`, `params`), Linear layer, Leaky (LIF) SNN layer, Sequential, losses, surrogate gradients; mandatory reuse for neural components—write adapters rather than reimplement unless performance evidence.
- dataLoaders: Dataset/DataLoader abstractions yielding Batch{inputs,targets} with deterministic shuffling via seeds; reuse for dataset iteration and batching semantics.
- optimizers: Adam/SGD interfaces with `step`/`zero_grad`/`attach`; reuse for parameter updates.
- statistics: metrics/test utilities; reuse for evaluation pipelines if applicable.
- utility: batching helpers, printing, etc.; reuse instead of custom loops when available.
- wave/wavelet: signal processing helpers; prefer these over bespoke implementations for consistency.
- saver: serialization helpers; reuse for model persistence when integrating with C++ stack.
- paraconsistent/linearAlgebra/initializers: specialized logic and math helpers; use directly where domain aligns.
- Reuse rule: any new implementation (even in another language) should preserve the above contracts; if binding, mirror interfaces; reimplementation allowed only with proof of missing capability or performance need, and documentation of divergence.

## D. Vendor Dependency Mapping (cmake/Vendor*)
- Eigen (VendorEigenParallel): primary linear algebra backend with OpenMP; substitute only with equivalently optimized BLAS/LAPACK-backed tensors; preserve float precision and threading behavior.
- FFTW3 (VendorFFTW): FFT with OpenMP/threading; substitute with high-performance FFT (MKL FFT, cuFFT) maintaining numerical fidelity and license compatibility.
- NFFT3 (VendorNFFT3): nonequispaced FFT; substitute with comparable NUFFT library (FINUFFT) with shared lib and OpenMP support.
- cnpy (VendorCnpy): NumPy .npy/.npz I/O; replace with format-compatible readers/writers only.
- matio + matio-cpp (VendorMatio, VendorMatioCppShim): MATLAB .mat I/O; substitute with libraries preserving v5 numeric support and safety caps.
- yaml-cpp (VendorYaml): YAML config; substitute with spec-compliant parser and similar API semantics.
- matplotlib-cpp (VendorMatplotlibCpp): plotting bridge; when porting, use equivalent plotting backend (matplotlib, seaborn, Vega) respecting headless usage.
- argparse (VendorArgparse): CLI parsing; use standard CLI parsers matching behavior.
- imgui/implot (VendorImgui, VendorImplot): UI/plot widgets; replace only with equivalent immediate-mode GUI if UI features required.
- GoogleTest (VendorGTest): testing; replace with mature unit-test framework offering fixtures and assertions.
- VendorIncludes aggregates the above; maintain linkage order and OpenMP propagation.

## E. Language-Agnostic Implementation Guide
- Layers: separate CLI/interface, domain rules, infra (I/O), services (business logic), utils (math/feature), model definition, experiments. Maintain dependency direction: CLI → services → utils/infra → core; avoid cycles.
- Interfaces: ConfigExtracao/ConfigSNN as immutable config structs; capture functions returning float32 mono vectors; dataset loader producing windowed, normalized feature matrices and integer labels; model factory accepting num_inputs/outputs, hidden size, residual depth; training loop consuming spike-encoded tensors and returning loss/acc stats.
- Data schemas: audio float32 [-1,1]; windows length `tamanho_janela`, hop `tamanho_passo`; WPT energies length `num_bandas`; normalized features [0,1]; spike tensors shape [T, B, F]; SNN outputs [T, B, C]; labels aligned to sorted person IDs.
- Runtime model: stateful SNN with explicit membrane state passed between steps; Poisson encoder stochastic per step; training iterates batches with optional multi-pass averaging; inference aggregates per-window probabilities then averages across windows for utterance decision; verification applies threshold to confidence.
- Portability rules: avoid CPython specifics; encapsulate randomness with seedable RNG; expose deterministic state init; map torch operations to target backend maintaining tensor shapes and broadcasting semantics; keep loss modes and readouts semantically equivalent.

## F. Engineering Constraints & Design Laws
- Abstractions: never bypass core tensor/layer/DataLoader abstractions; keep modular boundaries (CLI vs infra vs services vs model). No hidden global state.
- Error handling: raise explicit errors on missing windows, incompatible WAV formats, or invalid shapes; propagate exceptions rather than silent fallback (except documented noise fallback in capture when devices fail).
- Logging: keep progress logs for capture, training batches/epochs, and visualization outputs; in other languages use structured logging with similar granularity.
- Threading/async: capture is synchronous; training uses batched CPU/GPU loops; preserve ordering of temporal loops (time-major) and state updates; if parallelizing, do not reorder time steps.
- Determinism: torch.manual_seed(42) in model factory; Poisson encoding stochastic—provide seed control when porting; when re-sampling/janelamento ensure deterministic pad/truncation.
- Prohibited actions: bypass src/core abstractions, collapse modular boundaries, remove determinism safeguards, introduce hidden state, alter reproducibility behavior without explicit config.

## G. Extension & Scaling Model
- Adding features: introduce new CLI subcommands by delegating to services; add new loss modes or encoders by extending compute_loss/codificacao with clear contracts; keep configs dataclasses immutable.
- New modules: place feature extraction in utils, I/O in infra, business logic in services; respect dependency direction; reuse core modules where available (e.g., C++ tensor/layers for performance-sensitive ports).
- Scaling: parallelize dataset preprocessing (window+WPT) while preserving per-sample ordering; batch sizes adjustable; use device selection (CPU/GPU) abstraction; for larger models increase residual blocks/hidden with consideration of memory O(T*B*(hidden+classes)).
- Maintainability: document depth/hidden in metadata (already saved); version configs; keep experiment scripts writing CSV for traceability.

## H. Validation & Test Strategy
- Required tests: unit tests for feature extraction (WPT energy shape, normalization), Poisson encoder (rate statistics, adaptive frequency), janelamento (count, overlap), loss modes (behavioral parity), model forward (2D vs 3D inputs, stateful continuity), threshold rule; dataset loader (label ordering, resample correctness).
- Regression: reproduce CLI flows demo/treinar/identificar/avaliar; ensure confusion matrix output stable for fixture data.
- Cross-language equivalence: compare feature vectors, spike statistics, and model outputs against reference Python for golden inputs; validate JSON metadata parsing; ensure Poisson RNG seeded yields comparable distributions.
- Determinism validation: fixed seed should give stable model init and training trajectory given deterministic backend; add checks for padding/truncation logic.
- Performance benchmarks: measure windowing+WPT throughput, spike encoding throughput, training epoch time and memory footprint; compare against baseline to ensure no regressions.

## I. Compliance Checklist
- Uses src/core abstractions where functionality exists; no reimplementation without justification.
- Honors vendor dependency roles and substitutes only with equivalent precision/performance/licensing.
- Preserves execution order: capture → window → WPT → preprocess → Poisson → SNN → readout → decision.
- Maintains explicit state handling, reproducibility hooks, and error propagation; no hidden state or reordered time loops.
- Provides regression, equivalence, determinism, and performance tests before acceptance.
- Updates metadata (labels JSON, configs) and keeps outputs/paths consistent; writes experiment results and logs.

(End of guide)
