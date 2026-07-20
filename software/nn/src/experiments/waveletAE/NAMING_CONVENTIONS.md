# Experiment 02 Naming Convention

## Rules
- `PascalCase` for types and structs (example: `EEGSample`, `AggregatedFoldResults`).
- `snake_case` for functions, local variables, and file-scope helpers (example: `load_eeg_data`, `sample_index`).
- `kCamelCase` for constants and fixed configuration values (example: `kEegChannelCount`).
- Acronyms may stay uppercase in domain-specific type names when clearer (example: `EEGSample`).

## Applied in This Module Set
- Replaced magic numbers in data extraction with named constants.
- Centralized synthetic sample generation in `generate_synthetic_samples(...)`.
- Replaced repetitive wavelet `if` dispatch with table-driven dispatch (`kWaveletDispatch`).
