# Repository Naming Conventions

## Scope

These conventions apply to C/C++ code under `include/` and `src/`.

## Identifiers

| Type | Convention | Example |
|------|------------|---------|
| Types (`class`, `struct`, `enum`, aliases) | `PascalCase` | `Tensor`, `LinearLayer`, `OptimizerType` |
| Functions and methods | `snake_case` | `forward_pass`, `load_state_dict` |
| Variables (local/member/parameter) | `snake_case` | `batch_size`, `learning_rate` |
| Constants (`constexpr`) | `kCamelCase` | `kDefaultBatchSize`, `kMaxEpochs` |
| Macros and include guards | `UPPER_SNAKE_CASE` | `NN_TENSOR_HPP`, `EIGEN_MATRIX_H` |

## Acronyms

- **In type names**: May stay uppercase for domain clarity
  - `EEGSample`, `SNNResNet`, `LSTMModel`
- **In function/variable names**: Use lowercase
  - `load_eeg_data`, `snn_forward`, `lstm_hidden`

## File Naming

- **Public headers**: Descriptive `PascalCase` when matching type-centric modules
- **Internal/source files**: Follow existing module naming; avoid mixed-language names

## Language Consistency

- New identifiers use English
- Existing Portuguese identifiers may remain in legacy/demo code

## Refactoring Policy

- Prefer behavior-preserving renames with small PR-sized batches
- Run build + targeted smoke checks after each rename batch

## See Also

- [Architecture](../Architecture.md)
- [Core Modules](../Core/Tensor.md)