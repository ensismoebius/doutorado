# Repository Naming Conventions

## Scope
These conventions apply to C/C++ code under `include/` and `src/`.

## Identifiers
- Types (`class`, `struct`, `enum`, aliases): `PascalCase`.
- Functions and methods: `snake_case`.
- Variables (local/member/parameter): `snake_case`.
- Constants: `kCamelCase` for `constexpr`/compile-time constants.
- Macros and include guards: `UPPER_SNAKE_CASE`.

## Acronyms
- Acronyms may stay uppercase in type names when domain clarity improves readability (for example: `EEGSample`, `SNNResNet`).
- Prefer lowercase acronyms in function/variable names for consistency with `snake_case` (for example: `load_eeg_data`).

## File Naming
- Public headers: descriptive `PascalCase` only when matching type-centric modules already established.
- Internal/source files: follow existing module naming in directory; avoid introducing mixed-language names in new files.

## Language Consistency
- New C/C++ identifiers should use English.
- Existing Portuguese identifiers may remain in legacy/demo code until a dedicated migration pass to avoid behavior risk.

## Refactoring Policy
- Prefer behavior-preserving renames with small PR-sized batches.
- Run build + targeted smoke checks after each rename batch.
