Title: Fixes to EEGLoader vtable emission and MatFileDataset placement-new

Summary
- This document explains the code fixes I applied to get a clean build and all tests passing under sanitizers.
- Files changed:
  - [src/core/dataLoaders/10.1117/EEGLoader.cpp](src/core/dataLoaders/10.1117/EEGLoader.cpp)
  - [src/core/dataLoaders/MatFileDataset.h](src/core/dataLoaders/MatFileDataset.h)
  - [src/core/dataLoaders/TensorDataset.h](src/core/dataLoaders/TensorDataset.h)

Problem 1 — Linker errors: missing typeinfo / vtable for `EEGLoader`

What happened
- The build failed with linker errors such as:
  "/usr/bin/ld: ../libdataLoaders_10_1117.a(EEGLoader.cpp.o): undefined reference to `typeinfo for nn::dataLoaders::EEGLoader'"
- This points to missing RTTI/typeinfo and vtable emission for the `EEGLoader` class.

Why this occurs (background)
- In C++ the compiler emits the vtable and typeinfo for a polymorphic class in exactly one translation unit: the TU that contains the class's key function.
- The "key function" is typically the first non-inline, non-pure virtual function defined out-of-line. If all virtual functions (including the destructor) are defined inline in a header or are missing, the compiler may not emit the required typeinfo/vtable into any object file.
- The link error is the linker telling you it couldn't find that emitted RTTI/typeinfo symbol.

What I changed
- I added an out-of-line virtual destructor definition for `nn::dataLoaders::EEGLoader` in `EEGLoader.cpp` and implemented the declared virtual methods that had been left undefined (`open`, `close`, `readVariable`, etc.).
- This ensures the class's key function is present in `EEGLoader.cpp`, causing the vtable/typeinfo to be emitted into that object file and resolving the linker error.

Why this fix is correct
- Providing an out-of-line destructor (or any other out-of-line virtual method) is a common and correct way to guarantee vtable/typeinfo emission without changing the class API. It also avoids fragile inline-only polymorphic types.

Problem 2 — Runtime UBSAN error: placement-new of base subobject in `MatFileDataset` ctor

What happened
- Tests initially passed mostly, but one test failed under sanitizers with a dynamic-type/UBSAN runtime error showing that an object claimed to be `TensorDataset` where `MatFileDataset` was expected (vptr mismatch). This manifested as "member call on address ... which does not point to an object of type 'MatFileDataset' — object is of type 'TensorDataset'".
- Investigation found the code in `MatFileDataset`'s constructor used placement-new on `this`:
  - new (this) TensorDataset(std::move(inputs), std::move(targets));
- This is undefined behavior: you cannot safely placement-new construct the base subobject this way from within the derived object's constructor body.

Background / C++ object-model explanation
- When a derived object is created, its base subobjects and vptrs are constructed by the runtime in the well-defined order determined by the constructors and the initializer list.
- Doing placement-new on `this` to construct a subobject overwrites memory (including the vptrs already set by the compiler) and leads to UB: the object representation and the compiler's assumptions about object lifetime are violated.
- The correct way to initialize base class state is to call the base class constructor from the derived constructor's initializer list, or to design the classes so the derived class assigns into protected or public members that are intended to be set after construction.

What I changed (concrete)
- I added a small protected setter to `TensorDataset`:
  - `protected: void set_tensors(nn::Tensor inputs, nn::Tensor targets);`
  This assigns into the base's `inputs_` and `targets_` members.
- I replaced the placement-new in `MatFileDataset` with a call to `set_tensors(std::move(inputs), std::move(targets));`.

Why I chose this fix
- It is the minimal, low-risk change that removes the UB and preserves the existing class hierarchy and public API.
- Alternatives considered:
  - Change `MatFileDataset` to call the base constructor in the member initializer list. That would require rearranging the code to perform the MAT loading before the initializer list (e.g., using a static factory function or helper) so that the base can be constructed with the loaded tensors directly. That is a cleaner design but requires a larger refactor (factory or helper) and more churn across call sites.
  - Replace inheritance with composition (i.e., have `MatFileDataset` contain a `TensorDataset` member). Also a safe design but more invasive.
- Given the goal of getting the tree green with minimal invasiveness, adding `set_tensors` was appropriate. It documents intent and keeps the initialization explicit and safe.

Validation and results
- After these changes I rebuilt the project and executed tests with sanitizers enabled (ASAN, UBSAN, LSAN).
- The previous linker errors were resolved; all tests passed.
- I re-ran tests with increased sanitizer verbosity (ASAN/LSAN/UBSAN environment options). No leaks or UBSAN errors were reported.

Files touched (one-line rationale)
- [src/core/dataLoaders/10.1117/EEGLoader.cpp](src/core/dataLoaders/10.1117/EEGLoader.cpp): added out-of-line destructor and implemented missing virtual methods so the vtable/typeinfo are emitted and the class is fully defined.
- [src/core/dataLoaders/TensorDataset.h](src/core/dataLoaders/TensorDataset.h): added a `protected` setter `set_tensors(...)` to allow safe post-construction initialization of the base's storage.
- [src/core/dataLoaders/MatFileDataset.h](src/core/dataLoaders/MatFileDataset.h): replaced placement-new with a call to `set_tensors(...)`.

Suggested follow-ups (recommended)
1. Consider refactoring `MatFileDataset` to initialize the base from the initializer list using a static helper/factory:
   - Example pattern: add a private static helper `load_tensors_from_mat(...) -> std::pair<nn::Tensor, nn::Tensor>` and use `MatFileDataset(...) : TensorDataset(load_tensors_from_mat(...).first, load_tensors_from_mat(...).second) {}` or a two-stage factory `MatFileDataset::FromFile(...)` returning a constructed instance.
   - This is the most idiomatic C++ approach and avoids needing a setter at all.
2. Add an explicit comment in `TensorDataset.h` describing `set_tensors` and why it exists (transitional safety measure). If you accept the factory refactor later, remove the setter to reduce API surface.
3. Add a small targeted unit test that constructs `MatFileDataset` and checks its RTTI/vptr correctness under UBSAN to prevent regressions.
4. If you plan more changes to `EEGLoader` or other polymorphic types, prefer defining at least one non-inline virtual method (commonly the destructor) out-of-line in the corresponding cpp file to guarantee vtable emission.

Notes about decisions
- The changes were deliberately minimal to keep churn low and restore a working sanitizer-checked test run quickly.
- All changes are localized to `src/core/dataLoaders/` and documented here so future reviewers understand the intent and the safer alternatives.

If you want
- I can:
  - Implement the factory-style refactor for `MatFileDataset` now (safe, slightly larger change).
  - Add the suggested unit test(s) and a short code comment explaining `set_tensors`.
  - Commit the changes on a branch and open a PR message draft describing the fixes.

Status
- Local edits applied, rebuild and full test run under sanitizers passed (no leaks detected). The codebase is currently green.

---
Generated on 2025-12-11
