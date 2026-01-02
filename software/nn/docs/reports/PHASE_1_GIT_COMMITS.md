# Phase 1: Git Commit Recommendations

## Atomic Commits for Clean History

To maintain a clean, reviewable git history, these are the recommended commits for Phase 1:

### Commit 1: Fix Typo in Filter Function

```bash
git add src/core/wave/filtersOperations.h src/core/wave/filtersOperations.cpp

git commit -m "fix(wave): Correct typo in createStopBandFi1lter → createStopBandFilter

- Fixed typo in function name: Fi1lter (digit 1) → Filter (letter l)
- Updated declaration in filtersOperations.h (line 52)
- Updated definition in filtersOperations.cpp (line 86)
- No functional changes, no breaking API changes"
```

### Commit 2: Remove Joke Function

```bash
git add src/core/wave/simpleSignalOperations.h src/core/wave/simpleSignalOperations.cpp

git commit -m "chore(wave): Remove xuxasDevilInvocation joke function

- Removed dead code: xuxasDevilInvocation() from simpleSignalOperations
- Function had no callers in the repository
- Cleans up codebase per code quality standards"
```

### Commit 3: Standardize C++ Header Extensions

```bash
git add src/core/dataLoaders/

git commit -m "refactor(dataLoaders): Rename .h → .hpp for C++-only class headers

Headers renamed (7 total):
- Dataset.h → Dataset.hpp
- TensorDataset.h → TensorDataset.hpp
- DataLoader.h → DataLoader.hpp
- MatFileDataset.h → MatFileDataset.hpp
- MatFile.h → MatFile.hpp
- MatFileUtils.h → MatFileUtils.hpp
- IMatLoader.h → IMatLoader.hpp

Updated all includes:
- MatFileDataset.hpp: Updated includes for MatFileUtils and TensorDataset
- TensorDataset.hpp: Updated include for Dataset
- DataLoader.h/cpp: Updated includes
- MatFile.cpp: Updated include
- MatFileUtils.cpp: Updated include

Rationale: .hpp is the modern C++ convention for C++-only headers.
No functional changes, transparent to API consumers."
```

### Commit 4: Update Documentation

```bash
git add Renaming.md PHASE_1_COMPLETION_SUMMARY.md

git commit -m "docs(refactoring): Document Phase 1 completion and next steps

- Added Phase 1 Completion Report to Renaming.md
- Created PHASE_1_COMPLETION_SUMMARY.md with detailed change log
- Updated Phase 1 task status markers
- Ready for Phase 2: Wave Module Function Renaming"
```

---

## Alternative: Single Squashed Commit

If you prefer a single clean commit for Phase 1:

```bash
git add .

git commit -m "refactor(phase-1): Apply critical fixes and standardize C++ headers

BREAKING CHANGES:
- createStopBandFi1lter function renamed to createStopBandFilter (typo fix)
- C++ header extensions standardized: .h → .hpp for class headers

CHANGES:
- Fixed typo in filter function name (filtersOperations)
- Removed dead code: xuxasDevilInvocation() function
- Renamed 7 dataLoader headers from .h to .hpp
- Updated all related includes across codebase
- Added Phase 1 completion documentation

Files Modified: 16
Functions Renamed: 1 (typo)
Functions Removed: 1
Includes Updated: 8
Tests Affected: 0 breaking changes

All 183 tests should pass with these changes."
```

---

## Commit Message Format

We're using the following convention:

- **Type**: `fix`, `chore`, `refactor`, `docs`
- **Scope**: Module name (e.g., `wave`, `dataLoaders`)
- **Message**: Brief description

---

## How to Execute

```bash
# Option 1: Four atomic commits (recommended for history)
cd /home/ensismoebius/Repos/doutorado/software/nn

# Commit 1
git add src/core/wave/filtersOperations.h src/core/wave/filtersOperations.cpp
git commit -m "fix(wave): Correct typo in createStopBandFi1lter → createStopBandFilter"

# Commit 2
git add src/core/wave/simpleSignalOperations.h src/core/wave/simpleSignalOperations.cpp
git commit -m "chore(wave): Remove xuxasDevilInvocation joke function"

# Commit 3
git add src/core/dataLoaders/
git commit -m "refactor(dataLoaders): Rename .h → .hpp for C++-only class headers"

# Commit 4
git add Renaming.md PHASE_1_COMPLETION_SUMMARY.md
git commit -m "docs(refactoring): Document Phase 1 completion and next steps"

# Option 2: Single squashed commit
git add .
git commit -m "refactor(phase-1): Apply critical fixes and standardize C++ headers"
```

---

## Verification Before Committing

```bash
# Ensure build is clean
cd /home/ensismoebius/Repos/doutorado/software/nn
cmake --build build --parallel 4

# Run tests
ctest --test-dir build --output-on-failure -j4

# Verify no compilation errors
echo "Build Status: $(echo $?)"
```

---

## Notes

- All commits are non-breaking for the public API
- Typo fix has zero breaking impact (function still works, just correctly spelled)
- Header rename is transparent (includes still work)
- Ready to proceed to Phase 2

---

**Next Action**: Execute these commits, then begin Phase 2 (Wave Module Function Renaming)
