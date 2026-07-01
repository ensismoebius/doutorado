# Profile-Guided Optimization (PGO) Workflow

## What PGO Is and Why It Helps

A normal `-O3` build makes every optimization decision from **static heuristics**: the compiler guesses which branches are likely taken, which functions are hot enough to inline, and how to lay out basic blocks in memory — all without ever running the program. PGO replaces those guesses with **measured** execution data: build an instrumented binary, run it on a representative workload, and feed the recorded counts back into a second compilation.

Three optimization classes concretely change once real counts are available:

1. **Basic-block / hot-cold splitting.** Pettis and Hansen showed that reordering code so hot paths are contiguous (and cold paths — error handling, rare branches — are moved out of line) reduces instruction-cache misses and page faults, because the working set the CPU actually executes shrinks [52].
2. **Inlining decisions.** Size-based inlining heuristics routinely reject call sites that are actually hot. With edge counts, the compiler inlines the call sites that dominate runtime even if they'd normally be judged "too big," and avoids inlining rarely-executed ones [53].
3. **Branch prediction and if-conversion.** Static branch prediction (e.g. "loop back-edges are taken") is replaced by measured taken/not-taken frequencies, which also feeds register allocation (keep hot-path values in registers) and if-conversion decisions [53].

This is a two-phase workflow: **generate** (instrument, run, collect counts) then **use** (recompile, folding the counts into the optimizer).

## This Project's Presets

Configured in `CMakePresets.json` (values below read directly from the preset, not paraphrased):

| Preset | `CMAKE_BUILD_TYPE` | Flags | IPO/LTO | Fast linker |
|---|---|---|---|---|
| `pgo-generate` | Release | `-O3 -march=native -fprofile-generate -DNDEBUG -pipe` | OFF | OFF |
| `pgo-use` | Release | `-O3 -march=native -fprofile-use -fprofile-correction -flto -DNDEBUG -pipe` | ON | OFF |

Both presets set `CMAKE_C_COMPILER_LAUNCHER`/`CMAKE_CXX_COMPILER_LAUNCHER` to `ccache`. Neither pins a compiler, so the build uses whatever `cc`/`c++` resolves to (GCC or Clang — see [Advanced (Clang)](#advanced-clang) below for the difference that matters).

`-fprofile-correction` in `pgo-use` tells the compiler to tolerate profile data that's slightly inconsistent with the current source (e.g. after a small edit since the `pgo-generate` run) instead of aborting — useful because profiles go stale the moment the code changes, but it is not a substitute for periodically re-generating profiles after real changes to hot paths.

### 1. Generate Instrumented Build

```bash
cmake --preset pgo-generate
cmake --build --preset pgo-generate -j$(nproc)
```

Then run a **representative workload** — one that actually exercises the hot paths you care about (data loading, batching, forward/backward passes), not just a trivial smoke test. For example, Experiment03's training-flow profile exercises the fused-window data loader, prefetcher, and a small ANN's forward/backward pass end-to-end:

```bash
./out/build/pgo-generate/src/experiments/03/experiment03 \
    --profile sample-training-flow
```

`--profile` takes a **profile-name stem** (the `.json` file's basename under `src/experiments/03/profiles/`), not a path — `ProfileLoader::load` resolves the stem against several candidate directories and appends `.json` itself. Passing a full path with extension only works if that exact file exists.

> **Caveat:** `sample-training-flow.json` (and the other Experiment03 profiles) set `dataset_root_path` to a private, unpublished dataset path. Running it as-is requires that dataset; without it, substitute any profile/dataset combination that touches the same code paths (data loader → prefetcher → network forward/backward), since what matters for PGO is *which code executes*, not the specific dataset.

The compiler writes profiling counters (`.gcda` files, GCC format) into the build directory as the instrumented binary runs.

### 2. Produce Final PGO-Optimized Build

```bash
cmake --preset pgo-use
cmake --build --preset pgo-use -j$(nproc)
```

This recompiles from scratch (a fresh `binaryDir`, `out/build/pgo-use`) using the counts collected in step 1, with LTO also enabled so the compiler can apply profile-informed cross-translation-unit inlining, not just within a single `.cpp` file.

## Fast Linker Is Off Whenever LTO Is On (Project-Wide, Not PGO-Specific)

`NN_ENABLE_FAST_LINKER=OFF` in both `pgo-generate` and `pgo-use` is not a PGO peculiarity — `max-performance` (the project's other LTO preset) disables it too. `cmake/Flags.cmake` enables `mold`/`lld` (`-fuse-ld=`) only when `NN_ENABLE_FAST_LINKER=ON`, and includes special-case handling to make `lld` load GCC's `liblto_plugin.so` for GCC+LTO builds — but that combination has proven unreliable enough that every LTO preset in this project ships with the fast linker off by default. If you re-enable it, verify the resulting binary actually links and runs (a silent bad-codegen link is worse than a slow one).

## Advanced (Clang)

The presets pin no compiler, so building with `CC=clang CXX=clang++ cmake --preset pgo-generate` works — Clang accepts `-fprofile-generate`/`-fprofile-use` as GCC-compatible flag names. But the **on-disk format differs**, and that changes the workflow:

- **GCC**: `-fprofile-generate` writes one `.gcda` file per translation unit next to the object files. `-fprofile-use` reads that directory directly — no merge step, which is why the two-command sequence above is complete as written.
- **Clang**: `-fprofile-generate` instruments at the LLVM IR level and writes raw counters (`default.profraw`, or `$LLVM_PROFILE_FILE` if set) at *program exit*, not per translation unit. Before rebuilding, you must merge it into an indexed profile:
  ```bash
  llvm-profdata merge -output=default.profdata default.profraw
  ```
  and pass the merged file explicitly:
  ```bash
  cmake --preset pgo-use -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -fprofile-use=default.profdata -flto -DNDEBUG -pipe"
  ```
  Skipping the merge, or pointing `-fprofile-use` at the raw file, silently falls back to un-profiled `-O3` on Clang rather than erroring.

A separate, source-level instrumentation mode also exists in Clang (`-fprofile-instr-generate` / `-fcoverage-mapping`, paired with `llvm-cov`) — that one is for **coverage reporting**, not optimization; don't conflate it with `-fprofile-generate`/`-fprofile-use` above.

## Measuring the Gain

There is no PGO benchmark checked into this repo yet — treat any specific speedup number for this project as unverified until measured. To measure honestly:

1. Build `pgo-use` **and** a same-flags-minus-PGO baseline (`max-performance`, which shares `-O3 -march=native -flto`) from a clean `binaryDir` each, so ccache doesn't mask the comparison.
2. Time the same workload multiple times on each binary (`time ./binary ...`, or `perf stat -r 5 ./binary ...` if `perf` is available) and compare medians, not single runs — build-machine noise (thermal throttling, background load) easily swamps a few-percent PGO gain.
3. Record the profile used, the flags of both binaries, and the measured numbers in this page (or link to a `results/` artifact) so the claim stays falsifiable — see [LSTM-Performance.md](LSTM-Performance.md) for the format this project uses for that kind of table.

## Troubleshooting

- **Linking fails with fast linker enabled:** set `NN_ENABLE_FAST_LINKER=OFF` (see above) — this is the default for both PGO presets already.
- **Profile changed but source hasn't been re-profiled:** `-fprofile-correction` (already in `pgo-use`) tolerates small drift; a `pgo-generate` → `pgo-use` re-run is still needed after any change to hot-path code, since stale profiles guide the optimizer toward code paths that are no longer representative.
- **`pgo-use` binary is no faster than `max-performance`:** the workload used in `pgo-generate` may not exercise the same code paths as your target workload — PGO amplifies gains only where the profile and the deployment workload overlap.

## References

[52] K. Pettis and R. C. Hansen, "Profile guided code positioning," in *Proc. ACM SIGPLAN 1990 Conf. on Programming Language Design and Implementation (PLDI)*, White Plains, NY, 1990, pp. 16–27. [Online]. Available: https://doi.org/10.1145/93542.93550

[53] P. P. Chang, S. A. Mahlke, and W. W. Hwu, "Using profile information to assist classic code optimizations," *Software: Practice and Experience*, vol. 21, no. 12, pp. 1301–1321, 1991. [Online]. Available: https://doi.org/10.1002/spe.4380211204

[54] "Instrumentation Options," in *Using the GNU Compiler Collection (GCC)*. [Online]. Available: https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html

## See Also

- [Build-System](Build-System.md) — full preset list, CI wiring
- [Architecture](../Architecture.md) — preset summary table
- [LSTM-Performance](LSTM-Performance.md) — example of a measured before/after optimization writeup
