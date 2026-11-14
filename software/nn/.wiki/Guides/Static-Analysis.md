# Static Analysis

## Why it exists, and why it is kept at zero

A compiler accepts code that is legal but wrong. Two examples taken from this code base, both
found by the analysers on 2026-11-13:

- `nn::Tensor t(2, 2); t << 1, 2, 3;` gave three values for four cells. The comma initializer
  dropped the problem without a word, and the fourth cell kept whatever was in memory.
  clang-tidy's division-by-zero path led to it; the initializer now throws.
- `countShardRows` built a shell command with the file path pasted into it, ran Python to read a
  JSON file, and counted a shard without a `count` field as 0 rows. clang-tidy flagged the
  command processor; the function now parses the JSON in C++ and refuses a missing count.

A warning list that is never empty stops being read: the 301st warning looks like the other 300.
So the project keeps **every analyser at zero findings**, and CI fails on the first new one.

| Tool | Finds | Run locally (from `software/nn`) | In CI |
|---|---|---|---|
| clang-tidy | undefined behaviour, wrong arithmetic, lost exceptions, needless copies | `run-clang-tidy -p out/build/max-performance 'software/nn/src/'` | yes, fails on any finding |
| cppcheck | uninitialised members, wrong conditions, dead stores, portability | `cmake --build --preset=max-performance --target analysis-cppcheck` | yes, fails on any finding |
| compiler (`-Wall`) | the usual warnings | the normal build | build log: keep it at 0 warnings |
| flawfinder | risky C functions (security) | `--target analysis-flawfinder` (HTML report) | no |

`--target analysis-all` runs cppcheck, flawfinder and clang-tidy, and fails when one of them is
not installed (it used to skip the missing one and pass).

## clang-tidy: which checks, and why not all of them

The configuration is [`.clang-tidy`](../../.clang-tidy). It enables the families that find
**defects**:

| Enabled | Example of what it caught here |
|---|---|
| `clang-analyzer-*` | a division whose divisor can be 0 on some path |
| `bugprone-*` | `static_cast<size_t>(a * b)` with `int a, b`: the product overflows **before** the cast widens it |
| `performance-*` | a `std::function` or `shared_ptr` copied on every call |
| `modernize-use-override`, `modernize-use-nullptr`, `cppcoreguidelines-slicing`, `cppcoreguidelines-virtual-class-destructor`, `readability-misleading-indentation`, `readability-duplicate-include` | single high-value checks from otherwise style-only families |

The style-only families (`readability-*`, most of `modernize-*` and `cppcoreguidelines-*`) are
**off**. Measured on this code they report 300–900 warnings per file: braces on every `if`,
trailing return types, upper-case literal suffixes, `.at()` instead of `[]` in numeric loops.
None of those is a defect, and the last one would slow the hot loops.

Three checks are disabled on purpose. The reason is written next to each in the file:

| Disabled | Why |
|---|---|
| `bugprone-easily-swappable-parameters` | numeric APIs take several floats or ints by design |
| `bugprone-random-generator-seed` | it flags constant seeds; here every run **must** be reproducible, so constant seeds are required |
| `performance-enum-size` | saving 3 bytes per enum buys nothing and changes the underlying type of enums written to files |

> **The trap that hid all of this (loud vs silent: silent).** Until 2026-11-13 the file listed
> its checks in a YAML folded block with `#` comment lines **inside** it. YAML keeps those lines
> as text, so every glob after a comment became `# … bugprone-*`, which matches no check. Only
> `clang-analyzer-*` (the one glob not preceded by a comment) ever ran, and CI was green with
> 131 of the intended ~400 checks. Keep comments outside the `Checks` string, and check what
> actually runs with
> `clang-tidy --list-checks -p out/build/max-performance src/core/utility/progress.cpp`.

## cppcheck: analysed through the compile database

The `analysis-cppcheck` target (`cmake/DevAndAnalysisTargets.cmake`) runs
`cppcheck --project=out/build/max-performance/compile_commands.json`. Each translation unit is
then analysed with its real include paths and definitions. Build-tree and vendored (`_deps`)
headers are suppressed. The report goes to `out/build/max-performance/cppcheck-report.xml`, and
any finding makes the target fail.

Two cppcheck checks are switched off for the whole code base, on purpose. They are style
opinions, not defects, and they have the same status as a check left out of `.clang-tidy`:

| Check | What it says | Why it is off |
|---|---|---|
| `useStlAlgorithm` | "this raw loop could be `std::transform` / `std::accumulate`" | A preference. It named 138 loops, which read more clearly as loops. |
| `shadowFunction` | "a local variable has the same name as a member function" (`size`, `rows`) | Legal and unambiguous where it is used; renaming 52 locals fixes nothing. |

Before, the target handed cppcheck a bare list of every file under `src/`, headers included. A
header analysed on its own, without the file that uses it, made cppcheck report about a hundred
struct members as "unused". Those false positives buried the real findings.

### CI runs other versions of the analysers

CI runs on `ubuntu-latest` and installs clang-tidy and cppcheck with `apt`, so their versions are
whatever that image ships, and they move when GitHub moves the image. On 2026-11-14 that was
Ubuntu 26.04: clang-tidy 21 and cppcheck 2.19, against 23 and 2.22 on a local Arch install. The
versions disagree: a check one version reports, another may not. CI's log names the image under
"Operating System" and the versions in its "Install dependencies" step; CI also prints the
cppcheck findings when that step fails.

| | Local (Arch) | CI (`ubuntu-latest`, 2026-11-14) |
|---|---|---|
| clang-tidy | 23 | 21 |
| cppcheck | 2.22 | 2.19 |

To reproduce CI before pushing, run its image in a container that mounts the repository at the
same absolute path (the compile database stores absolute paths). Use the image CI's log names:

```bash
B=$PWD/out/build/max-performance   # from software/nn, after a build
docker run --rm -v "$HOME/Repos/doutorado:$HOME/Repos/doutorado" -w "$PWD" ubuntu:26.04 bash -c \
  "apt-get update -qq && apt-get install -y -qq cppcheck clang clang-tidy clang-tools \
     build-essential libopenblas-dev libomp-dev libfftw3-dev libhdf5-dev libeigen3-dev \
     libsqlite3-dev zlib1g-dev python3-dev python3-numpy \
   && run-clang-tidy -quiet -p out/build/max-performance 'software/nn/src/' \
   && cd $B && cppcheck \
   --project=$B/compile_commands.json --enable=warning,style,performance,portability \
   --inline-suppr --library=googletest --suppress=missingIncludeSystem \
   --suppress=unmatchedSuppression --suppress=useStlAlgorithm --suppress=shadowFunction \
   '--suppress=*:*/_deps/*' '--suppress=*:$B/*' -i$B --std=c++20 -j \$(nproc) --error-exitcode=1"
```

**Trap — reproduce with the wrong image and you fix the wrong findings.** On 2026-11-13 the
reproduction used `ubuntu:24.04` (cppcheck 2.13, clang-tidy 18) while CI had moved to 26.04; the
fixes were sound but did not match what CI reported. Read the image from CI's log first.

`--suppress=unmatchedSuppression` is what lets one inline suppression serve several versions: a
suppression only one version needs is not an error under the others.

### An optional in a test: `if (!x) GTEST_FAIL()`, not `ASSERT_TRUE(x.has_value())`

CI's clang-tidy (18) cannot see through gtest's `ASSERT_TRUE` macro. After
`ASSERT_TRUE(info.has_value());` it reports every later `info->digit` as
`bugprone-unchecked-optional-access`, although the test has already returned when the value is
missing. clang-tidy 20 does the same; 23 (local) does not, so the findings appear only in CI.
Write the check as a plain `if`, which every version follows:

```cpp
const auto info = FsddLoader::parse_filename("0_jackson_0");
if (!info.has_value())
{
    GTEST_FAIL() << "info is empty"; // fatal failure + return, exactly what ASSERT_TRUE does
}
EXPECT_EQ(info->digit, 0);
```

## When a finding is wrong

First check that it really is wrong: most findings that look wrong are real, as with the two
examples at the top. If it is wrong, suppress **that line only**, in the code, with the reason:

```cpp
// try_push moves from its argument only when it returns true; every push failed, so
// `prefetched` still holds the batch.
if (prefetched.batch.inputs.size() > 0 || // NOLINT(bugprone-use-after-move)
```

```cpp
// cppcheck-suppress knownConditionTrueFalse   (reason here)
```

Never disable a check globally to make one line pass. A new `--suppress=` in the cppcheck
target is a decision about the whole code base: it goes in the table above, with its reason. `NOLINT` must sit on the line clang-tidy reports. clang-format can move a
trailing comment onto another line, so for a statement that spans several lines use
`// NOLINTNEXTLINE(<check>)` on the line above it.

## See also

- [Build System](./Build-System.md): presets, `NN_ENABLE_CLANG_TIDY` (clang-tidy during compilation)
- [Test Quality and Determinism](./Test-Quality-and-Determinism.md)
