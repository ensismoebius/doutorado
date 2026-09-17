# Memory Diagnostics — Leak vs. Bounded High-Water-Mark

How to tell whether a process eating RAM is actively leaking or has simply
plateaued at a large-but-bounded working set. Written up after the 2026-07-14
`thesis` memory investigation (case study below).

## Quick triage

1. `free -h` — is swap climbing over time, is `MemAvailable` falling?
2. `ps aux --sort=-%mem | head` — top consumers by RSS. **Caveat**: plain RSS
   undercounts swapped-out and some pinned/shared memory, so the top-RSS
   process isn't always the real hog.
3. `ps -eo pid,ppid,pgid,cmd` — trace parent/launcher. Group by PGID before
   concluding one binary is broken — a launcher/sweep script fanning out
   several heavy children looks like several unrelated leaks otherwise.

## Is it actively growing, or already plateaued?

Sample `/proc/<pid>/status` twice, 60–90s apart:

```bash
awk '/VmRSS|VmSwap/' /proc/<pid>/status
```

- Flat between samples (within noise) → plateaued high-water mark, not a live
  leak.
- Monotonically climbing across multiple samples → likely a real leak — keep
  sampling and look for what's still allocating (containers that grow every
  iteration and are never cleared, per-iteration objects that outlive their
  intended scope).

Don't rely on a single snapshot. A process caught mid-ramp-up looks identical
to a genuinely leaking one; only a second sample distinguishes them.

## Before blaming the binary: check the launcher

Several heavy processes that each plateau at a "reasonable" size can still
exhaust system RAM if a wrapper script fans out more concurrent jobs than the
box can hold. Trace `ps -o pid,ppid,pgid,cmd` up to the parent; if it's a
sweep/launcher script (e.g. `run_thesis_profiles.sh`), check its concurrency-vs-memory
math before assuming the child binary leaks.

## Case study (2026-07-14)

Symptom: 16/17GB RAM used, 14/25GB swap used, system sluggish.

1. `ps aux --sort=-%mem` initially looked dominated by VS Code (many
   renderer/utility processes, ~1.8GB combined) — a red herring; plain RSS
   undercounted the real hog.
2. Scanned `/proc/*/status` for `VmSwap` directly → found four `thesis`
   processes carrying 2–4.4GB of swap each, dwarfing everything else.
3. Sampled `VmRSS+VmSwap` for those four PIDs 90s apart → totals flat to
   within a few KB. **Not** an active leak.
4. `ps -o pid,ppid,pgid,cmd` traced all four back to one launcher:
   `scripts/testing/run_thesis_profiles.sh phase00` (shared PGID). Each of the 4
   profiles (`base_voice`, `base_eeg`, `small_voice`, `small_eeg`) was an
   independent job.
5. Root cause: the launcher's job-sizing math assumed 2048MB/job
   (`THESIS_JOB_MEM_MB` default). Real `snn-ae`/poisson voice profiles peak at
   ~4.4GB, EEG at ~2.1GB. The old default let 4 heavy jobs launch when RAM
   held roughly 2.
6. Outcome: while investigating, the sweep exited on its own (no OOM-kill
   trace found in `dmesg`/`journalctl` — exact cause unconfirmed) after
   18/300 phase00 profiles had checkpointed successfully. The 4 in-progress
   profiles were lost — resumable, since `run_thesis_profiles.sh` checkpoints per
   completed profile, not per epoch — and were skip-resumed automatically on
   relaunch. See
   [Running Experiment05 Profiles § Crash / power-loss recovery](./Running-Thesis-Profiles.md#crash--power-loss-recovery).

## Related

- [Running Experiment05 Profiles](./Running-Thesis-Profiles.md) —
  memory-gated parallelism (`THESIS_JOB_MEM_MB` / `THESIS_JOBS`)
