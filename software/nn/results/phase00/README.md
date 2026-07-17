# Phase 00 results — intentionally empty (re-run pending)

All Phase 00 results were removed on 2026-07-16. They are **not** lost work: they were
invalidated by three fixes landed the same day, and a single coherent re-run under the
corrected code will replace them.

Why they were invalid:

- **D3** — `snn_lr_scale` was applied uniformly to every parameter instead of only the SNN
  biophysical scalars, so the 24 autoencoder profiles trained every weight at 1e-4 while
  declaring 1e-3. Their stored numbers were produced at an effective lr the profiles never
  declared.
- **D1** — the 18 `snn-ae` profiles now declare `firing_rate_reg_lambda=0.5`; the stored
  results predate that field and were produced with it off.
- **D6** — the 92 EEG bark/mel profiles were retired, so their 276 result files had no
  profile left to generate them.

Also fixed the same day and affecting any trained model: Adam applied its decoupled weight
decay *after* the gradient step rather than against θ_{t-1} (found by ground-truth parity
against `torch.optim.AdamW`), and the optimizer is now selectable per profile.

Note the handcrafted results (184 profiles) were **not** numerically stale — handcrafted
extraction trains nothing, so a re-run reproduces them bit-for-bit. They were cleared anyway
so the whole phase is regenerated as one self-consistent set whose summaries carry the new
`training` block (optimizer + resolved learning rate + its source), which the old summaries
lack entirely — the provenance gap that made D3 possible.

The thesis tables under `documentation/00-thesis/monography/tables/` were regenerated from
the pre-deletion data **before** it was removed, so the thesis still compiles and its
handcrafted numbers are valid. The autoencoder rows there are provisional pending the re-run.

The CSVs are recoverable from git history (they were tracked); the `*_summary.json` files
were gitignored and are not.

To regenerate (expensive — see the guard in CLAUDE.md; poisson/latency at T=16 exceeded 2h
per run in testing):

    ./scripts/pipeline/run_e05_profiles.sh phase00
