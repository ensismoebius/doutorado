# Round 4: what changed since your rulings (2026-11-05)

Paths are relative to this snapshot. Nothing has been deployed. Nothing below is a result.

## Satisfied, code + tests
- D1-D4, s recalibrated (kept 0.10 by replay; see Meeting01Stopping.hpp), checkpoints stamped with
  window values (checkpoint_hash), format bumped to 11. Real headers checked: FSDD 3000 and AudioMNIST
  30000 files, data at byte 44, 16-bit mono 8 kHz; EEGMMIDB 1490 at 160 Hz + 36 at 128 Hz; Siena 512 Hz.
- D7: detail::rescore_front in Meeting01GaSearch.hpp, winner_seeds fresh seeds, search score left out;
  test RescoreAveragesFreshSeedsOnlyNeverTheSearchScore.
- D9, V_th: NOT clamped; removed from the clamp in the text, stated as unbounded, lif_v_th_min logged.
  Why: project rule against silently clamping. Asymmetry: decision in LifBPTT.hpp compute_grad_step:
  kept on purpose (membrane factor multiplies through time, V_th term is a single accumulated term and
  its removal biased V_th, pinned by layers_spiking_basic_gtest); "Exact" comment gone; Eq.(2) has (1-s).
- Logging: rows gain energy_nj_encoder, hidden_silent_share, hidden_always_share, lif_v_th_mean/min,
  lif_beta_mean, train_mse (Meeting01RunMetrics.hpp, count_snn_ops, summarize_lif_parameters).
- D5: pca_energy_nj in meeting01_results.py (158 nJ k=16, 903 nJ k=64, hand-worked test), PCA row and
  ratio in the energy table, PCA in the verdict and the axis rule.
- D6: group_level in 04_ is primary: mean d_g, Student t interval, one-sample t p, Holm, each d_g
  printed and drawn (fig_forest). Recording level descriptive.
- Fold 1 of FSDD/EEGMMIDB out of the headline (DESIGN_INFORMED_FOLDS), headline folds 2-5, all folds as
  sensitivity. Fold 5 disclosed. Development choices listed.
- Margin Delta = 0.02 (EQUIVALENCE_MARGIN), mean-based, interval must lie inside; wide interval centred
  on 0 flagged as not equivalent (test). Provisional: the paper's author has not confirmed it yet.
- N10 window draw: kept fixed per fold, disclosed as limitation (x); Delta sized to it.
- Paper text: paper.tex rewritten per P/Q with your corrections. Saturation claim gone; "matched" gone;
  title changed; Eq.(2); V_th; M7 row; m22, m24, m27, m28; 64-bit disclosure + halved-memory column;
  N7 wording; limitation (ix), (x); D1 disclosure in the design section.
- Figures: m8 replaces Fig. 1 (fig_matched.tex), F2 (fig_folds.tex), F7 (fig_forest.tex, data path done).

## Removed rather than regenerated, for now
- Table III and T1: their numbers carried the defects, so they left the paper; data/paper_design.tex is
  dashes, tOneVerdict is "[T1 has not run yet]". Design findings are now stated as questions.

## Cannot be satisfied before running (be explicit if you consider any of them a blocker for the
## PRE-VALIDATION, as opposed to the grid or submission)
- Headline numbers, T1 rerun, logged twin (>=4 seeds crossed with decoder and pooling, Poisson winner).
- F1 right panel, F3 with real winners, F5, F6: need those runs. Table VIII gives per-side shares meanwhile.
- AudioMNIST/Siena: raw files read on the cluster (speech-like, first window quiet; Siena Fp1 in uV,
  sd 10-40). Windows through the real loader are not yet seen; planned from a short job's split dumps.

## Plan after your verdict
Commit on a branch, deploy, rerun the pre-validation (format 11, ~13 h), T1, logged twin. Main grid only
after that and only with the user's confirmation.
