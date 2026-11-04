# Mathematical review — Reviewer 2

**Manuscript:** *Spiking autoencoders cost more than they save* (working title), `documentation/07-articlesProduced/meeting01/paper.tex`.
**Reviewer role:** adversarial mathematical referee. I checked definitions, derivations, code, and the arithmetic of every table/claim I could reach.
**Independence:** I did not read `dossier/review_visual.md` or `dossier/visual_mockups/`. I used `dossier/author_dossier.md` only to locate claims and code, and re-derived every number below. Where I agree with the author's own status, I say so; where I do not, I say why.

**Status of the manuscript (read this first).** `data/paper_config.tex` sets `\cfgDataStatus{placeholder}`. The main grid has not run: `paper_loso_overview.csv` has rows Mean/PCA/SNN-AE/ANN-AE but every value is `--`; `paper_loso_energy.tex` is all `--`; `paper_loso_energy_verdict.tex` is `[the energy comparison has not run yet]`; `paper_prevalidation.tex` is all `--`; the `paper_loso_*_pareto.csv` files are 78 bytes. The only real results file is `data/paper_t1.tex` (`\tOneDataStatus{results}`), a side experiment on Transformer warm-up/pre-norm. Consequently the paper's *central empirical claim* — that at matched reconstruction quality the SNN-AE uses more energy, and that the excess is not removable by training choices — is not evaluated anywhere in the manuscript. This is the first and largest objection.

Notation used below: $V$ is the pre-reset membrane potential, $s=\Theta(V-V_{th})$ the spike, $\beta=e^{-\Delta t/(RC)}$, $V_{reset}=0$ (hard reset), $\hat\sigma'$ the surrogate derivative, $E$ energy per window, $L$ reconstruction loss.

---

## FATAL objections

### F1 — The headline result does not exist in the submission

**Severity:** FATAL
**Claims:** Abstract (all "we show" sentences), Contributions 1–4, Tables `tab:loso-overview`, `tab:loso-significance`, `tab:loso-energy`, `tab:snn-selection`, `tab:prevalidation`, Conclusion.
**Statement.** The paper asserts a measured quality–energy separation between SNN-AE and ANN-AE and a set of secondary results (significance, energy verdict, pre-validation). None of the corresponding numbers is present. `\cfgDataStatus{placeholder}` and `--` are not a presentation issue: they mean the estimand $\Delta=E[L_{SNN}]-E[L_{ANN}]$ and the energy ratio have not been estimated.
**Why FATAL.** A conference paper cannot claim "we show" a result that is absent. The one real table (T1) concerns Transformers, not the proposed comparison; it cannot carry the abstract, the contributions, or the conclusion. The design study is a pilot on two datasets with three seeds, not the main grid.
**What would satisfy.** Run the grid (or the reduced headline) and fill the tables with means, dispersion, $n$, and the exact split; or remove the empirical claims and submit the design study as a methods/negative-result paper with an explicit "preliminary" status. The current mixture — real T1 numbers beside placeholder headline tables — is the worst option because it invites the reader to assume the headline was run.
**Author's status.** AGREE (dossier line 5 admits the placeholders and that the design table predates formats 7/8/9).

### F2 — The "saturation" mechanism is contradicted by the paper's own design table, and the LIF transfer is a staircase, not a clip

**Severity:** FATAL (for the mechanism claim; the quality–energy result, if it existed, could survive with a reworded mechanism)
**Claims:** Abstract ¶3: "the cost is the saturation of the neuron's transfer function, not the spike"; design study (iii); Conclusion; Contribution 4.
**Formal statement.** Let $H_{sat}$ be "a bounded saturating transfer function causes the excess error". The paper's own rows:
- sigmoid (bounded, smooth, saturating): FSDD 0.685, EEG 0.308;
- leaky-ReLU control: FSDD 0.667, EEG 0.489;
- clamp $[0,1]$ (bounded, hard, not smoothly saturating): FSDD 0.743, EEG 0.547;
- SNN (per-step): FSDD 0.780, EEG 0.592; SNN (latent mean, T=4): FSDD 0.724, EEG 0.484.

If $H_{sat}$ were true, sigmoid would be as bad as the SNN. It is not: on FSDD it is 0.095 below the per-step SNN and 0.039 below the latent-mean SNN; on EEG it is 0.284 below the per-step SNN. If "hard clipping" were the cause, clamp should be as bad as the SNN; it is (0.743 vs 0.780 FSDD; 0.547 vs 0.592 EEG). The common factor is the **hard decision boundary**, not saturation. The paper's own sentence "what remains is reproduced by a non-spiking network with a hard-clipped activation" supports hard clipping; the abstract then misnames it "saturation".
**Derivation of the true LIF transfer.** For constant current $I$, $V_0=0$, hard reset to $0$:
$$
V_n=\beta V_{n-1}+I,\qquad s_n=\Theta(V_n-V_{th}),\qquad V_n\leftarrow 0\ \text{if }s_n=1 .
$$
Between resets $V_n=I(1-\beta^n)/(1-\beta)$. The inter-spike period is
$$
k(I)=\min\{n\ge 1:\ I(1-\beta^n)/(1-\beta)>V_{th}\}.
$$
Hence: no spike for $I\le V_{th}(1-\beta)$; a staircase with $k\ge 2$ for $V_{th}(1-\beta)<I<V_{th}$; $k=2$ at $I=V_{th}$ (not $1$); $k=1$ for $I>V_{th}$. Numerically (Vth=1):
- $\beta=0.9$: $I=0.5\mapsto k=3$, $0.9\mapsto2$, $0.99\mapsto2$, $1.0\mapsto2$, $1.001\mapsto1$;
- $\beta=0.99$: $I=0.5\mapsto3$, $0.9\mapsto2$, $0.999\mapsto2$, $1.001\mapsto1$.

So the neuron is a **quantized phase/frequency encoder** (a staircase), not a clipped analog function. The paper's "never fires for $I<V_{th}(1-\beta)$, fires every step for $I\ge V_{th}$" is wrong at $I=V_{th}$ and omits the whole staircase; "a clipped transfer function" is a different object. The correct mechanism candidates are threshold quantization, reset, and the resulting gradient sparsity — not saturation.
**Counterexample to the mechanism in one line.** A sigmoid has the same bounded range and a smooth saturation and is near the control; a hard clamp has no smooth saturation and is near the SNN. Therefore saturation is neither necessary nor sufficient for the gap.
**What would satisfy.** (a) Rename the mechanism to "hard threshold/quantization" and re-derive the consequences; (b) add an experiment that separates smooth saturation from hard clipping (e.g. sigmoid with temperature, hard sigmoid, straight-through); (c) show the claimed saturating-but-smooth activation reproduces the SNN gap — it does not in the current table.
**Author's status.** DISAGREE with the paper's wording. The dossier may mark the design-study mechanism as PARTIAL; the arithmetic above is decisive: the paper's own sigmoid row falsifies its abstract.

### F3 — The "matched" control is not matched on encoding, time, or transform

**Severity:** FATAL (for the causal attribution "the cost of the spike")
**Claims:** Contribution 1 ("only the encoder's hidden neuron differs"), Abstract, Fig. 1 caption ("red: parts that exist only in the spiking network"), Sec. 4.2, Conclusion.
**Formal statement.** The estimand is
$$
\Delta=E[L_{SNN}]-E[L_{ANN}].
$$
For $\Delta$ to be the causal effect of "spiking", all other factors must be held fixed. The code does not:
- `run_ann_twin` forces `lstm_frame_size=window_size`, `recurrent_time_steps=1`, `ProtocolAutoencoder(make_ann_twin_cfg(config, widths))`, `encoding="direct"`, `val_mse=NaN`, hyperparameters "selection none/copied_from".
- The SNN genome includes `encoding` (direct/Poisson/latency), `time_steps` $T\in[2,32]$, and `input_transform` (dense/conv1d/recurrent). The ANN always gets direct, $T=1$, no transform.
- The SNN's latent is averaged over $T$; the ANN's is a single frame. The SNN's encoder is evaluated $T$ times; the ANN's once. The SNN's decoder is LReLU; the ANN's is LReLU, but on a different latent.

Thus
$$
\Delta=\tau_{spike}+\tau_{encoding}+\tau_{T}+\tau_{transform}+\tau_{latent\ average}+\text{interactions},
$$
and the paper's "only the encoder's hidden neuron differs" is false. Fig. 1's red parts are not the only difference. The design study varies some factors (encoding, $T$, loss, decoder) but not in the matched control; a reader cannot attribute the gap to the LIF.
**Why FATAL.** The headline sentence "the cost is the spike" is an attribution to one factor in a bundle. Even if the ANN wins, the paper cannot say whether the loss comes from Poisson/latency coding, from $T>1$, from the recurrent transform, or from the LIF. This is not a wording nit: it is the paper's causal claim.
**What would satisfy.** A factorial control: same encoding, same $T$, same transform, same latent averaging, with only the hidden activation swapped (LIF vs LReLU); or a decomposition that varies one factor at a time and reports interactions. At minimum, replace "the spike" by "the spiking pipeline" and delete "only".
**Author's status.** DISAGREE with the paper. The code evidence is decisive; the paper's own Sec. 4.2 sentence "applied once to the analog window" concedes the $T=1$/direct difference, but the abstract and Fig. 1 do not.

---

## MAJOR objections

### M4 — Eq. (2) omits the reset term; the paper, the code, and the exact derivative are three different objects

**Severity:** MAJOR
**Claims:** Eq. (2), method "surrogate gradient".
**Statement.** With $v_{post}=v_{pre}+s(v_{pre})(V_{reset}-v_{pre})$, $s=\Theta(v_{pre}-V_{th})$,
$$
\frac{\partial v_{post}}{\partial v_{pre}}=(1-s)-(v_{pre}-V_{reset})\,\hat\sigma'(v_{pre}-V_{th}),
\qquad
\frac{\partial v_{post}}{\partial V_{th}}=(v_{pre}-V_{reset})\,\hat\sigma'(v_{pre}-V_{th}).
$$
The code's hard-reset branch (`compute_grad_step`) uses
$$
g_{v_{pre}}=g_{out}\hat\sigma'+g_{next}\beta(1-s),
$$
which keeps $(1-s)$ but drops $-(v_{pre}-V_{reset})\hat\sigma'$. The paper's Eq. (2) has neither $(1-s)$ nor the reset-gate term: it is the no-reset recurrence
$$
\delta_l[t]=\hat\sigma'(V_l[t])\odot(W^\top\delta_{l+1}[t])+\beta\,\delta_l[t+1].
$$
So paper $\ne$ code $\ne$ exact. The code's comment "Exact local derivative for $v_{post}=v_{pre}+s(V_{reset}-v_{pre})$" is false.
**Numeric check** ($\beta=1$, $\hat\sigma'=0.5$, $V_{reset}=0$, $V_{th}=1$): at $v=1.2$ (spike, $s=1$): paper 0.5, code 0.5, exact $-0.1$; at $v=0.9$ (no spike, $s=0$): paper 0.5, code 1.5, exact 1.05. The missing term is $-0.60$ and $-0.45$, respectively. This is not a near-threshold correction; it changes sign for $v_{pre}>V_{reset}$.
**Why MAJOR.** The paper's gradient is not the gradient of its stated model; a reader cannot reproduce training from Eq. (2). The code's approximation is also not exact, and the inconsistency is invisible in the paper.
**What would satisfy.** State the exact derivative, state which approximation is implemented and why, and report a finite-difference gradient check on the hard-reset LIF. If the code is the reference, Eq. (2) must include $(1-s)$ and the reset term; if Eq. (2) is the reference, the code and experiments must change.
**Author's status.** DISAGREE with the paper; the code comment is also wrong.

### M5 — 32-bit words are priced at 64-bit SRAM energies; the bias favours the paper's conclusion

**Severity:** MAJOR
**Claims:** Sec. 5, Table 1, Contribution 3.
**Statement.** The paper says "one access moves one 32-bit word" and prices memory with Horowitz's 64-bit cache-access energies: 10 pJ at 8 KB, 20 pJ at 32 KB, 100 pJ at 1 MB. `Meeting01Energy.hpp` admits: "one access moves one 32-bit word ... at the 64-bit figure." The 64-bit figure is not automatically the price of a 32-bit word: a wider access can cost more, but the paper must justify applying a 64-bit access energy to a 32-bit word, or report the 32-bit sensitivity. A natural sensitivity halves the memory energy (the author's own `energy.py` does this). The overcharge scales with the number of accesses; the SNN has more accesses per MAC than the ANN (spike FIFO, potential read/write, threshold/reset). Numerically, ANN $[59,16]$: $E=854.4853$ nJ at 64-bit, $501.55$ nJ at 32-bit (ratio $1.704$); memory is $82.6\%$ of $E$. Correcting the word width lowers the SNN/ANN ratio, i.e. the 64-bit price *inflates the SNN more* and supports "SNN costs more". The paper does not disclose this or run a 32-bit sensitivity.
**Why MAJOR.** This is a directional bias in the headline energy comparison, not rounding. The "three memory sizes" sensitivity does not fix it: all three use the same 64-bit price. A reader who takes "32-bit word" literally will not reproduce the numbers.
**What would satisfy.** Price 32-bit words at 32-bit energies (or use 64-bit words consistently and say so); report both; show the SNN/ANN ratio under both. If the ratio is robust, the conclusion survives; if not, weaken the headline.
**Author's status.** The dossier's C014 says the model is SUPPORTED; I DISAGREE on this point. C144 ("extended, not unchanged") is relevant but does not cure the word-width error.

### M6 — "Where the model leaves a choice, we take the one that works against the conclusion" is false for the first-layer choice

**Severity:** MAJOR
**Claims:** Sec. 5.
**Statement.** Read "the conclusion it could support" as the SNN-cheaper conclusion that the energy model could support. The sentence says every modeling choice is made against that conclusion, i.e. in favour of "SNN costs more". Most choices are: the 8 KB floor, the threshold add per step, and the 64-bit price all raise $E_{SNN}$ relative to $E_{ANN}$ (M5). But the **first-layer-once** choice is the exception: it lowers $E_{SNN}$ by $(T-1)M w_1$ MACs, $(T-1)(M+1)w_1$ weight reads, $(T-1)M$ input reads/adds and integer adds, and costs only $w_1$ writes and $T w_1$ reads. It therefore favours the SNN, i.e. supports the SNN-cheaper conclusion. The code header says it moved FSDD fold 0, $T=28$, from $14.4\times$ to $1.53\times$. So the blanket sentence is false: at least one choice goes the other way. The 64-bit price is also a modeling error, not a "conservative choice" (M5), so the sentence conflates an error with a choice.
**Why it matters.** The net direction of the modeling choices is not established by the sentence; the first-layer-once choice is a real, large, anti-conservative effect for the paper's conclusion, and it is not disclosed in Table 1 or in the energy verdict. The paper cannot claim "conservative" without quantifying it.
**What would satisfy.** A table of modeling choices $\times$ direction $\times$ magnitude from the code, and a sensitivity analysis that flips each choice (including first-layer-once). Report both primary and as-executed $E$.
**Author's status.** DISAGREE with the blanket claim. The header itself admits first-layer-once favours the SNN; the sentence should be removed or qualified.

**Withdrawn objection (former M11).** I initially suspected that selecting the lowest validation MSE on the first Pareto front could exclude a lower-MSE genome. On checking `pick_winner`, the global minimum validation MSE is on the first front (a genome with lower MSE cannot be dominated), so the paper's "lowest re-scored validation MSE (ties: lower cost)" is consistent with the code. I withdraw that objection and note it here so the authors do not spend time on it.

### M7 — First-layer-once is only a caption note; the $T$ read-backs are missing

**Severity:** MAJOR
**Claims:** Contribution 3, Sec. 5, Table 1.
**Statement.** The code path `add_dense_fc(e, rows/repeated_steps, n_in, n_out, writes_output=true); e.access(e.rd_in, rows*n_out, n_out)` implements first-layer-once. Table 1's caption does mention it ("1 for the first layer of a direct-coded input"), so the objection is not that it is absent from the caption. The objection is that the row counts at $R=1$ charge only $n_{in}$ input reads, while the code also charges $T n_{out}$ read-backs of the stored current (`e.access(e.rd_in, rows*n_out, n_out)`), and the table has no row for those reads. The table therefore cannot reproduce $E$ for the first-layer-once configuration, and the choice's effect (which is anti-conservative for "SNN costs more") is invisible in the table. The code also keeps `energy_nj_as_executed` as a check; the paper reports only one number.
**What would satisfy.** Add the $T n_{out}$ read-back row (or a separate "first layer once" line), state $T$ and the flag per experiment, and report both primary and as-executed $E$ (or prove they coincide for the chosen configs).

### M8 — Table 1 omits the analog-mean, encoder, and nonlinearity costs

**Severity:** MAJOR
**Claims:** Table 1, Contribution 3.
**Statement.** Most cells match the code; the omissions that matter for the adopted model are:
- **Spike add (bias, synapse):** table `(\theta+R)n_{out}` matches `events + rows*n_out` because the code defines `events = spikes_in * n_out`; not a defect.
- **TimeMeanPool:** the note "one add per input spike, one multiply per output" is the `in_is_spikes` branch. For the analog latent (the paper's model), the code sets `summed = in->size() = T d`, charges $T d$ adds and $T d$ input reads, plus $d$ writes; the note omits the analog branch, so the table under-describes the SNN's cost.
- **Encoder:** the spike encoder's RNG/latency cost (Poisson sampling, latency coding) is not in Table 1; the paper says "spike encoder" but has no row.
- **Nonlinearity:** leaky ReLU compare/multiply is not in Table 1; `add_nonlinearity` charges $v$ adds and (for LReLU/Tanh) $v$ muls.
- **First-layer read-backs:** see M7: the caption says $R=1$ for the first layer, but the $T n_{out}$ read-backs of the stored current are not a row.
**Why MAJOR.** $E$ is not reproducible from Table 1 for the adopted analog-latent model. The missing analog-mean and encoder rows are undercharges to the SNN, so the table cannot support the "conservative" claim; the direction is not uniformly one way, but the omissions are not negligible.
**What would satisfy.** A table generated from the code, one row per term, and a worked example for one config ($M$, widths, $T$, spiking/analog) that reproduces $E$ exactly.
### M9 — The winner's-curse re-score includes the search seed

**Severity:** MAJOR
**Claims:** Search ("re-scored over 3 seeds"), `\cfgGaWinnerSeeds{3}`.
**Statement.** The code:
```
mse_sum = ind.val_mse;                       // search seed
for extra in 1..winner_seeds-1:
    evaluate(ind.genome, base_seed + extra*7919)
    mse_sum += probe.val_mse
ind.val_mse = mse_sum / winner_seeds
```
The first term is the search seed's validation MSE — the same noisy value that made the genome win. The "mean over 3 seeds" is $1/3$ lucky draw $+$ $2/3$ fresh, not a fair 3-seed mean. The code's own comment says the search "scores one seed per genome ... the best of a noisy population is partly selected on seed luck — the winner's curse"; the fix removes only part.
**Quantitative check.** Simulate 48 candidates with true quality $\mu\sim N(0,1)$, $val=\mu+\epsilon$, $test=\mu+\epsilon'$, select $\arg\min val$: $\mathbb E[\mu\mid sel]=-1.579$ (the winner's true quality is 1.58 sd below average). Re-score with search seed + 2 fresh: $\mathbb E[\mu\mid sel]=-1.118$. Fresh-only 3 seeds: $\mathbb E[\mu\mid sel]\approx 0.00$. The re-score removes about $29\%$ of the selection bias, not all. The reported test MSE is an unbiased estimate of the selected model's true MSE, but the selected model is not the best true model; "re-scored" suggests otherwise.
**What would satisfy.** Re-score on 3 **fresh** seeds (exclude the search seed), or nested CV (search on inner folds, select on outer); report the selection bias. If the search seed must be included, say "1 search + 2 fresh" and do not call it a 3-seed mean.

### M10 — "More search favours the SNN, so a gap in the ANN's favour is conservative" is not a theorem

**Severity:** MAJOR
**Claims:** Sec. 4.2; Contribution 1; Abstract.
**Statement.** The premise is true (SNN searched, ANN not). The inference is not: more search can *hurt* test performance by selecting a genome that overfits the validation block. The control is a single unselected configuration, so the comparison is "selected SNN vs one ANN", not "best SNN vs best ANN". A gap in the ANN's favour could be due to the SNN's search selecting a val-overfit genome, not to spiking; a gap in the SNN's favour could be due to search. The direction of the bias is not identified without the ANN's own search distribution.
**Counterexample.** In the simulation above, the selected model's true quality is $-1.58$ sd below the average candidate; a second family with no search has true quality $0$. The searched family loses by $1.58$ sd even though its average candidate is equal. "More search favours it" is false in true-quality terms; it favours the *validation* score, not the test score.
**What would satisfy.** Search the ANN control with the same budget (or a budget-matched random search), or report the SNN's test distribution over the front and the ANN's over a random sample of architectures. At minimum, remove "conservative" unless the direction is proven.

### M12 — "Xavier is better or equal" and the Kaiming comparison are selective

**Severity:** MAJOR
**Claims:** Design study (ii); Abstract ("the initializer alone decided the comparison").
**Statement.** The table: SNN latent-mean LReLU $T=4$, FSDD Kaiming 0.724, Xavier 0.729 — Kaiming is better by 0.005, contradicting "Xavier is the better or equal choice for each arm on both datasets". At $T=16$ FSDD Kaiming 0.729, Xavier `--`; EEG $T=16$ Kaiming 0.457, Xavier 0.396 — Xavier better. The text's "SNN looked equal (0.484)" uses EEG $T=4$: Kaiming 0.484, control 0.489. At $T=16$ Kaiming the SNN is 0.457, i.e. **better** than the control 0.489, not equal. The "gap vanished" claim is true only at $T=4$; at $T=16$ the SNN wins. The paper selects the $T$ that supports "initializer decided".
**What would satisfy.** Report all $T$, both initializers, both datasets; state the interaction; do not claim "Xavier better or equal" when one cell reverses. Pre-register the $T$ selection or show the conclusion is invariant.

### M13 — The recording-level primary test is pseudoreplicated

**Severity:** MAJOR
**Claims:** Statistical methods ("recording-level paired statistics"); significance table; "Energy first".
**Statement.** The primary unit is the recording, but recordings are nested in speakers/subjects (6, 60, 109, 14 groups; 5 FSDD speakers on the headline folds). Windows within a recording and recordings within a speaker are dependent. The recording-level Wilcoxon/bootstrap treats recordings as independent, so the null variance is underestimated; the CI is too narrow and the $p$ too small. The valid group-level test (speaker/subject) has $n=5$ for FSDD, and the exact two-sided Wilcoxon minimum $p=2/2^5=0.0625>0.05$, so FSDD can never reject at 0.05 even with a perfect effect. The paper acknowledges this in limitations but still uses the invalid recording-level test as primary and as the "Energy first" trigger.
**What would satisfy.** Make the group-level test primary (or a hierarchical/cluster-robust test); report the recording-level as descriptive; for FSDD, state that the design cannot reach 0.05 and use a pre-specified equivalence/CI margin instead. Do not trigger "Energy first" on a test that cannot reject.

### M14 — "Energy first" is not a valid decision rule

**Severity:** MAJOR
**Claims:** Statistical methods: "Separation requires the CI to exclude zero and the Holm $p$ to be $<0.05$; otherwise the lower-energy model is recommended."
**Statement.** (i) The estimand is the mean difference in reconstruction error; the Wilcoxon tests a median of paired differences (or a rank-based hypothesis), not the mean; a CI for the mean and a $p$ for the median can disagree. (ii) The rule is a hybrid: CI excludes zero AND $p<0.05$; if the CI excludes zero, the $p$ is usually $<0.05$, but not always (different tests). (iii) "Otherwise recommend lower $E$" is a non-inferiority/equivalence claim, but no margin is defined; failing to reject superiority is not evidence of equivalence. (iv) "Fixed in advance" is not pre-registration; the rule can be chosen after the design study.
**What would satisfy.** Define the estimand (mean $\Delta$), the test (paired $t$/permutation on recordings or cluster-robust), the margin for "no meaningful difference", and a pre-registered decision rule; or drop the recommendation and report the CI.

### m15 — T1 is underpowered; the paper's text is descriptive but the counts do not support a remedy

**Severity:** MINOR
**Claims:** T1 verdict; the surrounding paragraph.
**Statement.** The T1 data: FSDD final fits 1/4 warm-up armed, 1/4 pre-norm, 0/4 control; EEG 0/4 warm-up, 1/4 pre-norm, 1/4 control. All 8 trainings per arm include 4 GA evaluations + 4 final fits; the GA evaluations are not the final models. The final-fit means: FSDD 0.9162 (warm-up) vs 0.9350 (control), Welch $p=0.338$; EEG 0.9934 (warm-up) vs 0.9925 (control), $p=0.244$. The FSDD warm-up mean is driven by one seed (0.867 vs 0.933, 0.931, 0.934); the EEG pre-norm mean is driven by one seed (0.792 vs 0.924, 0.993, 0.975). Fisher exact 1/4 vs 0/4 $p=1$; 1/8 vs 0/8 $p=1$. The paper's `\tOneVerdict` is in fact descriptive ("1 of 8", "0 of 8"), so this is a caution rather than a false claim: a reader can still infer a remedy that the data do not support. The actual pre-validation cost claim (Contribution 4) is a separate table and is a placeholder (F1).
**What would satisfy.** Report exact intervals (1/4: $[0.028,0.716]$; 0/4: $[0,0.445]$; 1/8: $[0.014,0.454]$; 0/8: $[0,0.262]$), the per-seed values, and the test $p$; say "did not reliably restore" rather than implying a remedy.

### m16 — "Every model" in Contribution 3 is broader than the energy table

**Severity:** MINOR
**Claims:** Contribution 3; Table energy; reading guide.
**Statement.** Contribution 3 says the model is "applied unchanged to every model"; the energy table as specified has SNN-AE and ANN-AE (and the ratio). PCA and Mean have no $E$, so they cannot be placed on the energy plane, yet the reading guide says "a model that does not beat PCA has not shown that its nonlinearity helps". Clarify whether "every model" means every model with a defined energy, or every trained family; the PCA/Mean references are not in the energy comparison.
**What would satisfy.** Say "every model for which an energy is defined" or "SNN-AE and ANN-AE"; add PCA/Mean if the reading guide depends on them.
### M17 — The search-space depth range is wrong

**Severity:** MAJOR (reproducibility of the search)
**Claims:** Search: "depth $L\in[1,4]$ and widths ..."; `\cfgSnnDepth{1,4}`.
**Statement.** `random_genome` draws depth $\in[1,4]$, but `repair_widths` truncates the hidden widths to `max_layers-1=3` and then pushes `latent_dim`, so `encoder_widths.size()` $\in\{2,3,4\}$. $L=1$ (a single Linear, no LIF) is not representable. The paper's "$L-1$ hidden widths" formula is correct for $L\ge2$, but the range includes an impossible $L$. If $L$ is the number of encoder Linear layers including the latent, the reachable range is 2–4; if $L$ is the number of hidden layers, it is 1–3. Either way, `[1,4]` is wrong by one. The latent is fixed at $d$ and not searched (hidden widths in $(d,128]$, strictly decreasing).
**What would satisfy.** State the reachable range and the definition of $L$; report the actual distribution of depths/widths in the search (no histogram is given).

### M18 — The design study is one-factor-at-a-time, confounded, and underpowered

**Severity:** MAJOR
**Claims:** Design study; Contribution 4.
**Statement.** The table varies loss (per-step vs time-mean), decoder (LIF vs LReLU), $T$ (4 vs 16), initializer (Kaiming vs Xavier), activation (LIF vs sigmoid vs clamp), often with `--` cells; it is not a factorial design, so interactions are confounded. $n=3$ seeds; no dispersion or CI. The "saturation" conclusion is an inference from separate rows, not an identified effect. The per-step effect (FSDD 0.780 vs 0.734; EEG 0.592 vs 0.454) is 0.046/0.138 with 3 seeds; no CI.
**What would satisfy.** A factorial or at least a crossed design with the same cells; report mean$\pm$sd and the interaction; or soften to "consistent with" and drop the mechanism claim.

### M19 — The per-step/spiking-decoder claim is not separable in the table

**Severity:** MAJOR
**Claims:** Abstract ("two avoidable choices enlarged the gap: the loss on every step and the spiking decoder"); design study (i).
**Statement.** The table has per-step loss rows (0.780/0.592), but no "spiking decoder" row; the text says "replacing the spiking decoder by a leaky ReLU decoder (0.724/0.484)" — that is the latent-mean LReLU row, which also changes the loss to time-mean. The two factors (loss, decoder) are changed together, so the 0.780$\to$0.724 and 0.592$\to$0.484 differences are not separable into loss and decoder effects. The abstract says "two ... choices", but the table cannot attribute the gap to each.
**What would satisfy.** A $2\times2$ (loss $\in\{$per-step, time-mean$\}$ $\times$ decoder $\in\{$LIF, LReLU$\}$) with the same $T$/initializer; report the main effects and interaction.

---

## MINOR objections

### m20 — Vth is not clamped, contrary to Eq. (1)

`LifBPTT::compute_backward_params` clamps $R,C$ to `kMinPositiveParam=1e-6F` but uses `voltage_threshold.at(0,0)` unclamped. Eq. (1) says "$R$, $C$, and $V_{th}$ are clamped above $10^{-6}$". If $V_{th}\le0$, the threshold test is degenerate. Fix the text or the code.

### m22 — "48 evaluations" is nominal

Population 8, generations 5, initial population + offspring $=48$; the cache dedups identical genomes, so distinct trainings are fewer. Also, whether the initial population counts as "evaluations" depends on the accounting. Clarify "candidates" vs "distinct trainings".

### m23 — `\cfgVth{0.5,1.5}` and `\cfgAlpha{0.8,0.99}` vs code defaults

The code defaults are $V_{th}\in[0.1,2.0]$, $\alpha\in[0.5,0.99]$; the paper's macros are narrower. The run profile is not in the snapshot, so I cannot confirm the actual bounds. If the profile uses the defaults, the text understates the search; if it uses the macros, the text is right. Flag as unverifiable, not proven.

### m24 — "below 8 KB" vs "$\le$ 8 KB"

The floor applies at exactly 8 KB too. Minor wording.

### m25 — Threshold-add citation

The threshold add is from Hoefflin & Wassner 2025, not Lemaire et al.; check `paper.bib`. If absent, add it. Minor.

### m26 — `finalize_baseline_selection` recomputes test rows

The code comment says the test-row restore was removed 2026-11-02 and every run recomputes test/per-window rows. This is a reproducibility risk if a rerun changes baseline selection; document the exact commit/version and the no-restore behavior. (Not a paper claim; a code-review note.)

### m27 — "Development fold kept out" is vacuous for AudioMNIST/Siena

`\cfgDevFoldsAudioMNIST{--}`, `\cfgDevFoldsSiena{--}`: those datasets have headline folds 0–5 and no development fold. The sentence is true vacuously, but the design study is FSDD/EEG only. Say so.

### m28 — "Fixed in advance" without pre-registration

The design study and the main grid use the same folds; the rule can be post hoc. Minor (overlaps M14).

---

## Verified as correct (independent checks)

These are the claims/implementation details I could reproduce exactly; I list them so the authors know what not to change.

1. **LIF forward (Eq. 1).** $V_l[t]=\beta V_l[t-1]+\sum_j W_{lj}s_j[t]+b_l$; $s_l[t]=\Theta(V_l[t]-V_{th})$; hard reset to $0$; $\beta=e^{-\Delta t/(RC)}$; code matches (`LifBPTT.hpp` forward, $V_{reset}=0$, `V > Vth`). Caveat: $V_{th}$ clamp mismatch (m20).
2. **Surrogate.** $\hat\sigma'(x)=\alpha/\bigl(2(1+(\pi\alpha x/2)^2)\bigr)$; `ArcTanSurrogate` default $\alpha=2$ uses `alpha/(2*(1+x*x))` with $x=(\pi/2)\alpha(v-V_{th})$; $\hat\sigma'(0)=\alpha/2=1$. Matches the paper.
3. **Energy arithmetic.** ANN $[59,16]$: $E=854.4853$ nJ; memory $82.6\%$; arithmetic $148.61$ nJ; 32-bit price $501.55$ nJ; repricings $481.3/813.9/3475.2$ nJ at 8 KB/32 KB/1 MB. Reproduced with an independent Python implementation (`dossier/math_scripts/checks.py`), matching `dossier/_scripts/energy.py` and the dossier's line 13.
4. **Table 1 dense/spike/LIF core rows.** Most cells match the code: dense MACs/reads/writes/bias/address; spike input; LIF leak/reset; threshold/reset per step/per spike. Exceptions are M8.
5. **MAC estimator (`estimate_snn_macs`).** Encoder $=M w_1+\sum_i w_i w_{i+1}$; decoder mirrors reversed widths $+\,w_1M$; returns encoder $\cdot\max(1,T_{enc})+$ decoder $\cdot\max(1,T_{dec})$; the `first_layer_steps` overload validates $1..\max(1,T_{enc})$. Matches the cost objective.
6. **GA/NSGA-II.** Population 8, generations 5, crossover 0.9, mutation 0.2, tournament 2 $\Rightarrow$ 48 nominal; `pareto_front` is the first feasible front; `pick_winner` sorts the first front by `val_mse` then `inference_cost`; the global minimum validation MSE is on the first front, so "lowest validation MSE (ties: lower cost)" is consistent with the code. `winner_seeds=3` default.
7. **Split/leakage.** Fold $f$ test block $f$, validation block $f+1$, train the rest; headline folds 1–5, development fold 0; block 0 is in fold 1's training, so the design choices (selected on fold 0 test) are not in the headline test. Matches the paper (split code not fully read; see unverifiable).
8. **Statistics plumbing.** Recording-level keys `(cv_fold, recording_id)`; means over windows and 5 seeds; percentile bootstrap over recordings; Wilcoxon signed-rank on $d_r$; Holm running max `min(1,(m-rank)*p)`; `_min_attainable_p=2/2^n`; coverage refuses missing fold/recording/seed. Matches the paper's description.
9. **T1 arithmetic.** Counts and means match `paper_t1.tex`: FSDD 1/8 warm-up, 1/8 pre-norm, 0/8 control; EEG 0/8 warm-up, 1/8 pre-norm, 1/8 control; means 0.9162/0.9305/0.9350 (FSDD) and 0.9934/0.9210/0.9925 (EEG). The numbers are right; the interpretation is not (m15).
10. **Design-table numbers.** The quoted numbers match the table (0.667/0.654, 0.489/0.293, 0.685, 0.308, 0.743, 0.547, 0.780/0.592, 0.734, 0.724/0.729, 0.484/0.449, 0.729, 0.457/0.396). The exceptions are the "Xavier better or equal" and "gap vanished" claims (M12).
11. **Status bookkeeping.** `\cfgDataStatus{placeholder}`; `paper_loso_*` and `paper_prevalidation*` are placeholders; `paper_t1.tex` is real. `\cfgHolmRefs{three}` is consistent with the overview table's four models (Mean, PCA, SNN-AE, ANN-AE), so $m=3$ is not an inconsistency.
12. **ANN twin.** `run_ann_twin` forces direct encoding, $T=1$, no input transform, no search; this matches the paper's "applied once to the analog window" (but not "only the hidden neuron differs", F3).

---

## Could not verify (and why)

1. **Run profile.** The actual `meeting01` profile (search bounds, `encodings`, `input_transforms`, `snn_architectures`, `latent_dim`, `winner_seeds`) is not in the snapshot; so the paper's `\cfgVth{0.5,1.5}`, `\cfgAlpha{0.8,0.99}`, `\cfgSnnDepth{1,4}`, `\cfgTmin{2}`, `\cfgTmax{32}` vs the code defaults cannot be confirmed. I flag the narrow/wide mismatch as unverifiable (m23), not as a proven defect.
2. **Headline grid, significance, energy, pre-validation.** All are placeholders; there is nothing to check. The design-study primary files (job 5315335) are not in the tree; the design table is hand-typed and predates formats 7/8/9 (dossier line 5). I checked the table against the text, not against raw runs.
3. **Split implementation.** I read the paper/docstring and the fold convention, but not `build_loso_split`/`assert_split_disjoint_and_manifest`; the leakage claim is from the paper/docstring, not from a line-by-line code audit.
4. **`Meeting01Metrics.cpp` full functions.** I read the cost formulas and `estimate_snn_macs`, but not every line of `snn_input_repeats_every_step`/`arith_energy_nj`; the contract is clear, the edge cases are not.
5. **Real data.** Siena and AudioMNIST are not on this machine; real-data EDF header reads are outside the snapshot. Cannot check the $16$ kHz/256-sample/window arithmetic for those datasets.
6. **C++ build.** Not run (forbidden); energy is reimplemented in Python. The Python reimplementation reproduces the author's numbers, but it is not the same code path.
7. **Exact re-score seed collisions.** `base_seed+extra*7919` could collide with other seeds in principle; not checked. Minor.

---

## Overall verdict

**REJECT** in the current form.

The paper has a genuine question (does the spiking autoencoder's energy advantage survive matched quality?), a real pilot (T1), and a useful energy model. But the submission is not a finished paper: the headline result is a placeholder (F1), the causal claim "the cost is the spike" is not identified because the control differs in encoding, $T$, and transform (F3), and the stated mechanism "saturation" is contradicted by the paper's own sigmoid/clamp rows (F2). These are not presentation issues; they go to the paper's thesis. On top of them, the energy model prices 32-bit words at 64-bit energies (M5), the "conservative" claim is false/inconsistent (M6), the winner's-curse re-score keeps the search seed (M9), and the primary test is pseudoreplicated (M13).

**Path to MAJOR REVISION.** (i) Run the grid and fill the tables; (ii) replace the matched control with a factorial one (same encoding/$T$/transform, only LIF vs LReLU) or reword "the spike" to "the spiking pipeline"; (iii) rename the mechanism to hard threshold/quantization and support it with a smooth-vs-hard experiment; (iv) price 32-bit words at 32-bit energies (or use 64-bit words explicitly) and report both; (v) re-score on fresh seeds only (or nested CV), and report the selection bias; (vi) make the group-level test primary or use a cluster-robust/hierarchical test, and define a non-inferiority margin for "Energy first"; (vii) correct Eq. (2) to the exact derivative and add a finite-difference check; (viii) fix Table 1 (first-layer-once read-backs, analog TimeMeanPool, encoder, nonlinearity); (ix) state the reachable depth range and the run profile. If (i)–(iii) are done, the paper can be a credible MAJOR REVISION; without them, it is not reviewable as an empirical paper.

**Top 5 objections (for the response):** F1 (no headline results), F3 (control not matched), F2 (saturation mechanism contradicted), M5 (64-bit price for 32-bit word), M9 (winner's-curse re-score includes the search seed). M13/M6/M8 are the next tier.
