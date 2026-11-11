# Author's dossier: "What Does Spiking Cost an Autoencoder?" (paper.tex, format 9, 2026-11-04)

Written by the author's side for two hostile reviewers. Every claim in document order, status, evidence, reasoning. Where I could not defend a claim I say so; proposed text changes are at the end and are proposals only (paper.tex is untouched).

**Paper status.** Section Results and the pre-validation table/verdicts are placeholders. The main grid has not run; pre-validation job 5316016 is queued. `paper_t1.tex` is real (format 9). Table tab:design and the design-study prose are hand-typed and predate formats 7, 8, 9.

**Status vocabulary.** SUPPORTED, PARTIAL, UNSUPPORTED, WRONG, CONDITIONAL (depends on a run that has not happened).

**Path shorthand.** `H/` = `software/nn/src/experiments/meeting01/lib/`; `P/` = `software/nn/scripts/pipeline/meeting01/`; `W:n` = `software/nn/.wiki/Experiments/Meeting01.md` line n; `QT` = `src/experiments/meeting01/questionsAndTasks.md`; `LIF.hpp` = `software/nn/include/layers/spiking/LifBPTT.hpp`.

**What I actually ran** (all in `dossier/_scripts/`, or inline):
- `checks.py`: surrogate integral, LIF constant-current simulation, Wilcoxon floor, a re-implementation of `repair_widths`.
- `energy.py`: independent re-implementation of Table tab:cost's analog path. It reproduces the notes' ANN-AE figure for widths [59,16] to four decimals (854.4853 nJ) and its three repricings (481.3, 813.9, 3475.2 nJ), arithmetic 147.64 nJ.
- `02_`, `03_`, `04_ --self-test`: all pass (known-answer checks on synthetic data). C++ gtests were NOT run: no build exists in the snapshot.
- T1: recomputed every number of `\tOneVerdict` from `results/meeting01_t1/t1_summary.csv`.
- Real-data reads outside the snapshot (read-only): EDF headers of all 1526 EEGMMIDB files and the FSDD filename list, under the databases folder named in the profile. Siena and AudioMNIST are not on this machine.

**What I could not run.** An encoder-only clamp under Xavier (the experiment that would settle pressure point 3): I wrote the probe, and the harness's PreToolUse hook blocked the command. I did not bypass it. See "Not checked".

**Literature access.** PDF text read directly (pdftotext, passages cited): Lemaire (arXiv 2210.13107), Horowitz (ISSCC PDF, Fig. 1.1.9 viewed as an image), Xiong (arXiv 2002.04745), Eshraghian (arXiv 2109.12894), Spikformer (arXiv 2209.15425v2), Kamata (arXiv 2110.00375, keyword search of the text). Read through a fetch tool that returns summaries and short quotes, so reviewers should re-check: Comşa 2021 (Frontiers), Pfeiffer (first 100k characters). Abstract only: Dampfhoffer, Yan, Cawley, AudioMNIST. Metadata only (Crossref): most of the rest. Not opened: Goodfellow Alg. 7.1 (the fetch returned nothing), Maass 1997, Gerstner.

---

## 1. Claims

### Title and abstract

**C001** Title — "Leakage-Controlled Comparison" · logic · **PARTIAL** · Split-level leakage is controlled by assertion (C191). Design-level leakage is not fully: fold 0's validation speaker is the test speaker of fold 1 (C194).

**C002** Title — "Matched Non-Spiking Control" · design choice · **PARTIAL** · Matched in widths, initializer, optimizer, data. Not in input code, input transform, time axis (C007, C041).

**C003** Abs — "Spiking autoencoders are usually compared with non-spiking networks that differ from them in more than the neuron" · literature · **UNSUPPORTED** · No citation gives "usually". The only two comparisons the paper cites are the opposite: Kamata uses "ANN VAE of the same architecture" (arXiv 2110.00375, contributions), Comşa §2.8 "conventional ANN (non-spiking) autoencoders of similar architecture". Related Work then says both "support the matched design". The paper contradicts its opening line.

**C004** Abs — "and mostly on images" · literature · **PARTIAL** · True of both cited works. No survey cited. Audio exists: Spiking Music (arXiv 2402.01571, piano, against VQ-AEs).

**C005** Abs — "We ask how much reconstruction quality the spike costs, and what it returns in energy" · design choice · **SUPPORTED** · Statement of aim.

**C006** Abs — "for short speech and single-channel EEG windows" · code · **SUPPORTED** · 256 samples; signal 0 only (`H/src/Meeting01Eeg.cpp`, `read_signal0`).

**C007** Abs — ANN-AE "copies its selected widths, initializer and training and replaces only the leaky integrate-and-fire neuron by a leaky ReLU" · code/logic · **WRONG** · Widths, initializer, training: yes (`run_ann_twin`, `H/src/Meeting01Experiment.cpp:1062`; `make_ann_twin_cfg`, `H/src/Meeting01Training.cpp:284`). "Only": no. It also drops T steps, the time mean, the spike encoder, the input transform (conv1d/recurrent), and the trained R, C, V_th. Fig. 1 itself paints "mean over T" red as spiking-only. Dev-fold evidence: 3 of 4 SNN winners in job 5315335 were Poisson (QT T2), so the extra differences are the common case, not an edge case.

**C008** Abs — "and with PCA" · code · **SUPPORTED** · `P/03_`.

**C009** Abs — SNN-AE "selects its architecture, the number of simulation steps, the initial threshold and decay, the spike encoding and an input transform by multi-objective search" · code · **SUPPORTED** · `Genome`, `H/include/Meeting01GaGenome.hpp`; `evaluate_individual` objectives.

**C010** Abs — "inside nested leave-one-group-out cross-validation" · design choice · **PARTIAL** · Outer loop LOGO. Inner "loop" is one validation block, block f+1 (`assign_speaker_fold`, `H/src/Meeting01Dataset.cpp`). That is nested hold-out, not nested CV in Cawley & Talbot's sense.

**C011** Abs — "on four datasets" · config · **CONDITIONAL** · Profile lists four; none has run. Siena and AudioMNIST have never been executed under format 9.

**C012** Abs — "differences are tested at the recording level" · code · **SUPPORTED** · `recording_level`, `P/04_...py:226`.

**C013** Abs — "The target is a low-power device, so energy is the cost we put first" · design choice · **PARTIAL** · Energy is not in selection: the winner is the lowest-MSE front member, and the GA's second objective is a MAC count, not E. "First" holds only in the reading rule, and that rule fails for PCA/Mean (C206).

**C014** Abs — "estimated per window with a published analytical model that counts memory accesses and address arithmetic as well as operations" · literature/code · **SUPPORTED** · Lemaire Eqs. 4–19; `H/src/Meeting01Energy.cpp`. Reproduced numerically for the ANN path (`energy.py`). "Extended", not "unchanged": C144.

**C015** Abs — "at published 45 nm prices" · literature · **SUPPORTED** · Horowitz Fig. 1.1.9, viewed.

**C016** Abs — "every energy comparison is repeated under each of three memory sizes, with arithmetic alone, and with a direct-coded input's first layer recomputed at every step" · code · **PARTIAL** · True for SNN-AE vs ANN-AE (`ENERGY_CHECKS`, `P/meeting01_results.py:114`; `energy_verdict`, `P/02_...py:951`). Self-test passes. Not "every": PCA and Mean have no E. The outputs are placeholders.

**C017** Abs — sequence AEs "differ from the SNN-AE in more than the neuron" · logic · **SUPPORTED** · Recurrence, frames, tanh latent.

**C018** Abs — "were measured once, under one reduced search budget on development folds" · history · **PARTIAL** · Three jobs: 5314513 (format 6), 5315335 (format 7), 5316016 (queued). The stopping rule's s=0.10 was tuned on the first (`H/include/Meeting01Stopping.hpp`). "Reported once" is true, "measured once" is not. Numbers are CONDITIONAL on 5316016.

**C019** Abs — "for their training and inference cost and their quality" · code · **CONDITIONAL** · Report machinery exists (`P/00_...report.py`); data pending.

**C020** Abs — "and left out of the main comparison" · config · **SUPPORTED** · `evaluation.baselines = ["ann-ae"]`, `meeting01-loso.json`.

**C021** Abs — "A design study on a development fold, excluded from the headline numbers" · code · **PARTIAL** · The fold's test group is excluded (`DEVELOPMENT_FOLDS`, `P/meeting01_results.py:268`). Its validation group is not (C194). There are two folds, one per dataset.

**C022** Abs — "two avoidable choices -- scoring every simulation step and a spiking decoder -- enlarged the gap" · measured · **PARTIAL** · Per-step: Table row 4→5 FSDD −0.046, EEG −0.138 (wiki only, 3 seeds, no primary). Spiking decoder: the table does not show it. Row 5 (LIF decoder, time mean, T=16) vs row 7 (LReLU decoder, latent mean, T=16): FSDD 0.734→0.729, EEG 0.454→0.457. The evidence that exists is outside the table: `enc_lif_only_rate` 0.721 vs `snn_rate` 0.734 (W:1420, 1431) and EEG output-mean 0.403 vs 0.454 (W:1537). Effect size 0.013 on FSDD with no dispersion reported.

**C023** Abs — "the initializer decided the EEG comparison" · measured · **PARTIAL** · Twin gap ≈0 under Kaiming, 0.156 under Xavier at T=4 (W:1508). C++ agrees in direction (W:1564–65). No primary log, n=3 and n=2.

**C024** Abs — "with fan-in scaling in both arms the control was handicapped and the gap vanished" · measured · **PARTIAL** · Same sources. Holds at T=4 (0.484 vs 0.489). At T=16 the Kaiming SNN is 0.457, also below the handicapped control.

**C025** Abs — "what remains is reproduced by a non-spiking network with a hard-clipped activation" · measured/logic · **UNSUPPORTED** · "What remains" is the Xavier gap of the adopted design. No clamp network was run under Xavier (Table: "−"). The clamp was applied in both encoder and decoder (`Ae.a1`, `Ae.a2`, `scripts/analysis/snn_gap_ablation_torch.py:158-160`), so it does not match the adopted encoder-only design. On EEG it does not reproduce the SNN with the same loss (0.547 vs 0.454): C235, C236.

**C026** Abs — "the cost is the saturation of the neuron's transfer function, not the spike" · logic · **UNSUPPORTED** · The strongest sentence in the paper and the weakest. See pressure point 3. The twin evidence is direct-input only, EEG winners were Poisson (QT T2), T matters on EEG though a clip does not depend on T, and a hard-reset LIF's time mean under constant current is a staircase with 4 levels at T=4 (`checks.py`), which is discreteness. Section design itself says "argued rather than measured".

### Keywords, Introduction

**C027** Intro — "SNNs communicate with binary events" · literature · **SUPPORTED** · Textbook; Maass, Pfeiffer.

**C028** Intro — "a synapse whose input did not spike does no work" · literature · **PARTIAL** · Pfeiffer §1.2: "does not compute much". But in the paper's own energy model each neuron pays a bias, a leak MAC, a threshold test and potential traffic at every step regardless. Attribution to Maass 1997 not verified (not opened); Maass is a computational-power paper.

**C029** Intro — "a synapse whose input did spike adds a weight instead of multiplying by it [maass, pfeiffer]" · literature · **PARTIAL** · Pfeiffer §3.2 says "spike operations are cheaper than real-valued matrix multiplications" and warns the SNN may need more of them; it does not say "adds". Lemaire §III-A says it exactly. Cite Lemaire.

**C030** Intro — "Surrogate gradients ... made such networks trainable end to end" · literature · **SUPPORTED** · Neftci 2019 (title and metadata verified), Zenke; Eshraghian §on surrogates (read).

**C031** Intro — "autoencoders are an obvious use: compress a signal window to a small code on a low-power device and reconstruct it elsewhere" · logic · **PARTIAL** · But the paper prices encoder + decoder together on one chip. The stated use case splits them, and only the encoder would be on the device. E as defined does not answer the scenario.

**C032** Intro — "a question about two numbers measured on the same network" · logic · **SUPPORTED** · Framing.

**C033** Intro — "a spike turns a multiply-accumulate into an add" · literature · **SUPPORTED** · Lemaire Eq. 2.

**C034** Intro — "every operand must also be fetched, at a higher price than the arithmetic [horowitz]" · literature · **SUPPORTED** · Fig. 1.1.9: 10 pJ ≥ any arithmetic there.

**C035** Intro — "a spiking layer reads and writes its membrane potentials at every simulation step [horowitz, dampfhoffer]" · literature · **PARTIAL** · True (Lemaire Eqs. 10, 15). Horowitz says nothing about SNNs. Dampfhoffer abstract does not state it. Wrong citation, right fact.

**C036** Intro — "Spiking autoencoders have been reported as comparable to conventional ones on MNIST-type images, against baselines of similar size [comsa]" · literature · **SUPPORTED** · Comşa §3.1: "reconstruction loss values ... are similar". Note their abstract says "similar to or better".

**C037** Intro — "or the same architecture [kamata]" · literature · **PARTIAL** · Same architecture: yes. "MNIST-type": Kamata also uses CIFAR-10 and CelebA.

**C038** Intro — recurrent/attention models "differ in recurrence, sequence length and capacity at once, so a quality difference cannot be attributed to the neuron" · logic · **SUPPORTED** · Sound.

**C039** Intro — "We are not aware of a matched comparison for the reconstruction of speech or EEG" · literature · **PARTIAL** · Two searches (2026-11-04) found none. Nearest: Spiking Music (piano, unmatched), Baronig et al. arXiv 2408.07517 (spiking waveform reconstruction, not matched). An honest "not aware" claim from a non-systematic search.

**C040** Intro — "windows of one recording and one speaker are near-duplicates, so pooling windows before splitting measures memorization" · logic · **PARTIAL** · Directionally right and it was the rejection reason (W:7-10). "Near-duplicates" is unmeasured.

**C041** Intro — "selecting hyperparameters on the test speaker biases every number [cawley]" · literature · **SUPPORTED** · Cawley abstract: selection bias from over-fitting the criterion.

**C042** Intro — comparison "can be skewed by design choices that are not the spike, and in our case were" · measured · **PARTIAL** · Depends on C022–C024.

**C043** Intro — "a per-step loss and a spiking decoder that inflated the gap" · measured · **PARTIAL** · As C022.

**C044** Intro — "an initializer that, applied to both arms, handicapped the control enough to make the gap vanish on EEG" · measured · **PARTIAL** · As C023.

**C045** Contrib 1 — ANN-AE "differs from it only in the encoder's hidden neuron" · code · **WRONG** · As C007.

**C046** Contrib 1 — "so the gap between them is the cost of spiking" · logic · **PARTIAL** · Causal language not earned for "the spike". Earned for "the selected spiking pipeline" (LIF dynamics + code + transform + T + time mean), under a 30-epoch budget at 1000 windows. W:1364–1372: the gap grows with training windows (0.165 at 1000, 0.220 at 10000), so it is budget-dependent too.

**C047** Contrib 2 — "nested 6-fold leave-one-group-out cross-validation" · design · **PARTIAL** · As C010.

**C048** Contrib 2 — "on two speech and two EEG datasets" · config · **CONDITIONAL** · As C011.

**C049** Contrib 2 — "selected on inner validation groups only" · code · **SUPPORTED** · `evaluate_individual` passes `split.val_samples`; `pick_winner` uses that MSE. The test group is not reachable from the search.

**C050** Contrib 2 — "recording-level paired statistics" · code · **SUPPORTED** · C012.

**C051** Contrib 3 — "counts memory accesses and address arithmetic as well as operations" · code · **SUPPORTED** · `add_matmul`, `add_dense_fc`.

**C052** Contrib 3 — "with the analytical model of [lemaire] at the 45 nm prices of [horowitz]" · literature · **PARTIAL** · Mixed: Lemaire price ops with 32-bit integer figures ("we use respectively 0.1pJ and 3.1pJ"). We use fp32 for ops, int32 for addresses. Defensible (fp32 weights), but it is not Lemaire's pricing.

**C053** Contrib 3 — "applied unchanged to every model" · literature · **WRONG** · See C144.

**C054** Contrib 3 — "the SNN's counts taken from its executed forward pass" · code · **PARTIAL** · Spikes yes (`Meeting01Evaluation.cpp:177`). The primary count explicitly departs from execution for a repeated first layer.

**C055** Contrib 3 — "every comparison checked under three memory sizes, under arithmetic alone, and with a repeated input recomputed at every step" · code · **PARTIAL** · As C016.

**C056** Contrib 4 — design study "locates where the SNN-AE's reconstruction error comes from" · logic · **UNSUPPORTED** · It eliminates some causes; it does not locate the remainder (C026).

**C057** Contrib 4 — "a development fold kept out of the headline numbers" · code · **PARTIAL** · As C021.

**C058** Contrib 4 — "a pre-validation on the same folds that measures what searched sequence autoencoders cost to train and to run" · code · **CONDITIONAL** · Job 5316016.

### Related Work

**C059** RW — Comşa "trained spiking autoencoders with temporal coding" · literature · **SUPPORTED** · Title and text.

**C060** RW — "found reconstruction errors comparable to conventional autoencoders of similar size on MNIST and Fashion-MNIST" · literature · **SUPPORTED** · §2.8, §3.1. They note ANNs have "slightly smaller parameter count".

**C061** RW — Kamata "compared against a non-spiking VAE of the same architecture on image datasets" · literature · **SUPPORTED** · arXiv 2110.00375: MNIST, FashionMNIST, CIFAR10, CelebA; contribution bullet.

**C062** RW — "Both support the matched design we adopt" · logic · **PARTIAL** · They use a matched baseline; they do not argue for it. And this undercuts C003.

**C063** RW — "neither addresses temporal signals or a group-disjoint protocol" · literature · **PARTIAL** · Comşa: confirmed no audio task. Kamata: images. "Group-disjoint" I inferred from image-only datasets; not separately checked.

**C064** RW — "We train with surrogate gradients and backpropagation through time" · code · **SUPPORTED** · LIF.hpp backward; Werbos metadata verified.

**C065** RW — "the three common input codes: direct currents, rate (Bernoulli) coding, and latency coding" · literature · **PARTIAL** · Latency cited (Thorpe, Comsa). Direct and rate uncited; Lemaire §IV or Eshraghian would do.

**C066** RW — "A widespread convention prices an SNN's analog-input layers by MACs and its spike-driven layers by ACs, with 45 nm figures from Horowitz; Spikformer is one example" · literature · **SUPPORTED** · Spikformer appendix: "EMAC = 4.6pJ and EAC = 0.9pJ"; first-layer FLOPs × E_MAC, SOPs × E_AC. "Widespread" from one example; Lemaire §II backs it ("Most energy consumption metrics").

**C067** RW — "It leaves out memory, which in the same table costs more than the arithmetic it feeds (a 32-bit float add 0.9 pJ, one access to a 32 KB SRAM 20 pJ)" · literature · **PARTIAL** · Numbers correct. The table says "Cache (64bit)": a 64-bit cache access, not an SRAM of 32-bit words. Must be stated.

**C068** RW — "the traffic only an SNN has: its potentials, read and written at every step" · logic · **PARTIAL** · Against feed-forward ANNs, yes. LSTM/GRU cells also read and write state, and the paper's code prices it (`recurrent_ae_energy`, `rd_pot`).

**C069** RW — "Estimates that count memory do not agree on the outcome, which depends on the spike rate, the number of steps and the hardware" · literature · **SUPPORTED** · Lemaire §II surveys this; Dampfhoffer, Yan abstracts.

**C070** RW — Lemaire "find their SNNs 6 to 8 times more efficient than non-spiking networks" · literature · **SUPPORTED** · §V: "between 6.25 and 8.02 times lower".

**C071** RW — Dampfhoffer "that SNNs match efficient ANN accelerators only at very high sparsity" · literature · **SUPPORTED** · Abstract: 0.15–1.38 spikes per synapse per inference. Abstract only (HAL record).

**C072** RW — Yan "that an advantage requires strict limits on both" · literature · **SUPPORTED** · Abstract: T=5, spike rate below about 5.7%. Abstract only.

**C073** RW — "We therefore count memory, and report whether each conclusion survives a change of the memory model" · code · **SUPPORTED** · C016.

**C074** RW — LSTM, GRU, self-attention AEs "are the usual non-spiking baselines for sequences" · literature · **PARTIAL** · Cited works are the architectures (Hochreiter, Cho, Chung, Vaswani, Ba), not autoencoder baselines. Sutskever is the nearest.

**C075** RW — "Group-disjoint, nested evaluation is the standard remedy for leakage in model selection [cawley]" · literature · **PARTIAL** · Cawley covers selection bias; the abstract names neither nested CV nor group-disjoint splits. "Group-disjoint" needs another source (e.g. Roberts et al. 2017 Ecography; Varma & Simon 2006 for nested; both proposed but unverified here).

### Methods: task, SNN

**C076** Task — "A window of M=256 consecutive samples" · config · **SUPPORTED** · Profile.

**C077** Task — "is z-scored on its own statistics" · code · **PARTIAL** · True (`WindowZScore`), but incomplete. EEG is band-passed 0.5–40 Hz with a notch at 50/60 Hz on the whole recording first (`H/src/Meeting01Eeg.cpp`, `bandpass_notch`; `Meeting01Dataset.cpp`). The paper never says so: an omitted method.

**C078** Task — "minimizing the mean squared error" · code · **SUPPORTED** · Masked MSE.

**C079** Task — "d is fixed per signal domain (d=16 speech, d=64 EEG) and is never searched" · code · **SUPPORTED** · `repair_widths` forces `back()`.

**C080** Task — "so every model faces the same compression ratio" · logic · **SUPPORTED** · Same d. PCA k=d. Per-window mean and std are discarded for all alike.

**C081** Task — "A recording's last window is zero-padded" · code · **PARTIAL** · Speech only. The EEG loader drops the trailing partial window (`offset + window_size <= size`).

**C082** Task — "padded samples are excluded from normalization, every loss and every reported error" · code · **SUPPORTED** · W:378–398; `make_activity_mask`; `03_` masks too, with a self-test.

**C083** SNN — encoder hidden layers are LIF "simulated for T steps" · code · **SUPPORTED** · `render_encoder_spec`: "linear:w:leaky".

**C084** SNN — Eq. lif, $V[t]=\beta V[t-1]+I[t]$, $s[t]=\mathbf 1[V[t]>V_{th}]$ · code · **SUPPORTED** · LIF.hpp:237, 254 (strict `>`).

**C085** SNN — "and $V[t]\leftarrow 0$ after a spike (hard reset)" · code · **SUPPORTED** · LIF.hpp:260; `reset_zero` default.

**C086** SNN — "$\beta=e^{-\Delta t/(RC)}$ and $\Delta t=1$" · code · **SUPPORTED** · LIF.hpp:207–208; `make_snn_cfg`.

**C087** SNN — "R, C and V_th are trained per layer" · code · **SUPPORTED** · 1×1 tensors in `params()`.

**C088** SNN — "(clamped above 10^-6)" · code · **WRONG** · R and C only (LIF.hpp:205–206). V_th is read unclamped at line 209; a search of `include/` and `src/core` finds no V_th clamp. Header comment item 1 says R/C. Trivial fix.

**C089** SNN — "initial values R=1, C=−1/ln β^(0)" · code+math · **SUPPORTED** · RC = −1/ln β⁰ gives e^{ln β⁰} = β⁰.

**C090** SNN — "so the searched β^(0) is the initial decay" · math · **SUPPORTED** · Above.

**C091** SNN — surrogate $\hat\sigma'(V)=\alpha/(2(1+(\frac{\pi}{2}\alpha(V-V_{th}))^2))$, α=2 · code+math · **SUPPORTED** · `ArcTanSurrogate.hpp:41-46`, default α=2. ∫=1, peak 1 (`checks.py`). It is exactly the derivative of Eshraghian's $\tilde S=(1/\pi)\arctan(\pi U)$ at α=2. Their printed gradient line, $(1/\pi)\cdot 1/(1+(\pi U)^2)$, carries an extra 1/π and disagrees with their own $\tilde S$; ours follows $\tilde S$.

**C092** SNN — backward recursion $\delta_l[t]=\hat\sigma'(V_l[t])\odot(W_{l+1}^\top\delta_{l+1}[t])+\beta\,\delta_l[t+1]$ · math/code · **WRONG** · Derivation: with hard reset, $V[t+1]=\beta(1-s[t])V[t]+I[t+1]$, reset path detached, so $\partial V[t+1]/\partial V[t]=\beta(1-s[t])$. The code does this: `dvpost_dvpre = 1 - spike` (LIF.hpp:412), `grad_v_pre = grad_out*surr + grad_from_next*beta*dvpost_dvpre` (LIF.hpp:424). The paper's equation omits $(1-s_l[t])$, so it describes a no-reset neuron. Consequence for readers: the gradient through a step that fired is zero.

**C093** SNN Network — "$\mathrm{Lin}(M\to w_1)\to\mathrm{LIF}\to\dots\to\mathrm{Lin}(w_{L-1}\to d)$ with $w_1>\dots>d$; the latent layer is analog" · code · **SUPPORTED** · `render_encoder_spec`; `repair_widths`.

**C094** SNN Network — "averaged over the T steps and decoded once" · code · **SUPPORTED** · `time_mean` token, `TimeMeanPool`.

**C095** SNN Network — decoder is the "mirror image of the encoder with leaky ReLUs (slope 0.01) and a linear output" · code · **SUPPORTED** · `render_decoder_spec`; `LeakyReLU.hpp:24` default 0.01.

**C096** SNN Network — "The loss is computed on that single reconstruction" · code · **SUPPORTED** · `snn_output_steps`.

**C097** SNN Network — the earlier design "was measurably worse for reasons unrelated to spiking" · measured/logic · **PARTIAL** · "Measurably worse": per-step yes. "Unrelated to spiking": a per-step loss hurts because one step carries one binary event per neuron. That is a spiking reason, as Section design says in its own words (C223).

**C098** Input — direct: "constant current" · code · **SUPPORTED** · `encode_sample`.

**C099** Input — Poisson: "Bernoulli spikes with probability proportional to the min-max-scaled sample" · code · **SUPPORTED** · Equal to it; per-window min/max. Test: `meeting01_encoding_gtest` (not run).

**C100** Input — latency: "one spike per sample at a time that decreases with its value" · code · **SUPPORTED** · `t_f = round((1-scaled)(T-1))`.

**C101** Input — conv1d: "a fixed 3-tap smoothing along the window" · code · **SUPPORTED** · {0.25, 0.5, 0.25}.

**C102** Input — recurrent: "one LIF cell per sample with the network's threshold and decay" · code · **PARTIAL** · Uses the initial genes, not the trained values. And it is a different neuron: soft reset (`- s_prev*v_th`) and `>=` (`recurrent_lif_run`), where the network's LIF has hard reset and strict `>`.

**C103** Input — "which turns any code into spikes" · code · **SUPPORTED** · Output is binary.

**C104** Fig. 1 caption — "Shared blocks have identical widths, initializer and training" · code · **SUPPORTED** · C007's positive part.

**C105** Fig. 1 caption — "only the encoder's hidden neuron differs (red: the parts that exist only in the spiking network)" · logic · **WRONG** · Self-contradictory: two red boxes and the "×T" input. Conv1d, recurrent, encoder not drawn.

### Matched control, references, search

**C106** Control — "takes the widths of that run's selected SNN-AE" · code · **SUPPORTED** · `run_ann_twin`.

**C107** Control — "replaces each hidden LIF by a leaky ReLU applied once to the analog window" · code · **SUPPORTED** · `render_encoder_spec`, ANN builder maps "leaky" to LeakyReLU (`AutoencoderStageBuilders.hpp:80-82`).

**C108** Control — "trained with the same initializer, optimizer, data and stopping rule" · code · **SUPPORTED** · `finalize_baseline_selection` uses the same carve seed and fit. Windows identical. Poisson spike trains are of course not.

**C109** Control — "It is not searched" · code · **SUPPORTED** · Manifest "selection: none".

**C110** Control — "The SNN-AE therefore receives the larger search budget, which favours it" · logic · **PARTIAL** · Larger budget: yes. Favours it on MSE: yes, against a fixed alternative. But widths are picked for SNN validation MSE, so the control inherits whichever capacity won; a high-capacity winner may help the ANN too.

**C111** Control — "a gap in the ANN-AE's favour is conservative" · logic · **PARTIAL** · Direction is right as regards search alone. Not globally: the protocol (30 epochs, lr, batch 1, ArcTan α, 1000 windows) was fixed and diagnosed around an ANN-friendly regime; W:1544 "still improving" at the cap applies to both, and W:1545 shows the SNN gains 0.019 at 200 epochs vs the ANN's 0.010 (Kaiming). With 48 evaluations over 7 genes the SNN search is thin; an under-searched SNN overstates the intrinsic cost. See pressure point 2.

**C112** References — "PCA with k=d components, fitted on all non-test windows of the fold" · code · **PARTIAL** · Fitted on train ∪ val after the caps (1150 windows, `P/03_...py:343`), not on all non-test windows. Neural models fit ≈1000 (train \ monitor ∪ val). PCA therefore has ~15% more data, including the monitor windows. No test data involved.

**C113** References — PCA "is the best linear code through the same bottleneck" · math · **PARTIAL** · Eckart–Young on the fit set, affine with the mean. Not on test windows; the test ordering against an AE can go either way. Padded fit samples are imputed with the position mean, self-tested in `03_`.

**C114** References — "a mean-window predictor fixes the scale of 'no model'" · code · **SUPPORTED** · `03_`; ≈1.0.

**C115** Search — "selected by NSGA-II" · code · **SUPPORTED** · `run_ga_search`; Deb 2002 metadata verified.

**C116** Search — "population 8 for 5 generations (48 evaluations)" · math/config · **SUPPORTED** · 8×(1+5). Cached duplicates reduce trained genomes.

**C117** Search — "minimizing validation MSE and an analytic inference cost" · code · **SUPPORTED** · `ind.objectives`.

**C118** Search — "The final front is re-scored over 3 seeds" · code · **SUPPORTED** · `winner_seeds`. Mean includes the original selected-for score, so winner's curse is reduced, not removed.

**C119** Search — "the member with the lowest re-scored validation MSE wins (ties: lower cost)" · code · **SUPPORTED** · Sort in `run_ga_search`; `pick_winner`.

**C120** Search — "depth $L\in[1,4]$" · code · **WRONG** · `encoder_widths` includes the latent. `repair_widths` always yields at least one hidden layer: `repair([d],d)=[d+1,d]`. Reachable lengths are 2, 3, 4 (d=16: 0.254, 0.257, 0.490; `checks.py`). L=1 would be a network with no LIF at all. The macro prints the config bound, not the reachable range.

**C121** Search — "widths in $(d,128]$" · code · **SUPPORTED** · `hidden_min = latent+1`, `max_width`.

**C122** Search — "$T\in[2,32]$ (an integer drawn uniformly)" · code · **SUPPORTED** · `rand_time_steps`.

**C123** Search — "$V_{th}^{(0)}\in[0.5,1.5]$, $\beta^{(0)}\in[0.8,0.99]$, encoding and input transform" · config · **SUPPORTED** · Profile.

**C124** Search — "The cost objective is a spike-blind MAC count" · code · **SUPPORTED** · `estimate_snn_macs`.

**C125** Search — "in which T multiplies every encoder layer" · code · **WRONG** · Stale since format 9: a direct input through dense/conv1d has its first layer counted once (`Meeting01GaFitness.cpp`, `first_layer_steps`; `Meeting01Metrics.cpp:108-124`; profile `_time_steps_comment`).

**C126** Search — "it keeps cheaper genomes on the front" · logic · **SUPPORTED** · Non-dominated by cost.

**C127** Search — "the winner is chosen by validation MSE alone" · code · **PARTIAL** · Among the front, yes. Which genomes reach the front depends on cost, so cost shapes the candidates.

**C128** Search — "so a longer simulation is selected only when it validates better" · logic · **SUPPORTED** · A longer-T genome on the front must beat cheaper members on validation, up to noise.

### Training

**C129** Train — "All models use Adam (learning rate 10^-3; 10^-4 for R, C, V_th)" · code · **SUPPORTED** · Profile; `snn_lr_scale`. Kingma metadata fine.

**C130** Train — "batch size 1 and at most 30 epochs" · config · **SUPPORTED** · Profile.

**C131** Train — patience 5 "counted only once the monitored loss is 10% below that of the mean predictor" · code · **SUPPORTED** · `make_stopping_rule`; trainer test `PlateauEpochsDoNotCountUntilArmed` (not run).

**C132** Train — mean predictor is "the mean training window, scored on the same monitor windows" · code · **SUPPORTED** · `input_independent_monitor_loss`.

**C133** Train — "On z-scored windows every network starts on that plateau" · logic · **PARTIAL** · The code's own header says "often starts". Not measured for every network; the "plateau" is also fold-specific (0.962 FSDD).

**C134** Train — in an earlier run "plain patience stopped networks that had not left it yet, which then entered the comparison as the mean predictor under another name" · measured · **PARTIAL** · W:1973: 8 of 11 LSTM-AE evaluations, 3 of 8 final refits. Events logs of job 5314513 are not in the snapshot.

**C135** Train — "A run that never reaches that level trains all 30 epochs" · code · **SUPPORTED** · T1 rows with armed 0 confirm it (`stop_reason` per notes).

**C136** Train — "every run keeps the weights of its best monitored epoch [Goodfellow Alg. 7.1]" · code+literature · **PARTIAL** · Code: `BestWeightsCallback`. Citation not opened (fetch empty); from memory Alg. 7.1 returns the best parameters.

**C137** Train — "SNN-AE and ANN-AE are both initialized with Xavier [glorot]" · code · **SUPPORTED** · `xavier.hpp:65` limit √(6/(in+out)) is Glorot's normalized init; bias zero.

**C138** Train — "Left to their builders, the two arms had different defaults (fan-in scaling [he] for the SNN)" · code · **SUPPORTED** · `kaiming_snn.hpp:47` limit √(6/fan_in), variance 2/fan_in, He's variance with a uniform rather than Gaussian draw.

**C139** Train — "the initializer decides the EEG comparison, so it is fixed and shared" · design · **SUPPORTED** · `validate()` per W:1515.

**C140** Train — "retrained on training ∪ validation groups" · code · **PARTIAL** · On (train \ monitor) ∪ val: 150 training windows are held out.

**C141** Train — "early-stopped on a recording-disjoint slice of the training groups" · code · **SUPPORTED** · `carve_recording_disjoint_monitor`. Note the monitor shares groups with the fit, unlike validation.

**C142** Train — "evaluated once on the test group" · code · **SUPPORTED** · `finalize_snn_selection`.

**C143** Train — "Each fold is repeated 5 times (seeds 42–46), each repeat with its own search" · code · **SUPPORTED** · Driver loop; profile `repeats`. The window sets are fixed across seeds by `base_seed`, deliberately.

### Energy section and Table tab:cost

**C144** Energy — "it is the analytical model of Lemaire et al., applied unchanged to every model" · literature · **WRONG** · The following paragraph lists changes. Lemaire has no LSTM, attention, encoder, or threshold cost, has FIFOs of a fixed 1000 elements, integer op prices, and says of repeated static input that "the whole processing must be repeated over several timesteps". First-layer-once contradicts that directly. Fix: "extended".

**C145** Energy — "E = E_ops + E_addr + E_mem" · literature · **SUPPORTED** · Lemaire Eq. 18 (printed as E_mem + E_ops+addr).

**C146** Energy — Horowitz prices: fp add 0.9, mul 3.7, "a MAC 4.6", int add 0.1, SRAM 10/20/100 at 8 KB/32 KB/1 MB, 45 nm, 0.9 V · literature · **SUPPORTED** · Fig. 1.1.9 viewed; table says "Cache (64bit)".

**C147** Energy — "each layer holds its weights, its potentials and its input and output buffers in local SRAMs, each priced at its own size on the straight lines through those three points" · literature/code · **PARTIAL** · Local SRAM and linear interpolation: Lemaire §III-B, §III-D. Lemaire's I/O FIFOs are fixed at 1000 32-bit elements; ours are n_in/n_out words. Extrapolation to 2 MB is ours (`kMaxSramBytes`).

**C148** Energy — "one access moves one 32-bit word" · code · **PARTIAL** · True, but priced at the 64-bit cache figure, undisclosed (header of `Meeting01Energy.hpp`; W:2057). It overprices every access; halving it takes the ANN [59,16] from 854 to 501 nJ (`energy.py`), and it hits the higher-memory-share model harder. Unstated and in the direction "SNN dearer".

**C149** Energy — "The SNN-AE's counts come from its forward pass on each test window, with that window's spikes" · code · **SUPPORTED** · `Meeting01Evaluation.cpp:150-177`.

**C150** Energy — "the ANN-AE's follow from its architecture" · code · **SUPPORTED** · `ann_ae_energy`, zero window; reproduced.

**C151** Energy — "Where the model leaves a choice, we take the one that works against the conclusion it could support" · logic · **WRONG** · Which conclusion? Floor, threshold add and encoder charge all work against "the SNN saves energy", and therefore flatter the opposite, which is what dev folds show (QT T2). First-layer-once works the other way. The code header says "against the claim it could flatter", the format-8 note says "the conservative side for a paper whose measured conclusion is that the SNN-AE costs more". The two framings conflict, and the sentence hides it. Also unlisted: fp32 ops vs Lemaire's int32, the 64-bit price (C148). See pressure point 2.

**C152** Energy — "Memories below 8 KB, the SNN's potentials among them, are priced at 8 KB's 10 pJ" · code · **SUPPORTED** · `sram_access_pj:27`. Against the SNN; flagged in C151.

**C153** Energy — "The LIF's threshold test costs one add per neuron and step, which [Lemaire] do not count" · code+literature · **SUPPORTED** · `add_lif`; Eq. 1–2 have no comparison term.

**C154** Energy — spike encoder charged to the SNN-AE (Poisson, latency as described); "the ANN-AE has none" · code · **SUPPORTED** · `encoding_energy`.

**C155** Energy — "Leaky ReLU costs a comparison and a multiply" · code · **SUPPORTED** · `add_nonlinearity`. Comparison priced as an fp add.

**C156** Energy — "Reading the window from off-chip memory is left out for every model, since all start from the same buffer" · code · **SUPPORTED** · No such term.

**C157** Energy — direct coding: first layer and conv1d filter "computes the same current T times; E counts it once per window, plus one write of that current and its read at every step, as an accelerator that reuses a constant input would" · code · **SUPPORTED** · `walk_sequential:237-243`; guards at 201-214; `snn_input_repeats_every_step`.

**C158** Energy — "This choice favours the SNN-AE" · measured · **SUPPORTED** · QT T2: 14.4× → 1.53× for one winner (no primary file; the ANN side reproduced by `energy.py`).

**C159** Energy — "every energy comparison is repeated with every access priced at one SRAM size ..." · code · **PARTIAL** · As C016.

**C160** Energy — arithmetic alone "$4.6 N_{MAC}+0.9 N_{AC}$, the convention of [zhou]" · literature/code · **SUPPORTED** · Spikformer appendix; `arith_energy_nj`.

**C161** Energy — "Table tab:energy reports all of them" · code · **CONDITIONAL** · `write_energy_tex`; placeholders now.

**C162** Energy — "$E$ estimates a modeled accelerator; it is not a measurement on a chip" · logic · **SUPPORTED** · Honest.

**C163** Table cost caption — "R=T steps for the SNN-AE, but 1 for the first layer of a direct-coded input; 1 for the ANN-AE" · code · **SUPPORTED** · `walk_sequential`.

**C164** Table cost — MAC row, analog $Rn_{in}n_{out}$; spike "--" · code+math · **SUPPORTED** · `add_matmul`; Lemaire Eq. 2 first line.

**C165** Table cost — Add row, $Rn_{out}$; $(\theta+R)n_{out}$ · code · **SUPPORTED** · `add_dense_fc`, `add_spiking_fc`. Lemaire's Eq. 2 is printed as θ×N_in×N_out + T×N_out; the text, and Eqs. 8, 10, 15, give θ×N_out. We follow the latter; "after Lemaire" is fair, and the discrepancy deserves a footnote.

**C166** Table cost — Address add, $R(n_{in}+n_{out})$; $\theta n_{out}$ · code · **SUPPORTED** · Lemaire Eq. 17.

**C167** Table cost — Read input, $Rn_{in}$; $\theta$ · code · **SUPPORTED** · Eqs. 4, 5.

**C168** Table cost — Read weight, bias, $R(n_{in}+1)n_{out}$; $(\theta+R)n_{out}$ · code · **SUPPORTED** · Eqs. 6, 8.

**C169** Table cost — Read, write potential, "--"; $(\theta+R)n_{out}$ each · code · **SUPPORTED** · Eqs. 10, 15.

**C170** Table cost — Write output, $Rn_{out}$; spike-input cell "--" · code · **WRONG** · For the spike-input layer with no LIF after it, i.e. Lin(w_{L-1}→d), the latent layer present in every winner, `add_spiking_fc(..., writes_output=!next_is_lif)` writes $Rn_{out}$ (`Meeting01Energy.cpp:151, 236`). Table says none. W:2050 has the same gap.

**C171** Table cost note — LIF per neuron and step: leak MAC, threshold add; after analog input a read and a write of potential; per spike reset add and event write · code · **SUPPORTED** · `add_lif`.

**C172** Table cost note — "A layer that feeds an LIF writes no output of its own" · code · **PARTIAL** · Except the direct first layer's one stored write, per C157.

**C173** Table cost note — "Mean over T: one add per input spike, one multiply per output" · code · **WRONG** · The pool's input is the analog latent, so `in_is_spikes` is false: $Td$ adds, $Td$ reads, then $d$ MACs (not multiplies) and $d$ writes (`Meeting01Energy.cpp:265-275`). Note describes an input the network never has.

### Experimental setup

**C174** Setup — "FSDD and AudioMNIST are spoken digits at 8 kHz" · literature/code · **PARTIAL** · FSDD yes. AudioMNIST is 48 kHz, offline-resampled (W:40), undisclosed.

**C175** Setup — FSDD citation · literature · **PARTIAL** · Repository real; 3000 files verified on disk. bib gives "Zoë Jackson"; the maintainer is Zohar Jackson. Printed initial hides it. AudioMNIST has no citation at all: add Becker et al.

**C176** Setup — EEGMMIDB "(motor movement/imagery)", Siena "(epilepsy)" "are scalp EEG" · literature · **SUPPORTED** · PhysioNet pages. PhysioNet asks for dataset citations (Schalk 2009, Detti 2020 database) not in the bib.

**C177** Setup — "read from the first channel of each recording (Fc5 ...)" · measured · **SUPPORTED** · EDF headers: 'Fc5.' in 1526 of 1526 files.

**C178** Setup — "(... and Fp1 in the file headers)" · measured · **PARTIAL** · Wiki says verified on downloaded files. Siena is not on this machine; I could not re-check, and "each recording" in a corpus with varying montages deserves a full scan.

**C179** Setup — "The group is the speaker or subject" · code · **SUPPORTED** · Filename or parent directory.

**C180** Setup — "Windows do not overlap" · code · **SUPPORTED** · Loader offsets.

**C181** Setup — "long recordings are capped to a number of windows" · code · **SUPPORTED** · `max_windows_per_recording`.

**C182** Setup — "each fold keeps at most 1000 training, 150 validation and 1500 test windows, sampled round-robin across recordings" · code · **SUPPORTED** · `stratified_window_cap`; T1 manifest window_counts 1000/150/1500.

**C183** Table datasets — FSDD: speaker, #G 6, cap --, d 16 · measured · **SUPPORTED** · 6 names, 500 files each.

**C184** Table datasets — AudioMNIST: speaker, #G 60, cap 2, d 16, "8000 Hz" · literature · **PARTIAL** · 60: Becker et al. abstract. Rate after resampling.

**C185** Table datasets — EEGMMIDB: subject, #G 109, cap 40, d 64, "160 Hz" · measured · **PARTIAL** · 109 subject folders verified. Rate is not universal: 36 of 1526 files (S088, S092, S100, runs R03–R14) have 128 samples per 1 s record. PhysioNet's page says 160 Hz; the profile comment says it verified 160 on all 1526, which is false. The loader passes sampling_rate=160 for all, so on those files the 0.5–40 Hz band and the 60 Hz notch land at 0.4–32 Hz and 48 Hz, and a window spans 2.0 s. Silent, 2.4% of recordings, subjects in blocks 4 and 5, both headline folds.

**C186** Table datasets — Siena: subject, #G 14, cap 40, d 64, 512 Hz · literature · **PARTIAL** · PhysioNet: 14 patients, 512 Hz database-wide. Per-file rate not checked (no local copy); after C185 it should be.

**C187** Setup — "Groups are sorted and split into 6 contiguous blocks" · code · **SUPPORTED** · `group_of(rank)`; `split_audit_gtest` (not run).

**C188** Setup — "block f is the test set, block f+1 (mod 6) the validation set, and the rest train" · code · **SUPPORTED** · `assign_speaker_fold`.

**C189** Setup — "for FSDD this is leave-one-speaker-out" · code · **SUPPORTED** · T1 manifest: test george, val jackson.

**C190** Setup — "Every window is assigned by its group before any pooling" · code · **SUPPORTED** · `build_loso_split`.

**C191** Setup — "the harness aborts if a recording or group appears in two splits" · code · **SUPPORTED** · `assert_split_disjoint_and_manifest`, `Meeting01Experiment.cpp:1420`, throws "LEAKAGE". Compares train, val, test pairwise.

**C192** Setup — "All selection uses the validation block only" · code · **PARTIAL** · Architecture selection, yes. Epoch selection in the final fit uses the monitor slice of training groups. Design selection used fold 0 entirely.

**C193** Setup — design study "compared variants on fold 0 of FSDD and of EEGMMIDB, including their test windows" · history · **SUPPORTED** · W:1546.

**C194** Setup — "Those folds are therefore excluded from every headline number" · code/logic · **PARTIAL** · Fold-level exclusion is coded and tested by `04_`'s self-test. It is not sufficient. T1 manifest: fold 0 of FSDD has test george, val jackson, train the rest. Design decisions were read on jackson's validation windows ("Every choice above came out the same on the fold's validation windows"), and jackson is the test speaker of headline fold 1. Same for EEGMMIDB block 1. The remaining speakers trained the models that informed the design. AudioMNIST and Siena are the only clean datasets. Fold-1 sensitivity should be reported. This is the main leakage attack.

**C195** Setup — "FSDD and EEGMMIDB are reported on folds 1–5, AudioMNIST and Siena on folds 0–5" · code · **SUPPORTED** · `DEVELOPMENT_FOLDS`.

**C196** Setup — "Including them is reported as a sensitivity analysis" · code · **CONDITIONAL** · `write_sensitivity_tex`.

**C197** Stats — "Every model reconstructs the same test windows" · code · **SUPPORTED** · `check_coverage` refuses otherwise; `03_` refuses mismatched windows.

**C198** Stats — primary estimand $d_r$ · code · **SUPPORTED** · `recording_level`. Seeds are averaged first, so d_r excludes seed and search variance from the interval.

**C199** Stats — "a 95% bootstrap interval over recordings" · code · **SUPPORTED** · Percentile, 10,000 resamples, `_bootstrap_ci`. Recordings resampled, groups not.

**C200** Stats — "Wilcoxon signed-rank p, Holm-corrected across the three references" · code · **SUPPORTED** · `_holm` per model within a dataset. Not across the four datasets: 12 tests, corrected in threes. `_wilcoxon_p`'s docstring says exact; at n of hundreds scipy uses the normal approximation.

**C201** Stats — Holm 1979 citation · literature · **PARTIAL** · Content standard. bib DOI 10.2307/4615733 returns 404 at doi.org (2026-11-04), and Crossref has no such record. Remove it; cite the JSTOR stable URL if a link is wanted.

**C202** Stats — "Group-level and seed-level tests are reported as robustness only" · code · **UNSUPPORTED** · `04_` computes them and writes them to JSON only. No table, no macro, no sentence in the paper prints them; `paper_loso_significance.tex` and `_sensitivity.tex` are recording-level. As written the paper promises results it does not contain.

**C203** Stats — "window-level statistics are pseudoreplicated and descriptive" · logic · **SUPPORTED** · Correct.

**C204** Energy first — "The application is a low-power device" · design · **SUPPORTED** · Stated aim.

**C205** Energy first — "we fixed in advance how the two axes are read" · history · **PARTIAL** · Fixed 2026-11-03, before the grid (W:1950–55, W:2085–89), after pre-validation job 5314513 had already shown error on the dev folds and an arithmetic energy; the rule was then kept through job 5315335. Not preregistered, and the rule's wording was chosen knowing the likely direction.

**C206** Energy first — where the test does not separate the SNN-AE from a reference "the model with the lower E is the one we recommend" · logic · **WRONG** · (a) Two of the three references, PCA and Mean, have no E ("not costed"), so the rule is undefined for them. (b) Non-significance is treated as equivalence, with no margin. (c) No uncertainty on E. Point (a) has teeth: pricing PCA with the same model gives 158 nJ at k=16 vs 854 nJ for ANN-AE [59,16], and 903 nJ vs 1079 nJ at k=64 vs [65,64] (`energy.py`, my estimate, not a run), and PCA had lower test error than both AEs on the dev folds (QT "Outros achados" 2: EEG 0.183, FSDD 0.649). Taken seriously, "energy first" recommends PCA, and the paper's table excludes it from the rule.

**C207** Energy first — "where it does, we report what the lower-energy model gives up" · design · **CONDITIONAL** · Text to be written after the run; machinery only gives ratios.

**C208** Energy first — "Selection inside the search is unchanged by this rule" · code · **SUPPORTED** · W:2088.

### Design study

**C209** Design — "varying one factor at a time on the development fold" · history · **PARTIAL** · The table's rows change several things at once: row 5→6 changes the pooling point, the decoder neuron and T.

**C210** Design — "(1000 training windows, the search's most frequent winner, three seeds)" · history · **PARTIAL** · 1000 and 3 seeds: W:1401. "Most frequent winner" appears nowhere: W:1376 reports two winners tying on FSDD, and the twin ran direct input at the v_th/alpha of the *Poisson* winner (0.6/0.83). EEG: one winner of a small search (65/64 widths).

**C211** Design — "implemented in a PyTorch replica" · code · **SUPPORTED** · `snn_gap_ablation_torch.py`; read.

**C212** Design — "validated against the C++ harness (EEGMMIDB: earlier SNN-AE 0.592 vs. 0.593" · measured · **PARTIAL** · W:1523, no primary. C++ was a GA winner of 2 runs.

**C213** Design — "FSDD: ANN-AE 0.667 vs. 0.662)" · measured · **WRONG** · The pair compares different initializers: the twin's 0.667 is Kaiming (script default; Table), the C++ 0.662 came from the ANN builder, Xavier (W:1499–1501). Like-for-like is 0.654 vs 0.662, or vs 0.652 in format 5. It looks better than it is by accident and worse in fact; either way it is the wrong pair. Selective too: FSDD SNN validation agreement exists (W:1400, 0.646–0.656 vs 0.644–0.662) and is not quoted. The paper's own 0.827 vs 0.780 invites the question; the answer is genome mix, and direct/dense C++ 0.776–0.797 (W:1382) agrees better.

**C214** Table design caption — "mean of 3 seeds" · measured · **PARTIAL** · No dispersion; 0.724 vs 0.729 cannot be read.

**C215** Table design row — ANN LReLU 0.667/0.654/0.489/0.293 · measured · **PARTIAL** · Match W:1401, 1509, 1528. No surviving log. Twin keeps last weights with plain patience (script line 227), i.e. not the Section Training rule.

**C216** Table design row — ANN sigmoid 0.685/−/0.308/− · measured · **PARTIAL** · Same. Both encoder and decoder hidden layers.

**C217** Table design row — ANN clamp 0.743/−/0.547/− · measured · **PARTIAL** · Same. Both layers; zero gradient outside [0,1].

**C218** Table design row — SNN per-step 0.780/−/0.592/0.589 · measured · **PARTIAL** · Same.

**C219** Table design row — SNN time mean 0.734/−/0.454/− · measured · **PARTIAL** · Same.

**C220** Table design row — latent mean T=4 0.724/0.729/0.484/0.449 · measured · **PARTIAL** · Same.

**C221** Table design row — latent mean T=16 0.729/−/0.457/0.396 · measured · **PARTIAL** · Same. The caption "LReLU dec." is right.

**C222** Design (i) — "Two costs were not the spike" · logic · **PARTIAL** · Per-step is a coding-of-the-loss issue that exists because of spikes (C097). Decoder: C022.

**C223** Design (i) — per-step loss "asks each step, carrying at most one binary event per neuron, to reproduce the analog window alone" · logic · **SUPPORTED** · Correct, and it is an argument that the cost is about spikes.

**C224** Design (i) — "scoring the time mean removed 0.046 on FSDD" · math · **PARTIAL** · 0.780−0.734=0.046; the source numbers are C218–C219.

**C225** Design (i) — "A spiking decoder added error on top of the encoder's" · measured · **PARTIAL** · Not in the table (C022). Run the numbers in W:1420, or print row `enc_lif_only_rate`.

**C226** Design (i) — "averaging the latent and decoding once with leaky ReLUs is as good" · measured · **PARTIAL** · Under Xavier EEG 0.396 vs 0.392 (W:1536–37), FSDD 0.724 vs 0.724. Under Kaiming EEG 0.457 vs 0.403, not as good (W:1536–37).

**C227** Design (i) — "and runs the decoder once instead of T times" · code · **SUPPORTED** · Pool in the encoder.

**C228** Design (ii) — "the initializer decided the EEG comparison" · measured · **PARTIAL** · As C023.

**C229** Design (ii) — control "stopped at 0.489 (0.479 with a 200-epoch cap, so not for lack of epochs)" · measured · **PARTIAL** · W:1511, no primary.

**C230** Design (ii) — "the SNN-AE looked equal to it (0.484)" · measured · **PARTIAL** · Table.

**C231** Design (ii) — "with Xavier in both arms the control reaches 0.293 and the SNN-AE 0.449" · measured · **PARTIAL** · Table.

**C232** Design (ii) — "Xavier is the better or equal choice for each arm on both datasets" · measured · **PARTIAL** · FSDD SNN: Xavier 0.729 vs Kaiming 0.724, worse by 0.005 with unknown spread. C++ FSDD favours Xavier (0.768 vs 0.786, W:1562–63). "Equal" is unproven.

**C233** Design (ii) — "so the matched comparison is also a best-against-best one" · logic · **PARTIAL** · Best of two initializers tried at one genome, not "best". Under Xavier the search then runs; there is no re-test at other widths or gains.

**C234** Design (iii) — "What remains is saturation" · logic · **UNSUPPORTED** · C026.

**C235** Design (iii) — hard clamp "lands among the spiking variants (FSDD 0.743, against 0.734 for the SNN trained on the time mean; EEG 0.547, against 0.592 for the per-step SNN)" · logic · **WRONG** · Numbers match the table; the comparison is not valid. The comparator changes between datasets. On EEG the like-for-like SNN (time mean) is 0.454, so the clamp is 0.093 worse. The chosen comparator on EEG carries the per-step penalty that finding (i) says is not the spike, and a clamp ANN does not pay that penalty. And all of it is Kaiming, the regime finding (ii) calls handicapped.

**C236** Design (iii) — "while a sigmoid, bounded but smooth, stays near the control (FSDD 0.685 vs. 0.667; EEG 0.308, near the Xavier control's 0.293)" · logic · **WRONG** · Cross-initializer comparison. The sigmoid ran under Kaiming, whose control is 0.489. Within Kaiming the sigmoid *beats* the leaky-ReLU control by 0.181. That shows activation × initializer interaction, not "a smooth bound is harmless". No Xavier sigmoid exists.

**C237** Design (iii) — "A reason, argued rather than measured" · logic · **SUPPORTED** · Honest flag, and the abstract ignores it.

**C238** Design (iii) — "under direct input an LIF neuron sees the same current I at every step" · code · **PARTIAL** · First hidden layer under direct input only. Deeper LIF layers see spike-driven currents; Poisson, latency and recurrent inputs vary per step.

**C239** Design (iii) — "with a hard reset it never fires for $I<V_{th}(1-\beta)$" · math · **SUPPORTED** · $V[t]=I(1-\beta^t)/(1-\beta)<I/(1-\beta)\le V_{th}$. Also true at equality (0 spikes in 2000 steps, `checks.py`).

**C240** Design (iii) — "and fires at every step for $I\geq V_{th}$" · math/code · **WRONG** · Eq. lif is strict. At $I=V_{th}$, $V[1]=V_{th}$ does not fire; the neuron fires every second step (`checks.py`: [0,1,0,1,...]). Holds for $I>V_{th}$. W:1424 had it right.

**C241** Design (iii) — "-- a clipped transfer function" · math · **PARTIAL** · Clipped at the ends, but between them a staircase. Time mean under constant I at β=0.9, V_th=1: T=4 gives levels {0, 0.25, 0.5, 1}; T=16, 8 levels; T=32, 11 levels (`checks.py`). A clamp is continuous. That quantization is the spike. Under Poisson input there is no such function at all.

**C242** Design — "T mattered on EEG (0.449 at T=4, 0.396 at T=16)" · measured · **PARTIAL** · Table, and it is evidence against pure saturation: a clip does not depend on T, a quantizer does.

**C243** Design — "but not on FSDD (0.717 to 0.729 for T from 1 to 16)" · measured · **PARTIAL** · W:1459–62, four T values, Kaiming.

**C244** Design — "which is why T is searched rather than fixed" · history · **SUPPORTED** · W:1597–1608.

**C245** Design — "Every choice above came out the same on the fold's validation windows" · measured · **PARTIAL** · W:1547–49 documents latent/output, Xavier/Kaiming SNN, T. Not "every": no validation figure for the per-step, decoder, ANN or clamp comparisons.

**C246** Design — "(e.g. EEG, Xavier: 0.425 vs. 0.372 for T=4 vs. 16)" · measured · **PARTIAL** · W:1549.

**C247** Design — "but the test windows were seen, hence the development-fold rule" · history · **SUPPORTED** · Honest.

**C248** Design — "A check of the adopted design in the C++ harness on the same fold (two runs, small search) moved the SNN-AE from 0.827 to 0.768 on FSDD and from 0.593 to 0.532 on EEG, against controls of 0.652 and 0.302" · measured · **PARTIAL** · Match W:1563–67. But the "adopted design" of that check is format 5: T fixed at 4, no armed stopping rule, no best-weight restore, seeds not reaching shuffle, population 4×2. Undisclosed. No primary file. Newer dev figures (format 7, SNN-AE 0.713/0.407, ANN-AE 0.661/0.291) exist in QT and could replace these once 5316016 reports.

### Pre-validation and T1

**C249** Pre — "The submitted version also searched LSTM, GRU and Transformer autoencoders" · history · **PARTIAL** · True, but this phrase reveals a prior submission to reviewers and means nothing to a new reader. Remove.

**C250** Pre — they "cannot isolate what the spike costs" · logic · **SUPPORTED** · C038.

**C251** Pre — "their searches dominated the grid's compute" · measured · **PARTIAL** · W:1795–97: 2026-09-23 calibration, not re-measured, before T search and 1000 windows. Current figure CONDITIONAL on 5316016.

**C252** Pre — "Before leaving them out we measured them once" · history · **PARTIAL** · Wiki order is reversed: "The author removed them ... 'they were slow' is a sentence, not evidence ... The pre-validation produces them" (W:1800–02). Post-hoc justification, then. And not "once" (C018).

**C253** Pre — "on the development folds" · config · **SUPPORTED** · Profile datasets, `--plan` refusal.

**C254** Pre — LSTM/GRU: "stack L∈[1,3] recurrent layers of width H∈[8,256] (floored at d), project the last state to d and decode by repeating a projection of z over time" · code · **SUPPORTED** · Profile; `require_hidden_range`; `GRUAutoencoder.hpp:12`. Latent has a tanh, unstated.

**C255** Pre — Transformer description · code · **SUPPORTED** · `TransformerAutoencoder.hpp:8-15`, profile. Post-norm default.

**C256** Pre — "All three read the window as S=32 frames of 8 samples, so only the direct code is defined for them" · code · **SUPPORTED** · `recurrent_time_steps=1`, `recurrent_encoding_choices`.

**C257** Pre — "Every family, the SNN-AE included, ran one NSGA-II of population 4 for 2 generations (12 evaluations, front re-scored over 2 seeds), 2 runs (seeds 42–43)" · config/math · **SUPPORTED** · Profile; 4×3=12. Execution CONDITIONAL.

**C258** Pre — "with the windows, training and selection of Sections search–training" · code · **SUPPORTED** · `PrevalidationDiffersFromTheGridOnlyWhereDeclared` (not run).

**C259** Pre — "each family in its own process on one CPU thread" · code · **SUPPORTED** · sbatch header: "8 one-thread processes", one node, 16 CPUs.

**C260** Pre — E "applies the counts of Table to every matrix product of a cell or an attention block at each of the S positions" · code · **SUPPORTED** · `recurrent_ae_energy`, `transformer_ae_energy`.

**C261** Pre — "gate, softmax and normalization nonlinearities cost an add and a multiply each, below an exact exponential, which favours them" · code/logic · **SUPPORTED** · `add_nonlinearity`.

**C262** Pre — Table reports "the cost of training (the fold's wall time) and of processing data after training" · code · **CONDITIONAL** · Report script.

**C263** Pre — `\preCostVerdict`, `\preQualityVerdict` · measured · **CONDITIONAL** · Placeholders. If it repeats format 7, PCA beats all of them.

**C264** Pre — "The reduced budget can change any family's winner ... the table describes the winners found here, not the families' best" · logic · **SUPPORTED** · Good disclosure. Should add: a family stuck on the plateau is priced as a network that does not reconstruct (QT Q2).

**C265** T1 — "A Transformer whose layer normalization follows each residual sum, as ours does" · code · **SUPPORTED** · `TransformerEncoderBlock.hpp:9-11` (per QT), `norm_first` default false.

**C266** T1 — "can need a learning-rate warm-up to train at a fixed Adam rate" · literature · **SUPPORTED** · Xiong abstract, §3.2 with Adam lr_max 1e-3. "Can need" is careful and correct. Xiong's tasks are deep NMT and BERT, not 1–3 block models.

**C267** T1 — "one that normalizes before each sub-layer does not [xiong]" · literature · **PARTIAL** · Xiong: "the learning rate warm-up stage can be safely removed" in their tasks. As a universal it is too strong, and our own arm C mostly stays on the plateau.

**C268** T1 — "We therefore retrained the Transformer-AE winners of both development folds" · history · **PARTIAL** · Winners of job 5315335, format 7 (`WINNERS`, `P/05_...py:70`), typed in. Once 5316016 reports, the paper's winners will differ (seed fix, format-9 cost). The sentence then refers to networks no longer in the table.

**C269** T1 — "2 seeds each" · code · **SUPPORTED** · 2 genomes × 2 seeds × {GA evaluation, final fit} = 8 per arm and dataset.

**C270** T1 — "linear warm-up over 4000 steps" · code · **SUPPORTED** · Profile B `lr_warmup_steps=4000`.

**C271** T1 — "and, separately, with pre-norm blocks closed by a final layer normalization" · code · **SUPPORTED** · Profile C; Xiong §4.

**C272** T1 `\tOneVerdict` FSDD: 1, 1, 0 of 8 armed · measured · **SUPPORTED** · Recomputed from `t1_summary.csv`.

**C273** T1 FSDD mean test 0.916, 0.931, 0.935; mean predictor 0.996; best 0.867 warm-up · measured · **SUPPORTED** · Recomputed (0.9162, 0.9305, 0.9350).

**C274** T1 EEG: 0, 1, 1 of 8 armed · measured · **SUPPORTED** · Recomputed. The control's arm is a GA evaluation.

**C275** T1 EEG mean test 0.993, 0.921, 0.992; mean predictor 1.001; best 0.792 pre-norm · measured · **SUPPORTED** · Recomputed (0.9934, 0.9210, 0.9925).

**C276** T1 — "left the mean-predictor plateau" = armed at 10% · logic · **PARTIAL** · A definition chosen for stopping, not for this test. An arm at 0.921 mean test MSE with no armed s43 fit shows sub-threshold learning (test 0.924). Definition unstated in the sentence.

**C277** T1 — implicit: T1 supports the removal · logic · **PARTIAL** · The paper draws no conclusion, which is right, but the reader will. Evidence: 4 of 48 armed (1, 1, 2 per arm over two datasets), no arm distinguishable from another (notes agree). Limits: post-hoc test motivated by a seen failure; the a-priori criterion "clearly below L0" was unquantified (QT line 477 admits it); EEG added after the first run; batch size 1 and shared lr untested; 30-epoch cap; provenance `379a8570+dirty`. Supports "these two remedies did not reliably fix it", not "Transformers cannot do this".

**C278** Table pre caption — wall time definition; "×SNN: its ratio to the SNN-AE's, whose fold also trains the ANN-AE" · code · **SUPPORTED** · W:1843, 1863–65.

### Results

**C279** Results — banner: tables are placeholders · status · **SUPPORTED** · `\ifplaceholder`.

**C280** Results — overview, significance, Pareto, energy, selection tables exist as described · code · **CONDITIONAL** · All "--"/nan.

**C281** Results — `\energyVerdict` · measured · **CONDITIONAL** · "[has not run yet]".

**C282** Table energy caption — columns · code · **CONDITIONAL** · Machinery matches caption.

**C283** Table overview caption — "PCA and Mean are fitted once (no seeds, no std) and are not costed" · code · **SUPPORTED** · True, and see C206.

**C284** Table sig caption — "positive: the SNN-AE errs more; $^*$: Holm-corrected Wilcoxon p<0.05" · code · **SUPPORTED** · `write_overview_tex`.

**C285** Fig. pareto caption — "the dashed line is PCA with k=d" · code · **CONDITIONAL** · `\paretoPcaRow`. A horizontal line only because PCA has no E.

**C286** Table selection caption — modal encoding, median T, etc. · code · **CONDITIONAL** · `02_`.

**C287** Table sensitivity caption · code · **CONDITIONAL** · `write_sensitivity_tex`.

**C288** Reading guide — "binary inputs turn the first layer's MACs into adds, but every step adds a read and a write of every potential" · code · **SUPPORTED** · `add_spiking_fc`.

**C289** Reading guide — "a model that does not beat PCA has not shown that its nonlinearity helps on that dataset" · logic · **SUPPORTED** · Sound. On dev folds neither AE beat PCA.

**C290** Reading guide — "Orderings are asserted only where the Holm-corrected p is below 0.05 and the interval excludes zero" · design · **CONDITIONAL** · A commitment; the "Energy first" rule is its converse (C206).

### Discussion

**C291** Disc — "The design study already constrains what the main run can show" · logic · **PARTIAL** · It motivates hypotheses. The main run has a searched T, Poisson and latency, other widths.

**C292** Disc — "The spike's cost in this autoencoder is not a property of discrete events as such" · logic · **UNSUPPORTED** · C026.

**C293** Disc — "a non-spiking network with the same clipped transfer function pays it" · math/logic · **WRONG** · Not the same function (C241); both-layer clamp, Kaiming, wrong comparator (C235).

**C294** Disc — "and a smooth bounded one does not" · measured · **UNSUPPORTED** · Cross-initializer (C236). FSDD: +0.018 under Kaiming, one comparison.

**C295** Disc — remedies "are therefore the levers to test" · logic · **PARTIAL** · Reasonable hypotheses; "therefore" leans on C292.

**C296** Disc — "soft resets, which made things worse here" · measured · **PARTIAL** · 0.799 vs 0.780, FSDD, format-4 network, Kaiming (W:1416). Not in Table. No primary.

**C297** Disc — "Longer simulations helped on EEG only (Table)" · measured · **PARTIAL** · Table; 3 seeds, one genome, direct.

**C298** Disc — "and cost arithmetic in proportion to T and in memory traffic" · logic · **PARTIAL** · Not the first layer of a direct input, by the paper's own count.

**C299** Disc — "binary inputs save the first layer's multiplies while every step pays for the potentials" · code · **SUPPORTED** · C288.

**C300** Disc — "whether the SNN-AE ends up cheaper than its control depends on the encoding, T and spike rate the search selects, not on spiking alone" · logic · **SUPPORTED** · QT T2 illustrates it.

**C301** Lim (i) — "Recordings are nested in 6, 60, 109 and 14 groups (5 FSDD speakers on the headline folds)" · measured · **SUPPORTED** · C183–C186.

**C302** Lim (i) — "The recording-level test treats recordings as independent, so it speaks to new recordings of these speakers and subjects, not to new ones" · logic · **SUPPORTED** · Correct, and it is a severe limit for a study of leave-one-speaker generalization: the primary test does not address the population question.

**C303** Lim (i) — "the group-level tests that would are reported as robustness only" · code · **UNSUPPORTED** · C202. Also "would" is generous: 04_'s own note says "NOT population inference".

**C304** Lim (i) — "with 5 speakers the exact Wilcoxon p cannot fall below 0.0625" · math · **SUPPORTED** · 2/2⁵; scipy exact gives 0.0625 (`checks.py`). If Holm-corrected across three references, 0.1875.

**C305** Lim (ii) — "EEG is read from one channel and every window has 256 samples; results may not transfer" · logic · **SUPPORTED** · Durations differ: 32 ms, 1.6 s, 0.5 s.

**C306** Lim (iii) — "E models an accelerator with local SRAMs at 45 nm and leaves out off-chip traffic, leakage and control" · code · **SUPPORTED** · Yes.

**C307** Lim (iii) — "a real chip's consumption, and whether it can exploit sparsity, decide the final number" · logic · **SUPPORTED** · Yes.

**C308** Lim (iii) — "The repricings bound how much the conclusion depends on the memory model" · logic · **PARTIAL** · They probe it. Three SRAM sizes do not bound word width, DRAM, or FIFO sizing (Lemaire's 1000-element FIFOs matter).

**C309** Lim (iv) — search budget; "which favours the SNN-AE over its control; a larger budget could change the winner" · logic · **SUPPORTED** · C110; good disclosure.

**C310** Lim (v) — "The ANN-AE always reads the analog window, whereas the SNN-AE may read spikes" · code · **SUPPORTED** · Yes. This is the honest form of C007, and contradicts it.

**C311** Lim (v) — "the information available is the same, its code is not" · logic · **WRONG** · A T-step Bernoulli train is a noisy sample of the window; latency quantizes each value to T levels; both lose the window's min and max; `recurrent` on direct input silences every negative sample. The information is reduced, and this is part of the measured gap.

**C312** Lim (vi) — "d follows prior autoencoder practice for each domain, not a sweep on these datasets" · literature · **PARTIAL** · Profile: "no eegmmidb-specific AE paper found; extrapolated from Khan et al. 2023's ... CHB-MIT". No citation in the paper.

**C313** Lim (vii) — "on the development fold most EEG runs were still improving at the cap" · measured · **PARTIAL** · W:1544 ("Nearly every"), earlier stopping rule, no primary.

**C314** Lim (viii) — "searched with 12 evaluations on two development folds only, so the conclusions concern spiking against its matched control" · logic · **SUPPORTED** · Correct scope.

### Conclusion

**C315** Concl — "We compared a spiking autoencoder with a matched non-spiking control and PCA on two speech and two EEG datasets" · status · **CONDITIONAL** · Past tense for a run that has not happened.

**C316** Concl — "with every choice made on validation groups" · history · **WRONG** · Section design: "the test windows were seen". Design choices were made on dev test windows. Direct contradiction within the paper.

**C317** Concl — "and the development fold kept out of the headline numbers" · code · **PARTIAL** · C194.

**C318** Concl — sequence AEs "were measured in a pre-validation on the same folds" · status · **CONDITIONAL** · Job 5316016.

**C319** Concl — "Matching the control exposed two confounds that would have changed the conclusion" · logic · **PARTIAL** · Matching exposed the initializer mismatch. "Would have changed the conclusion" holds for EEG only, on one fold, and there is no conclusion yet.

**C320** Concl — "a per-step loss and a spiking decoder that inflated the gap" · measured · **PARTIAL** · C022.

**C321** Concl — "an initializer that, shared but poorly chosen, made it vanish on EEG" · measured · **PARTIAL** · C023.

**C322** Concl — "and located the remaining cost in the saturation of the LIF transfer function" · logic · **UNSUPPORTED** · C026.

**C323** Concl — "The harness, profile and analysis scripts are released [furlan2026doi]" · status · **PARTIAL** · doi 10.5281/zenodo.20074785 resolves to record 20074786; I could not read its contents. The bib entry is anonymized while the title block names the author, so the double-blind state is inconsistent. A release predating format 9 would not match this paper; the twin must be in it.

### Bibliography (implicit claims: the entries are correct)

**C324** Bib — verified by Crossref or the publisher, no defect: schalk2004, detti2020, eshraghian2023, deb2002, maass1997, gerstner2002, pfeiffer2018, neftci2019, zenke2018, zenke2021, werbos1990, thorpe1998rank (year 2001, key misleading), comsa2020, comsa2021, kamata2022, lemaire2022 (year 2023 as Crossref), dampfhoffer2023, cho2014gru, goldberger2000, prechelt1998, horowitz2014 · literature · **SUPPORTED** · Titles, authors, volume, pages match. Crossref abbreviates eshraghian's author list to nine, bib uses "and others".

**C325** Bib — hochreiter1997long, greff2017, `J{"u}rgen`; jackson2018fsdd, `Zo{"e}` · literature · **WRONG** · Missing backslash: literal `{"u}`, hidden only because the style prints initials. Jackson's first name is Zohar.

**C326** Bib — holm1979simple DOI · literature · **WRONG** · 404, C201.

**C327** Bib — zhou2023spikformer as ICLR 2023 · literature · **SUPPORTED** · iclr.cc virtual poster 12139.

**C328** Bib — yan2024reconsidering as arXiv · literature · **PARTIAL** · arXiv now says "accepted by TCAD". Update once volume and pages are known.

**C329** Bib — not checked against a primary record: hochreiter (metadata ok), sutskever, kingma, vaswani, xiong (arXiv ok), ba, chung, cawley (JMLR page ok), glorot, he2015, goodfellow · literature · **PARTIAL** · Standard entries, consistent with what I know; no volume/page check for the NeurIPS and PMLR items except Xiong.

**C330** Bib — AudioMNIST and PhysioNet dataset records missing · literature · **UNSUPPORTED** · C175–C176. Becker et al., J. Franklin Inst. 361(1), 418 (from a search snippet; check).

---

## 2. Pressure points, answered

**1. "Only the encoder's hidden neuron differs" / "the gap is the cost of spiking".** Not earned as written.
Between SNN-AE and ANN-AE, for a Poisson + conv1d winner: encoder hidden neuron, T steps, time average, spike code, input filter, and three extra trained scalars per layer. Only a direct + dense winner comes close to "one thing", and even there LIF-over-T plus the time mean is a package.
Causal reading that survives: *randomized-seed, paired comparison of the selected spiking pipeline against its non-spiking twin*. That is valuable; it is not "the spike". A clean decomposition would need a second control, e.g. ANN-AE reading the Poisson/latency code (encoder-only effect), which the harness does not run.

**2. "Conservative" and "favours X".** Directions, by sentence:

| Choice | Moves E of | Favours | Paper says |
|---|---|---|---|
| 8 KB floor | SNN potentials mostly | ANN | "against the conclusion" |
| threshold add | SNN | ANN | same |
| encoder charged | SNN | ANN | same |
| 64-bit price for 32-bit word | everyone, by memory share | ANN if SNN share is higher | unstated |
| first layer once | direct SNN | SNN | "favours the SNN-AE" |
| fp32 not int32 ops | everyone | unclear, depends on each model's MAC share | unstated |
| energy not in selection | SNN (picked for MSE) | ANN | stated, not as a bias |
| search budget | MSE | SNN | "conservative" |

The sentence "the one that works against the conclusion it could support" holds only if that conclusion is "the SNN saves energy". Dev data say the reverse, and three of the four named choices then push *with* the result. Counting the first layer once does favour the SNN (14.4× → 1.53×), correct. "A gap in the ANN-AE's favour is conservative" is correct about search only. Proposed rewording: list the choices with direction; claim no global conservatism.

**3. "The cost is the saturation ... not the spike".** Does not follow. Five independent problems:
(a) Clamp and sigmoid ran only under Kaiming, the regime the paper calls handicapped. There is no Xavier clamp, so "what remains" under the adopted design was never compared.
(b) Clamp and sigmoid sit in both encoder and decoder; the adopted SNN spikes in the encoder only.
(c) The comparators are cherry-picked by dataset (C235). Like-for-like on EEG, the clamp misses by 0.093.
(d) The sigmoid comparison crosses initializers (C236).
(e) The mechanism does not fit the data: T improves EEG by 0.053 although a clip is T-independent; the LIF time mean is a coarse staircase (4 levels at T=4); clamp has dead gradients while the LIF has a surrogate (`ann_clipste` exists in the twin and has no recorded result); and dev winners are Poisson, where the constant-current argument does not apply.
What the evidence supports: "on FSDD, under Kaiming, a two-layer hard clamp matched a two-layer spiking network trained on the time mean, which suggests bounded, non-smooth transfer as one candidate; this was not tested on the adopted design."

**4. Abstract and conclusion vs Section design.** Section design: "argued rather than measured", "(iii) What remains is saturation". Abstract, Discussion, Conclusion: "the cost is the saturation ... not the spike", "not a property of discrete events as such", "located". Escalation is real; the conclusion claims more than the section. Also Conclusion "every choice made on validation groups" contradicts "the test windows were seen".

**5. Design-study numbers and the format-9 seed fix.** The PyTorch twin has its own seeded init and shuffle (`torch.manual_seed`, `default_rng(seed)`), so the C++ seed bug did not touch Table tab:design. What the fix does touch: the C++ anchors (0.593, 0.662, 0.827, 0.768, 0.532, 0.652, 0.302), though for SNN/ANN only the window order was fixed; init was already seeded. Larger and undisclosed staleness: the twin keeps last weights under plain patience, while Section Training describes armed patience with best-weight restore; the C++ "adopted design" check is format 5 with T=4. None of this is disclosed. The numbers remain useful as history, but they do not describe the protocol the paper states. Raw logs have not survived; only wiki transcriptions do.

**6. PyTorch-replica validation.** Two anchors of 3 seeds vs 2 runs, one of them across initializers (C213), on selected architectures, on development folds, test split included. No dispersion, no test of equivalence. "Validated" should be "agreed at two points". Stronger agreement exists and should be used (C213). `compare_ann_ae_torch.py` is a second twin worth citing.

**7. Selection.**
- "Chosen by validation MSE alone": true within the front (C127).
- Re-scoring: mean includes the selected-for first score (C118).
- PCA: capped train ∪ val, 1150 windows, not "all non-test"; more data than the AEs, in PCA's favour (C112). No test data.
- Test data in selection: none in the grid. `overall_winner` compares families on validation only. Test data did inform the design (fold 0), and fold 0's validation speaker is headline fold 1's test speaker (C194). This is the leakage reviewers will find.
- Pre-validation removal decision used test MSE on dev folds; disclosed; order of decision and measurement is not (C252).

**8. Statistics.** Recording-level unit is pseudoreplicated relative to speakers; disclosed (C302) but it is the primary estimand. Group-level results are promised and not printed (C202). Holm is within dataset, three references; 12 tests overall; nothing across datasets. Bootstrap resamples recordings, ignores seeds (averaged) and groups. "Energy first" takes absence of evidence as equivalence, with no margin, no E uncertainty, and undefined for two of three references (C206). Rule is unpreregistered (C205). FSDD group-level is structurally unable to reach 0.05 (C304), so a reader sees only the pseudoreplicated test there.

**9. T1.** Post-hoc, prompted by a seen failure. No quantified criterion. EEG added after the first run; the first run's "warm-up helps" was a seed artifact (QT). n: 8 per arm per dataset from 2 genomes, 2 seeds, two of which are train-only evaluations. Winners from format 7, stale once 5316016 reports. "Can need" is right; "does not" is too strong. The verdict sentence reports numbers and draws no conclusion; it should state its limit in one clause.

**10. Unasked findings, added by me.**
- EEGMMIDB 128 Hz files (C185). Real, silent, small.
- EEG band-pass/notch undisclosed (C077).
- PCA, priced by the same model, is cheaper and better on dev folds (C206). The biggest strategic risk: an "energy first" paper that does not cost its best baseline.
- `L∈[1,4]` unreachable at 1 (C120), Eq. 2 missing the reset factor (C092), V_th not clamped (C088), Table errors (C170, C173).

---

## 3. Counts

Total claims: 330.

| Status | Count |
|---|---|
| SUPPORTED | 168 |
| PARTIAL | 111 |
| UNSUPPORTED | 11 |
| WRONG | 22 |
| CONDITIONAL | 18 |

A large PARTIAL count is expected: every design-study number without a surviving log is PARTIAL by rule, and many PARTIAL items are a wording fix away from SUPPORTED.

---

## 4. Weakest claims, ranked

1. **C026, C322, C292** "the cost is the saturation ... not the spike", repeated at three levels. Five independent flaws (pressure point 3).
2. **C206** "Energy first" recommendation rule: equates non-significance with equivalence, undefined for PCA/Mean, and PCA is likely cheaper and better.
3. **C007, C045, C105** "only the neuron differs". Contradicted by Fig. 1 and Limitation (v).
4. **C194** dev-fold exclusion misses fold 0's validation speaker, the test speaker of fold 1.
5. **C235, C236** clamp and sigmoid comparisons: shifting comparator, crossed initializers.
6. **C316** "every choice made on validation groups" contradicts Section design.
7. **C202, C303** group-level results promised, not printed.
8. **C151** "works against the conclusion it could support": direction undefined and mostly reversed.
9. **C144, C053** "applied unchanged": Lemaire says the opposite about repeated input.
10. **C311** "the information available is the same".
11. **C003** "usually compared ... differ in more than the neuron": own citations contradict it.
12. **C213** replica validation pairs different initializers.
13. **C092, C170, C173, C125, C120, C088, C240** factual errors, each a one-line fix.
14. **C185, C077, C174** undisclosed data facts.
15. **C018, C252** "measured once", "before leaving them out".

---

## 5. Proposed text changes (PROPOSALS, for the user's approval)

**P01 (C026, abstract).** OLD: "and that what remains is reproduced by a non-spiking network with a hard-clipped activation: the cost is the saturation of the neuron's transfer function, not the spike."
NEW: "and that, in a preliminary test, a non-spiking network with a hard-clipped activation approached a spiking one on speech, which makes a bounded, non-smooth transfer function a candidate explanation of what remains; it was not tested on the adopted design."

**P02 (C003).** OLD: "Spiking autoencoders are usually compared with non-spiking networks that differ from them in more than the neuron, and mostly on images."
NEW: "Spiking autoencoders have been compared with non-spiking networks of matching size mostly on images; on signals they are usually compared with recurrent or attention models, which differ in more than the neuron." (Second half needs a citation, or delete it.)

**P03 (C007, C045).** OLD (abstract): "that copies its selected widths, initializer and training and replaces only the leaky integrate-and-fire neuron by a leaky ReLU"
NEW: "that copies its selected widths, initializer and training, replaces the leaky integrate-and-fire neuron by a leaky ReLU, and reads the analog window once, without a spike code, input transform or time axis"
OLD (contribution 1): "and differs from it only in the encoder's hidden neuron, so the gap between them is the cost of spiking."
NEW: "and differs from it in the encoder's hidden neuron and in everything spiking requires around it (the time axis, the spike code and the input transform), so the gap between them is the cost of the selected spiking pipeline."

**P04 (C105).** OLD: "only the encoder's hidden neuron differs (red: the parts that exist only in the spiking network)."
NEW: "red: the parts that exist only in the spiking network; a selected spike code or input transform (not drawn) precedes its first layer."

**P05 (C111).** OLD: "The SNN-AE therefore receives the larger search budget, which favours it, and a gap in the ANN-AE's favour is conservative."
NEW: "The SNN-AE therefore receives the larger search budget, so as far as search is concerned a gap in the ANN-AE's favour understates what a searched control could reach."

**P06 (C144, C053).** OLD: "so it is the analytical model of Lemaire et al. [...], applied unchanged to every model:"
NEW: "so it follows the analytical model of Lemaire et al. [...], extended as listed below and applied alike to every model:"
OLD (contribution 3): "applied unchanged to every model, the SNN's counts taken from its executed forward pass"
NEW: "applied alike to every model, the SNN's spike counts taken from its executed forward pass"

**P07 (C148, C146).** After "32 KB and 1 MB." add: "These are Horowitz's figures for a 64-bit cache access; we apply them to one 32-bit word, which overprices every access. Lemaire et al. price operations with 32-bit integer figures; we use the floating-point ones, since weights are 32-bit floats."

**P08 (C151).** OLD: "Where the model leaves a choice, we take the one that works against the conclusion it could support."
NEW: "The model leaves several choices. The first four below count against the SNN-AE; the last favours it. We state each direction so that a conclusion can be read against it."

**P09 (C092).** OLD: `+ \beta\,\delta_l[t+1]`
NEW: `+ \beta\,(1-s_l[t])\odot\delta_l[t+1]` and add "(the reset is detached from the gradient, so no error flows through a step that fired)."

**P10 (C088).** OLD: "$R$, $C$ and $V_{th}$ are trained per layer (clamped above $10^{-6}$)"
NEW: "$R$, $C$ and $V_{th}$ are trained per layer ($R$ and $C$ clamped above $10^{-6}$)"

**P11 (C120).** OLD: "depth $L\in[\cfgSnnDepth]$" NEW: "$L-1\in[1,3]$ hidden layers" (needs a macro derived from `max_layers`).

**P12 (C125).** OLD: "a spike-blind MAC count in which $T$ multiplies every encoder layer;"
NEW: "a spike-blind MAC count in which $T$ multiplies every encoder layer that runs at every step (a direct-coded first layer counts once, as in Section~\ref{sec:cost});"

**P13 (C112).** OLD: "fitted on all non-test windows of the fold"
NEW: "fitted on the fold's capped training and validation windows"

**P14 (C140).** OLD: "retrained on training $\cup$ validation groups, early-stopped on a recording-disjoint slice of the training groups"
NEW: "retrained on the training and validation groups minus a recording-disjoint slice of the training groups, which serves for early stopping"

**P15 (C170).** Table cost, Write output, spike cell: OLD "--" NEW "$R\,n_{\mathrm{out}}$ if no LIF follows". Note: "A layer that feeds an LIF writes no output of its own" add ", except the stored current of a direct-coded first layer."

**P16 (C173).** OLD: "Mean over $T$: one add per input spike, one multiply per output."
NEW: "Mean over $T$ of the analog latent: an add and a read per value, a MAC and a write per output."

**P17 (C077).** After "z-scored on its own statistics" add for EEG, in Experimental Setup: "EEG recordings are band-passed at 0.5--40 Hz and notched at the mains frequency (50 Hz Siena, 60 Hz EEGMMIDB) before windowing."

**P18 (C174, C175, C185, C330).** OLD: "FSDD [..] and AudioMNIST are spoken digits at 8 kHz" NEW: "FSDD [..] and AudioMNIST [becker] are spoken digits, the latter resampled offline from 48 to 8 kHz". Add a footnote or Limitation: "Twelve runs each of EEGMMIDB subjects S088, S092 and S100 are sampled at 128 Hz and were processed as 160 Hz." Better: fix the loader to read the rate from the header before the grid. Fix bib `J{\"u}rgen`, `Zohar`, drop Holm's DOI, add dataset citations.

**P19 (C194, C192, C316, C317).** OLD (Setup): "Those folds are therefore excluded from every headline number:"
NEW: "Those folds' test groups are therefore excluded from every headline number, and the sensitivity analysis also drops the next fold, whose test group was the development fold's validation group:" (needs a code change in `04_`.)
OLD (Conclusion): "with every choice made on validation groups and the development fold kept out of the headline numbers"
NEW: "with every architecture chosen on validation groups, design choices made on development folds, and those folds kept out of the headline numbers"

**P20 (C200, C202, C303).** Either print a group-level column in tab:sig (one page is free), or OLD: "Group-level and seed-level tests are reported as robustness only" NEW: "Group-level and seed-level tests are in the released analysis output". Prefer the table. Add: "Holm correction is per dataset; datasets are not pooled or corrected against each other."

**P21 (C206, C205).** OLD: "we fixed in advance how the two axes are read. Where the recording-level test does not separate the SNN-AE from a reference (interval including zero or Holm-corrected $p\geq0.05$), the model with the lower $E$ is the one we recommend; where it does, we report what the lower-energy model gives up"
NEW: "we fixed before the main run, after seeing the development folds, how the two axes are read. For SNN-AE against ANN-AE, where the interval lies inside $\pm\Delta$ (a margin to be fixed before the run) the model with the lower $E$ is the one we recommend; otherwise we report what the lower-energy model gives up. PCA is priced by the same model as a linear encoder and decoder." (Requires choosing Δ and adding a PCA row to tab:energy.)

**P22 (C235, C236, C234, C241).** OLD: "\emph{(iii) What remains is saturation.} A non-spiking network with a hard clamp lands among the spiking variants (FSDD 0.743, against 0.734 for the SNN trained on the time mean; EEG 0.547, against 0.592 for the per-step SNN), while a sigmoid, bounded but smooth, stays near the control (FSDD 0.685 vs.\ 0.667; EEG 0.308, near the Xavier control's 0.293). A reason, argued rather than measured: ..."
NEW: "\emph{(iii) What remains is open.} Under fan-in scaling, with both hidden layers changed, a hard clamp scored close to the time-mean SNN on FSDD (0.743, 0.734) but not on EEG (0.547, 0.454), and a sigmoid scored 0.685 and 0.308, against leaky-ReLU controls of 0.667 and 0.489. These runs precede the adopted design and were not repeated under Xavier. A hypothesis, argued rather than measured: under direct input an LIF neuron sees the same current $I$ at every step, so with a hard reset it never fires for $I\le V_{th}(1-\beta)$ and fires at every step for $I>V_{th}$; between them its time mean is a staircase with few levels at small $T$ (four at $T=4$), which would explain why $T$ matters on EEG." Better: run the encoder-only Xavier clamp first, then write.

**P23 (C292–C294).** OLD: "The spike's cost in this autoencoder is not a property of discrete events as such: a non-spiking network with the same clipped transfer function pays it, and a smooth bounded one does not. Remedies that act on the transfer function -- ... -- are therefore the levers to test."
NEW: "The design study does not separate the spike's discreteness from the saturation it brings. Remedies that act on the transfer function -- ... -- are candidates to test."

**P24 (C322, C319).** OLD: "-- and located the remaining cost in the saturation of the LIF transfer function."
NEW: "and left the remaining cost to be explained; saturation and quantization of the LIF's output are the leading candidates."

**P25 (C210, C212, C213).** OLD: "(1000 training windows, the search's most frequent winner, three seeds). The variants were implemented in a PyTorch replica of the network, validated against the C++ harness (EEGMMIDB: earlier SNN-AE 0.592 vs.\ 0.593; FSDD: ANN-AE 0.667 vs.\ 0.662)."
NEW: "(1000 training windows, one genome per dataset from an earlier small search, direct input, three seeds). The variants were implemented in a PyTorch replica of the network, whose error agreed with the C++ harness at two points (EEGMMIDB: earlier SNN-AE 0.592 vs.\ 0.593; FSDD with Xavier: ANN-AE 0.654 vs.\ 0.652, the C++ widths being the search's). The study predates the stopping rule of Section~\ref{sec:training}: it stopped on plain patience and kept the last weights."

**P26 (C248).** OLD: "A check of the adopted design in the C++ harness on the same fold (two runs, small search)"
NEW: "A check of a first version of the design in the C++ harness on the same fold (two runs, small search, $T=4$ fixed, before the stopping rule above)". Better: replace with the format-9 numbers when job 5316016 reports.

**P27 (C018, C252, C249).** OLD: "The submitted version also searched LSTM, GRU and Transformer autoencoders. They differ [...]. Before leaving them out we measured them once, on the development folds"
NEW: "LSTM, GRU and Transformer autoencoders differ [...]. We left them out of the main run and report what they cost under one reduced budget on the development folds". Abstract: OLD "were measured once, under one reduced search budget" NEW "are reported under one reduced search budget".

**P28 (C267, C268, C276, C277).** OLD: "one that normalizes before each sub-layer does not~\cite{xiong2020layer}. We therefore retrained the Transformer-AE winners of both development folds,"
NEW: "one that normalizes before each sub-layer may not, as Xiong et al.\ show for deep translation models~\cite{xiong2020layer}. In a post-hoc check we retrained two Transformer-AE winners of each development fold from an earlier run,"
After the verdict add: ", where leaving the plateau means reaching 10\% below the mean predictor on the monitor; with these counts the arms cannot be told apart, batch size and the shared learning rate were not varied, and the test does not show that a Transformer cannot learn this task."

**P29 (C311).** OLD: "the information available is the same, its code is not."
NEW: "a stochastic or latency code also loses part of the window's information, which is included in the measured gap."

**P30 (C031).** Either price the encoder alone as well (the on-device part), or OLD: "compress a signal window to a small code on a low-power device and reconstruct it elsewhere." NEW: "compress a signal window to a small code and reconstruct it on a low-power device."

**P31 (C029, C035, C075).** Citations: move "adds a weight" to \cite{lemaire2022analytical}; replace \cite{horowitz2014computing,dampfhoffer2023snns} on potentials by \cite{lemaire2022analytical}; add a group-disjoint reference after verifying one.

**P32 (C308).** OLD: "The repricings bound how much the conclusion depends on the memory model, not on the chip." NEW: "The repricings probe how much the conclusion depends on the memory model; they do not bound it."

**P33 (code, before the grid).**
- Run the encoder-only clamp and sigmoid under Xavier, adopted design, both dev folds, 3+ seeds (a small variant of the twin; minutes per run). This decides P01, P22, P23, P24.
- Read sampling rate from EDF headers, or exclude S088, S092, S100, and state which.
- Add a PCA E (P21) and fold-1 sensitivity (P19).
- Re-run T1 on the format-9 winners, or keep the wording of P28.
- Archive the twin's logs.

---

## 6. Not checked, and why

- **Encoder-only Xavier clamp probe:** blocked by a PreToolUse hook on the Bash call; I did not add a `# CONFIRMED` label because nobody confirmed it. The file was not created. The windows exist at `software/nn/results/meeting01_t1/*/` in the real repo.
- **Twin numbers in Table tab:design:** no logs survive; not re-run. All "PARTIAL" rows rest on wiki transcriptions, which are internally consistent.
- **C++ gtests:** no build in the snapshot. Named tests are cited, not run.
- **Job 5315335 / 5314513 numbers** cited from QT and the wiki; files not in the snapshot.
- **Siena, AudioMNIST data:** not on this machine, so Fp1, 512 Hz per file, 60 speakers are from PhysioNet and papers, not headers.
- **Paywalled full text:** Dampfhoffer, Yan (abstract only used); Cawley (abstract only). Goodfellow Alg. 7.1 and Maass not opened.
- **Zenodo record contents:** DOI resolves; the API returned HTML, files not inspected.
- **Holm's DOI** 404 is one `doi.org` request; JSTOR stable URL not fetched.
- **PCA energy figures** are my own application of the table in `energy.py`, not a harness output.
