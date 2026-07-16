# TODO — active

## Experiment05 pipeline (thesis-critical, priority order)

1. Run `e05_phase00_rank.py` against the now-complete 300-profile result set to produce `winners.json`.
2. Run `e05_apply_winner.py` to inject the real winner into the 32 phase01 profiles' placeholder `feature_extraction` block.
3. Execute the 32 `classifier.type=dsnn` phase01 profiles (`run_e05_profiles.sh phase01`) --- this is the thesis's actual authentication experiment and currently has zero results.
4. Once real DSNN results exist, consider an explicit ablation pass for `weight_decay`, `firing_rate_reg_lambda`, and tdBN --- none of the three has ever been exercised outside the debug/smoke profile:
   - [ ] `training.weight_decay` > 0 --- only set in `debug.json`/`smoke/debug.json`; not in any real phase01 profile
   - [ ] `training.firing_rate_reg_lambda` > 0 --- same: debug/smoke only
   - [ ] `training.batch_normalization = "threshold-dependent"` (tdBN) --- code path implemented (`E05Config.hpp:135`, `ThresholdDependentBatchNorm`), but never set in any shipped profile at all, debug included --- still genuinely untested end-to-end

Full verified status: see "Experiment05 status log" below.

## Open questions

- Im seeing that you are going to test only ae profile with 4 layers, but im pretty sure that SOTA for low end devices like raspberry pi b can use more layers. m i wrong ?

## Aprofundar (revisão de texto/tese)

24. Página 31: Verificar consistência entre as equações do LIF e o código utilizado.
25. Figura 19: Explicar que os pulsos apresentados são resultado de uma simulação de neurônios de pulso/RNP.
30. Seção BPTT: ESTUDAR!!!!!
40. Revisar todo o trabalho para garantir que LFCC seja a representação espectral principal.
42. Criar tabela comparativa entre métodos manuais, automatizados, escalas e wavelets.
47. Criar seção específica para Threshold-Dependent Batch Normalization (TDBN). ok (ESTUDAR!!!)
48. Revisar toda a monografia/Wiki para garantir que toda variável seja explicitamente definida.
49. Avaliar arquiteturas compactas de autoencoders para Raspberry Pi.
51. Avaliar opcionalmente diferentes algoritmos de otimização.
53. Revisar sistematicamente e enriquecer a Wiki do projeto.
56. Avaliar comparativamente diferentes arquiteturas de autoencoders utilizando Engenharia Paraconsistente de Características.
57. Verificar e fundamentar a afirmação sobre taxas de aprendizado para parâmetros biofísicos em SNNs.
58. Fundamentar a tabela de associação entre codificações e funções de perda.

---

# Adiadas / rejeitadas

7. Página 17: Incluir passo a passo do cálculo das transformadas wavelet packet. (nem fodendo)
8. Página 18: Listar outras técnicas além da engenharia paraconsistente para consistência de características. (por enquanto não)

---

# Resolvidas

## C12 --- fusão voice+EEG não implementada (wiki x código)

Wiki (`Experiment05.md`:134,390) diz que `modality=fused` concatena os vetores de características de voz+EEG. Código (`E05FeatureExtraction.cpp::signal_for_modality`, ramo `else // "fused"`) na verdade escolhia áudio se presente, senão EEG — um único sinal, sem concatenação. Não era fusão precoce (fundir sinal bruto antes do autoencoder/handcrafted) nem fusão tardia (concatenar vetores de características depois) — nenhuma fusão ocorria.

Action:
(a) implementar fusão tardia real
(b) implementar fusão precoce
(c) a distinção fusão-precoce-vs-tardia deve ser discutida na tese/wiki como eixo experimental

**RESOLVIDO (2026-07-03):**
(a) Fusão tardia implementada. `extract_features` com `modality=fused, fusion_mode=late` (padrão) extrai voz e EEG independentemente (cada um na taxa nativa, 44100/1024 Hz) e concatena os vetores por amostra `[voz ‖ eeg]`. `E05FeatureExtraction.cpp`.
(b) Fusão precoce implementada. `fusion_mode=early` concatena os sinais brutos (voz seguida de EEG) e extrai numa única passagem, usando a taxa da voz. Novo campo `E05Config::Dataset::fusion_mode` (validado early/late).
(c) Eixo discutido na tese (cap. 08, itemize fusão precoce/tardia) e na wiki (`Experiment05.md`, tabela fusion_mode). README + schema do perfil atualizados; perfis `*-fused.json` agora declaram `fusion_mode: late` explicitamente.
    Testes: 3 novos (E05Fusion.*) — dimensão tardia = voz+eeg, precoce difere e rotula distinto, modo inválido lança. Suítes e05 verdes (feat 26, profile 56, classifiers 16). Tese compila (101 pág).
    Bug original corrigido de passagem: `signal_for_modality` (fallback áudio-senão-EEG, sem fusão) foi substituído; `modality_sample_rate` removido em favor de constantes por sinal `kVoiceSampleRate`/`kEegSampleRate`.

## Phase00 batch failures (4 daub32 profiles)

`lfcc_c2`×{eeg,voice}, `mel_c1`×{eeg,voice} failed under the original parallel 300-profile batch run. Re-run individually on 2026-07-16 and passed cleanly — transient resource contention from parallel jobs, not a code defect. `results/run_profiles_phase00.state` now shows 300/300 PASS.

## tempStrategy.tex accidentally included in the compiled thesis

`documentation/00-thesis/monography/tempStrategy.tex` (the source of the "Experiment05 status log" / "Experiment05 pipeline reference" sections below) was `\include`d as the very first thing in the thesis's pretextual section (`monografia.tex:12`), despite its own header comment claiming "scratch, not \input anywhere". Removed the `\include` line on 2026-07-16 (page count 118 → 111); content preserved here instead.

---

# Reference --- Experiment05 pipeline config & grid

_Non-actionable background/reference material, migrated from `tempStrategy.tex` on 2026-07-16 when that file was dropped from the compiled thesis. See "TODO — active" above and "Experiment05 status log" below for what's actually outstanding._

Modality (`dataset.modality`):
- `voice` --- audio only
- `eeg` --- EEG only
- `fused` --- voice+EEG combined (not a third recorded signal); sub-axis `fusion_mode` = `early` (fuse raw signals before extraction) or `late` (concatenate feature vectors after per-signal extraction, default)

Feature-extraction strategy (`feature_extraction.strategy`):
- `handcrafted` --- DTWPT + descriptors
- `autoencoder` --- `snn-ae` (spiking) and `ann-ae` (dense) implemented and wired into Phase 00; `lstm-ae` remains in code (Guayaquil paper) but no thesis profile uses it

Handcrafted scale (`handcrafted.scale`, only if strategy=handcrafted):
- `bark`
- `mel`
- `lfcc`

Handcrafted mother wavelet (`handcrafted.wavelet`, only if strategy=handcrafted) --- swept in Phase 00; 23 options with coefficient traits in `include/wavelet/Types.hpp`:
- `haar`
- `daub4`, `daub6`, `daub8`, ..., `daub46` (even N)

Handcrafted descriptors (`handcrafted.descriptors`, list, any subset): energy, zcr, entropy, teager, jitter, shimmer

Classifier:
- RNN (Residual Neural Network, non-spiking) --- Guayaquil paper only
- DSNN (Deep residual Spiking NN) --- thesis only

Eval mode:
- text-dependent
- text-independent (primary contribution mode)

CV scheme (`training.nested_cv`):
- `true` → nested 5-fold (outer test / inner model selection)
- `false` → flat grouped 5-fold

Optional regularization/normalization toggles (independent, all off by default):
- weight_decay (L2, AdamW-style)
- firing_rate_reg_lambda (DSNN only)
- batch_normalization = "threshold-dependent" (DSNN only)

Existing shipped profiles (modality × strategy, 6 total): handcrafted-eeg, handcrafted-voice, handcrafted-fused, autoencoder-eeg, autoencoder-voice, autoencoder-fused (+ debug, article-full)

Full combinatorial space (handcrafted): 3 modality × 3 scale = 9 base combos × classifier(2) × eval mode(2) × CV(2) = 72.
Autoencoder branch: 3 modality × classifier(2) × eval(2) × CV(2) = 24 (SNN-AE and ANN-AE implemented and wired; lstm-ae present in code but unused by thesis profiles).

## Thesis experimental grid --- what to actually test (2 phases)

_Grounded in chapters/08-proposedApproach.tex (secs. estruturaDaEstrategiaProposta, Métricas) and chapters/06-Introduction.tex (Questão 4 --- text-dependent/independent)._

### Phase 00 --- Feature-vector construction (best vector per signal, via paraconsistent EPC)

Goal: for **voice** and for **EEG** separately, find the feature-extraction method that maximizes paraconsistent separability (D_truth minimal), before any classifier sees the data (ch08 §estruturaDaEstrategiaProposta).

Candidates per signal (voice, EEG each get their own ranking --- fused vectors are built *after* this phase, from each side's winning vector):
- **Handcrafted**, swept over *mother wavelet* × *scale*:
  - wavelet: 23 options (`haar` + `daub4`...`daub46`)
  - scale: bark, mel, lfcc; and `cepstral` (bool): false = Categoria 1 (band energies), true = Categoria 2 (log+DCT-II → LFCC/MFCC/BFCC)

  each carrying the descriptor set: energy, ZCR, entropy, Teager-Kaiser, jitter, shimmer. So 23 × 3 = 69 handcrafted (wavelet × scale) combos per signal, ×2 for `cepstral` (Categoria 1 / Categoria 2) = 138 handcrafted variants per signal.
- **Autoencoder** --- 12 compact AEs per signal: 9 SNN-AE (spiking; 3 sizes × 3 temporal encodings --- poisson/latency/direct) and 3 ANN-AE (dense; 3 sizes), latent 8/16/32, 2:1 hidden.

So the Phase 00 grid = 2 signals × (138 handcrafted + 12 autoencoder) = **300 rankings**, each scored by α, β, G1, G2, D_truth (sec:conceitos). Output of this phase: one winning feature vector for voice, one for EEG.

The 12 autoencoders per signal are compact single-layer SNN-AEs (poisson/latency/direct × 3 sizes) and ANN-AEs (3 sizes), both families wired; the Categoria-2 cepstral handcrafted variants are also implemented and shipped. SNN-AE spike frames use temporal integration (state reset once per sample, then integrated over `time_steps`=16 frames, mean-latent readout) and a lowered encoder firing threshold (`voltage_threshold`=0.2 for poisson/latency; the LIF default 1.0 for direct, which reproduces the un-encoded baseline). Measured on tiny/EEG: poisson α=0.069, latency α=0.258, direct α=0.875 (near coin-flip) --- confirming temporal coding is what makes the SNN-AE separable at all.

### Phase 01 --- Authentication (DSNN classification, biometric ablation)

Goal: feed **only Phase 00's single winning combination** (the wavelet×scale or autoencoder chosen by the paraconsistent ranking) into the **DSNN** (deep residual spiking classifier, thesis-only --- the plain RNN/ResNet classifier is Guayaquil-paper-only, ch08 §82) and measure biometric verification performance under 4 signal-source modes, to isolate each source's contribution (ch08 §32):

- [00a] **voz + EEG, fusão precoce** (`fused-early`) --- raw voice+EEG concatenated before one extraction pass
- [00b] **voz + EEG, fusão tardia** (`fused-late`) --- per-signal extraction, feature vectors concatenated after (audit C12)
- [01] **apenas voz (fonada)** --- ablation, voice vector only
- [02] **apenas EEG (imaginada)** --- ablation, EEG vector only

Each of these 4 source modes is additionally crossed with 2 axes explicitly required by the thesis's own research questions / metrics section:
- **text-dependent vs. text-independent** (ch06, Questão 4): same phrase spoken+imagined vs. arbitrary/mismatched phrase between train and test. Tests whether extraction-method performance is text-sensitive.
- **CV scheme**: nested 5-fold (unbiased hyperparameter selection) vs. flat grouped 5-fold. Both are speaker-disjoint (verification protocol, ch08 §71) --- test speakers never appear in training.

The shipped grid also crosses a third axis not in the original plan, `training.standardize_features` (raw vs. per-fold z-score, audit G1), so Phase 01 grid = 4 source modes × 2 (dependent/independent) × 2 (nested/flat) × 2 (raw/std) = **32 DSNN profiles**, all shipped under `profiles/phase01/` (`classifier.type=dsnn`), using the single Phase-00 winning combination (the extractor axis is *not* swept in Phase 01 --- the paraconsistent ranking already picked it). The shipped profiles still carry a placeholder `feature_extraction` block (`daub4/lfcc`) to be replaced with the real Phase-00 winner via `e05_apply_winner.py` before running.

**Primary metrics** (ch08 §71, verification protocol): EER and AUC. Closed-set metrics (accuracy/F1/precision/recall/specificity) only reported if a closed-set evaluation is also run alongside verification --- not the main claim. MSE is reconstruction-only (autoencoder training), not a classification metric.

### Observations / things easy to forget

- **Paired loading invariant**: the loader always pulls both audio+EEG per trial (via `eeg_index`); the `modality` field only selects what feeds extraction. This guarantees voice-only / EEG-only / fused runs share the *exact same* subject/trial set --- otherwise the 3-mode ablation in Phase 01 would not be a fair comparison.
- **Regularization is a training-config axis, not a method-combination axis** --- but it must stay fixed (or itself be swept and reported) across Phase 01 runs to keep comparisons fair: decoupled L2 weight decay (AdamW-style, spares biophysical R, C, V_th) and firing-rate regularization (DSNN-only, keeps spike rate in [0.05, 0.80], prevents dead/saturated neurons).
- **Threshold-Dependent Batch Normalization (tdBN)** --- DSNN-only, stabilizes deep spiking net training; also a fixed/swept training config, not a feature-extraction combination.
- **Data leakage / input normalization** (audit G1, implemented): per-feature z-score standardization of the classifier input, statistics fit on each fold's training rows only and applied to train+test (`E05Classifiers.cpp`, `training.standardize_features`, default on). Prevents test-set leakage. Matches ch07 §normalizacaoEntrada.
- **Pre-emphasis** (α=0.97) applies to audio-derived vectors only, before energy/spectral computation --- relevant to every voice-involving combo in both phases (fused included).
- **LSTM-AE and plain RNN/ResNet classifier are out of thesis scope** --- they exist in the Experiment05 code because both papers (Guayaquil congress) share the same pipeline, but are not part of either phase's combination grid above.

---

# Experiment05 status log (re-verified against code + run-state, 2026-07-16)

_Status pulled from: `src/experiments/05/lib/{include,src}/E05Config.{hpp,cpp}`, `E05FeatureExtraction.cpp`, `E05Classifiers.cpp`, `profiles/*.json`, `results/run_profiles_phase00.state`, `results/phase00/`, `results/phase01/`._
`[x]` = implemented, has a shipped profile, AND has been executed (result files on disk). `[~]` = implemented + shipped profile exists, but never executed yet. `[ ]` = not implemented / rejected by validate().

## Phase 00 --- feature extraction (300 profiles: 2 signals × (69 handcrafted × 2 categories + 12 autoencoder))

**Category 1 vs 2** (audit G2, implemented): `handcrafted.scale` groups DTWPT sub-bands by frequency (linear/Bark/Mel). With `cepstral=false` the per-band energies are used directly (Categoria 1); with `cepstral=true` a log+DCT-II stage over those energies yields the cepstral coefficients LFCC/MFCC/BFCC (Categoria 2). Both categories are selectable and shipped.

Voice and EEG (identical status, signal-agnostic code path):
- [x] Linear/Mel/Bark-band energy (Categoria 1) --- `cepstral=false`, 23 wavelets × 3 scales × 2 signals = 138 profiles, all executed
- [x] LFCC/MFCC/BFCC (Categoria 2) --- `cepstral=true`, 138 profiles, all executed
- [x] SNN-AE --- `ProtocolSpikingAutoencoder`; 3 sizes × 3 temporal encodings (poisson/latency/direct) × 2 signals = 18 profiles, all executed
- [x] ANN-AE --- `ProtocolAutoencoder`; 3 sizes × 2 signals = 6 profiles, all executed

Paraconsistent ranking itself (α, β, G1, G2, D_truth): **[x] implemented**, `E05Paraconsistent.cpp`, exercised by `e05_profile_audit_gtest`.

**Phase 00 is fully run**: `results/run_profiles_phase00.state` shows **300/300 PASS** (see "Resolvidas" above re: the 4 daub32 profiles that needed a re-run). Every profile has its 3-repeat paraconsistent-ranking CSV/JSON under `results/phase00/`.

**Remaining gap**: `scripts/pipeline/e05_phase00_rank.py` (reads the 300 results, picks the per-signal winner, writes `winners.json`) has not been run yet against the now-complete result set --- `winners.json` does not exist. Phase 01's profiles still carry a placeholder `daub4/lfcc` extractor rather than the real Phase-00 winner. → TODO item 1.

## Phase 01 --- DSNN authentication (32 profiles: 4 source modes × 2 text_mode × 2 CV × 2 standardize_features)

Every `profiles/phase01/*.json` sets `classifier.type = "dsnn"` (32 profiles; the old `"rnn"` profiles now live only in `debug.json`/`smoke/debug.json` and are Guayaquil-paper-only, out of thesis scope). `E05DsnnClassifier` is implemented and unit-tested (`e05_classifiers_gtest`, 16/16 passing), and every axis is shipped as a real profile:

- [~] fused-early, fused-late, voice-only, eeg-only × classifier=dsnn --- 8 profiles/mode-pair combo, shipped, **not yet executed**
- [~] `text_mode`: `dependent` and `independent`, each paired with real (non-debug) data --- shipped, **not yet executed**
- [~] `nested_cv`: `true` (nested 5-fold) and `false` (flat grouped 5-fold) --- shipped, **not yet executed**
- [~] `training.standardize_features`: `true` and `false` --- shipped, **not yet executed**

**Biggest remaining gap**: `results/phase01/` does not exist --- **none of the 32 DSNN profiles have been executed**. The thesis-primary classifier has full code + profile coverage but zero results. Blocked on the `winners.json` gap above (replace the placeholder extractor via `e05_apply_winner.py` before running). → TODO items 2--3.

Regularization/normalization toggles: see TODO item 4 above (weight_decay / firing_rate_reg_lambda / tdBN, none exercised outside debug/smoke).
