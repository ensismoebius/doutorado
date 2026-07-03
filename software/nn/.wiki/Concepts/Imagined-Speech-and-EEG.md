# Imagined Speech and EEG

> **Plain language version:** [Imagined Speech and EEG — Plain Language Guide](./Plain/Imagined-Speech-and-EEG.md)

Imagined speech (also called covert speech or inner speech) is the phenomenon where a person mentally articulates words or phrases without producing audible sound. Brain regions activated during imagined speech overlap substantially with those active during overt phonated speech, making EEG a viable signal source for capturing speech-related neural activity — including in speakers who cannot produce intelligible overt speech.

This page documents the neuroscientific background underlying the thesis "Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined Speech" (A. Furlan, UNESP).

## Theoretical Background

### Brain Regions Involved in Speech

Both phonated and imagined speech activate a distributed perisylvian cortical network [1, 2]:

| Region | Role |
|---|---|
| **Broca's area** (frontal lobe, dominant hemisphere) | Speech production, fluency, motor planning |
| **Wernicke's area** (temporal + parietal, dominant hemisphere) | Speech comprehension, phonological processing |
| **Primary and premotor cortex** | Execution and planning of articulatory movements |
| **Supramarginal gyrus** | Sensorimotor integration for speech |
| **Superior and middle temporal gyrus** | Linguistic comprehension, syllable repetition |

In phonated speech, the primary motor cortex is additionally activated to drive the articulatory muscles. In imagined speech, this motor activation is reduced but not absent — cortical motor planning still occurs without the final motor execution step [3, 4].

**Hemispheric dominance:** In most right-handed individuals the dominant hemisphere (left) hosts Broca's and Wernicke's areas. In left-handers there is a higher (though still minority) probability of right-hemisphere dominance [1].

### Imagined vs Phonated Speech

The key difference for EEG-based biometrics:

| Aspect | Phonated | Imagined |
|---|---|---|
| Motor cortex activation | Strong | Reduced / absent |
| Auditory cortex | Self-monitoring active | Variable |
| Perisylvian network | Full | Full |
| Electromagnetic artifacts | Vocal muscle EMG present | Absent |
| Inter-subject variability | High (voice anatomy) | High (neural patterns) |

Because imagined speech activates the full language network without generating articulatory muscle artifacts (EMG contamination), it is advantageous for users with severe dysphonia [3].

### EEG Frequency Bands

EEG oscillations are classified by frequency band [5]:

| Band | Range | Associated state |
|---|---|---|
| **Delta** | 1–4 Hz | Deep sleep (adults), infants |
| **Theta** | 4–8 Hz | Drowsiness, memory recall; amplitude < 100 µV |
| **Alpha** | 8–12 Hz | Relaxed wakefulness, eyes closed; amplitude < 50 µV |
| **Beta** | 12–25 Hz | Active thinking, focused attention; amplitude < 30 µV |
| **Gamma** | >25 Hz | Multi-sensory processing; lowest amplitude |

For imagined speech classification, **Alpha, Beta, and Theta** bands are most commonly informative; using each band in isolation tends to improve classifier performance relative to full-spectrum analysis [6].

### Sampling Rate Requirements

Biological EEG in healthy humans: typically 0.5–40 Hz dominant activity. High-frequency oscillations (HFOs) up to ~200 Hz occur in the neocortex [7, 8].

By Nyquist, capturing activity up to ~200 Hz requires at least 400 Hz. The public dataset used in this project (Pressel Coretto et al., 2017) samples EEG at **1024 Hz**, well above that floor (Nyquist = 512 Hz).

Any anti-aliasing/bandpass upper cutoff must sit **below** the Nyquist frequency (i.e. below 512 Hz at 1024 Hz sampling); a highpass near 1 Hz removes DC drift, and a notch removes powerline interference (50/60 Hz depending on the acquisition site) [5, 9].

### The 10-20 Electrode System

Standard scalp electrode placement uses the international 10-20 system [10]. Odd indices = left hemisphere, even = right:

| Region | Channels | Tasks |
|---|---|---|
| Frontal | Fp1, Fp2, Fpz, F3, F4, F7, F8 | Memory, concentration, emotions |
| Parietal | P3, P4, Pz | Problem solving, attention, touch |
| Temporal | T3, T5, T4, T6 | Memory, face recognition, audition, words |
| Occipital | O1, O2, Oz | Vision, reading |
| Sensorimotor | C3, C4, Cz | Motor control, sensorimotor integration |

**Speech-specific electrode attention:**
- **F7, T5** (left frontal + temporal): closest 10-20 positions to Wernicke's area
- **Fp1, F3, F7**: closest to Broca's area
- Imagined speech activates bilateral temporal and parietal regions [3, 4], so both hemispheres should be monitored

### Aphasia Context

Lesions to the perisylvian speech regions produce aphasia [1]:

| Type | Lesion site | Effect |
|---|---|---|
| Broca's aphasia | Broca's area | Non-fluent production, intact comprehension |
| Wernicke's aphasia | Wernicke's area | Fluent but meaningless output |
| Transcortical motor | Superior/anterior to Broca | Non-fluent, intact repetition |
| Transcortical sensory | Surrounding Wernicke | Fluent, impaired comprehension |

The thesis targets speakers with **severe dysphonia** (degraded voice, not aphasia). Their neural speech representations are intact — EEG captures these intact patterns even when the acoustic output is unintelligible.

---

## How It Connects to the nn Library

### Data Collection Protocol (thesis)

Two collection cycles per speaker:
- **Silent cycle**: minimal ambient noise
- **Noisy cycle**: pre-recorded crowd noise played during collection

Sentences collected (Portuguese): first name, directional commands (cima/baixo/esquerda/direita), invented password, common phrases (e.g., "Estou com fome", "Sinto dor", "Entrar no sistema").

Three modalities recorded simultaneously:
1. Phonated speech (microphone)
2. Imagined speech (EEG only, no microphone)
3. Mixed (simultaneous phonation + EEG)

### 10.1117/12.2255697 Public Dataset

Used for preliminary validation experiments (15 Spanish-speaking subjects, vowels + directional commands):

```
include/data_loaders/10.1117/      # C++ loader
src/core/data_loaders/10.1117/     # Implementation
```

Relevant channels for imagined speech: F7, T5 (Wernicke), F3, F7, Fp1 (Broca).

### Signal Preprocessing Pipeline

```
EEG raw → highpass ~1 Hz + anti-alias lowpass (< Nyquist 512 Hz) → notch (50/60 Hz) → feature extraction (DTWPT)
                                            → energy bands (BARK / MEL / LFCC)
                                            → paraconsistent evaluation
                                            → SNN classifier
```

---

## See Also

- [Core/Paraconsistent](../Core/Paraconsistent.md) — Feature quality evaluation
- [Core/Wavelet](../Core/Wavelet.md) — DTWPT decomposition
- [Core/DataLoaders](../Core/DataLoaders.md) — 10.1117 EEG dataset loader
- [Concepts/LFCC](./LFCC.md) — Linear cepstral features for speaker verification
- [Research-Context](../Research-Context.md) — Full thesis overview

---

## References

[1] M. de Pinto, *Manual de Neurologia*, Editora Manole, 2012.

[2] T. W. Vanderah et al., *Netter's Neuroscience*, 3rd ed., Elsevier, 2020.

[3] A. G. Flinker et al., "Redefining the role of Broca's area in speech," *Proc. Natl. Acad. Sci. USA*, vol. 112, no. 9, pp. 2871–2875, 2015. DOI: 10.1073/pnas.1414491112.

[4] I. DeWitt and J. P. Rauschecker, "Phoneme and word recognition in the auditory ventral stream," *Proc. Natl. Acad. Sci. USA*, vol. 109, no. 8, pp. E505–E514, 2013.

[5] A. Jalaly Bidgoly et al., "A survey on methods and challenges in EEG based authentication," *Computers & Security*, vol. 93, p. 101788, 2020.

[6] P. Agarwal and A. Rakshit, "EEG based identification of imagined alphabets," *Neurocomputing*, 2022.

[7] E. W. Moffett et al., "High-frequency oscillations in the human brain," *J. Neurosci.*, 2017.

[8] A. K. Engel et al., "High-frequency oscillations in the neocortex," 2009.

[9] A. Jalaly Bidgoly et al., 2020 (same as [5]).

[10] "Standard for EEG electrode placement (10-20 system)," *IEEE*, 1991.
