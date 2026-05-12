# Imagined Speech and EEG — Plain Language Guide

> **Technical reference:** [Imagined Speech and EEG](../Imagined-Speech-and-EEG.md)

---

## What is imagined speech?

Imagined speech (also called *inner speech* or *covert speech*) is what happens when you think in words without actually speaking out loud. Right now, as you read this sentence, you are probably "hearing" it in your head — that internal voice is imagined speech.

When you speak out loud, your brain first plans the speech (which muscles to move, which sounds to make) and then sends signals to your vocal muscles to execute it. In imagined speech, the planning step still happens but the final "send it to the muscles" step is suppressed.

---

## Why does this matter for the thesis?

The thesis targets a specific problem: what if a person cannot produce clear voice? This is the case for people with **severe dysphonia** — conditions like advanced laryngeal cancer, complete aphonia, or laryngectomy.

Conventional speaker verification compares voice recordings. If the voice is degraded or absent, the system fails.

But the *brain activity* for speech is still there. Even a person who cannot speak at all still has the cortical planning activity when they think words. An EEG (electroencephalogram) can record this brain activity and use it as an identity signal.

---

## What is an EEG?

An EEG is a set of electrodes placed on the scalp that measure tiny electrical signals produced by groups of neurons firing in the brain. Think of it as listening to a crowd — you cannot hear individual voices, but you can hear the overall "sound" of the crowd changing.

The signals are measured in microvolts (millionths of a volt) at sampling rates of 800 Hz or higher in this project.

---

## Where in the brain does speech happen?

Speech involves a network of regions:

| Brain region | What it does |
|---|---|
| **Broca's area** (front-left) | Plans the sequence of sounds and movements for speech |
| **Wernicke's area** (back-left) | Understands language; links sounds to meaning |
| **Motor cortex** | Actually moves the muscles during spoken speech |
| **Temporal lobe** | Processes auditory information |

For most right-handed people, these regions are in the left hemisphere. For imagined speech, Broca's area and Wernicke's area are still active — but the motor cortex (the part that moves muscles) is mostly quiet.

---

## Which electrodes pick up speech signals?

The scalp electrodes follow the **10-20 system** — an international standard for electrode placement. Electrodes are named by region (F=frontal, T=temporal, P=parietal, O=occipital, C=central) plus a number (odd=left side, even=right side).

Electrodes closest to speech areas:
- **F7, T5** (left frontal-temporal): near Wernicke's area
- **Fp1, F3, F7** (left frontal): near Broca's area

Both hemispheres should be monitored because imagined speech also activates the right hemisphere.

---

## EEG frequency bands

The brain doesn't produce a single frequency — it produces a mix of oscillations at different speeds:

| Band | Frequency | Associated with |
|---|---|---|
| Delta | 1–4 Hz | Deep sleep |
| Theta | 4–8 Hz | Drowsiness, memory |
| Alpha | 8–12 Hz | Relaxed wakefulness |
| Beta | 12–25 Hz | Active concentration |
| Gamma | >25 Hz | Complex thinking |

For imagined speech recognition, **Alpha, Beta, and Theta** bands carry the most useful information. Processing each band separately tends to work better than using the full raw signal.

---

## Why EEG instead of just a microphone?

| Property | Microphone | EEG |
|---|---|---|
| Works with silent speakers | ✗ | ✓ |
| Affected by muscle noise from speaking | No | No (imagined speech has no muscle activity) |
| Captures speaker identity | Via voice anatomy | Via neural patterns |
| Can be spoofed by voice synthesis | Increasingly yes | Harder — needs brain signal |

The EEG signal also provides an anti-coercion property: a person under duress cannot easily produce the correct brain pattern on demand even if physically coerced into speaking.

---

## Data collection for the thesis

For each speaker, the thesis protocol collects three types of signals simultaneously:

1. **Phonated speech** — microphone only
2. **Imagined speech** — EEG only, the person just thinks the phrase silently
3. **Mixed** — both microphone and EEG at the same time

Collection is done in two noise conditions: silent room and noisy (crowd noise played back), to test robustness.

Phrases are in Portuguese: first name, directional commands (*cima/baixo/esquerda/direita*), invented password, common sentences like *"Estou com fome"* (I'm hungry).

---

## See also

- [Imagined Speech and EEG (technical)](../Imagined-Speech-and-EEG.md) — full neuroscience detail, electrode tables, references
- [LFCC (plain)](./LFCC.md) — how speech features are extracted from the audio signal
- [Research Context](../../Research-Context.md) — how this fits the thesis
