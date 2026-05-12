# Autoencoders — Plain Language Guide

> **Technical reference:** [Autoencoders](../Autoencoders.md)

---

## What is an autoencoder?

An autoencoder is a neural network trained to do one job: **compress data and then reconstruct it**.

Imagine you need to describe a photograph in just 10 words, then have someone else redraw the photo from your 10-word description. The process of writing that description is *encoding*; the process of redrawing is *decoding*.

```
Original signal  →  [ENCODER]  →  compressed code  →  [DECODER]  →  reconstructed signal
```

The key insight: because the bottleneck (the "10-word description") is much smaller than the original, the network is forced to learn what is *important* in the data. The compressed code is called the **latent representation** or **latent vector**.

---

## Why is this useful?

Once the autoencoder is trained, you can:
- Use the compressed code as a compact *feature vector* for a classifier
- Detect anomalies (if reconstruction is bad, the input was unusual)
- Generate new examples by sampling from the latent space

In this project, autoencoders extract feature vectors from speech and EEG signals. The latent code summarises "what makes this signal unique" — and that summary is then fed to a speaker-recognition classifier.

---

## Types of autoencoders in this project

### 1. Plain autoencoder (AE)

Compresses and reconstructs. Loss = how different is the reconstruction from the original?

### 2. Denoising autoencoder (DAE)

Same idea, but the input is deliberately corrupted (noise added) before encoding. The network must learn to reconstruct the *clean* original. This forces it to learn the structure of the data, not just memorise inputs.

### 3. Variational autoencoder (VAE)

Instead of compressing to a single code, the encoder outputs a *distribution* (mean and spread). The decoder samples from this distribution. This forces the latent space to be smooth and continuous — useful for generating new examples.

The VAE adds an extra penalty to the loss: the latent distribution should stay close to a standard normal bell curve. This prevents the code from collapsing to arbitrary hard-to-interpret values.

### 4. Spiking variational autoencoder (SNN-VAE / β-SVAE)

Like a VAE, but the encoder uses spiking neurons and the latent code is represented as spike counts rather than continuous numbers. The distribution used is Poisson (for spike counts) instead of Gaussian. This fits the spiking neuron paradigm naturally because spike counts are discrete numbers that follow a Poisson distribution in biology.

---

## The training loop in simple terms

1. Feed a signal through the encoder → get a compressed code
2. Feed the code through the decoder → get a reconstruction
3. Measure how bad the reconstruction is (loss)
4. Adjust weights to make the reconstruction better
5. Repeat

For VAE: step 3 also penalises the latent code for drifting too far from the expected distribution.

---

## Loss functions and their pairing with encodings

This is a critical rule that is easy to get wrong:

| How is information encoded? | What loss to use? |
|---|---|
| Continuous values (ANN) | MSE — mean squared error between numbers |
| Spike counts (rate coding) | SpikeCountLoss — MSE between counts |
| Timing of first spike (latency coding) | SpikeTimeLoss — MSE between spike times |

Using the wrong loss gives broken gradients — the network may still train without errors but will learn nothing useful.

---

## Latent space size

The compressed code size controls how much information is preserved:

- Too small → network cannot reconstruct well; too much is thrown away
- Too large → network can cheat by memorising; learns nothing general
- Just right → forces the network to find the most informative structure

For speaker authentication, the latent vector should capture speaker identity, not noise or content details.

---

## See also

- [Autoencoders (technical)](../Autoencoders.md) — full math, Poisson VAE details, API
- [SNN and Surrogate Gradients (plain)](./SNN-and-Surrogate-Gradients.md) — what spiking neurons are
- [Spike Encoding (plain)](./Spike-Encoding.md) — rate vs latency coding
- [Spike Rate Regularization (plain)](./Spike-Rate-Regularization.md) — preventing broken SNN training
