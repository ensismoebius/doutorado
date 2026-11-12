# Thesis run of 2026-11-12, revised protocol — autoencoder and PCA rows void (superseded)

Phase 00 (275 profiles x 3, GridUnesp array 5323432, revision 13fafd0b) and Phase 01
(142 profiles x 3, array 5323525, revision 524da1fd), moved here unchanged.

**Do not quote the autoencoder or PCA numbers.** Those extractors were fed the waveform
averaged in 256 blocks, a low-pass at ~32 Hz for voice and ~5 Hz for EEG, with no speaker
information left in it (a linear classifier names the speaker from that input with accuracy
0.13 for voice and 0.26 for EEG, chance 0.067; from the log power spectrum of the same
signal, 0.85 and 0.62: `scripts/analysis/thesis_ae_input_reference.py`). Every `p00_ae_*`,
`p00_pca_*` and `p01_cmp-ae_*` / `p01_cmp-pca_*` result here is void: their Phase 01 EER of
0.48-0.52 is the input, not the extractor.

The handcrafted rows (`p00_hc_*`, and every Phase 01 profile on the handcrafted winner) do
not read that input and were not affected. They are superseded only because both phases
were run again under one source revision after the fix (`thesis::spectral_input`;
`.wiki/Experiments/Thesis.md`, "Protocol revision" section 9). Kept for provenance.
