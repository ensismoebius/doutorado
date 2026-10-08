import numpy as np

from efficient_nn_lab.live.features import log_energy_bins


def test_log_energy_bins_has_requested_length():
    sample_rate = 16000
    samples = np.zeros(int(0.4 * sample_rate))
    bins = log_energy_bins(samples, sample_rate, n_bins=20)
    assert bins.shape == (20,)


def test_log_energy_bins_silence_is_near_zero_everywhere():
    sample_rate = 16000
    samples = np.zeros(int(0.4 * sample_rate))
    bins = log_energy_bins(samples, sample_rate, n_bins=20)
    assert np.allclose(bins, 0.0, atol=1e-9)


def test_log_energy_bins_concentrates_energy_near_the_tone_frequency():
    # A pure 1000 Hz tone should light up the band whose filter is
    # centered nearest 1000 Hz far more than a band near the opposite end
    # of the range -- the whole reason this feature exists (vowels are
    # told apart by WHERE their energy concentrates, not how much there is).
    sample_rate = 16000
    duration = 0.4
    t = np.arange(int(duration * sample_rate)) / sample_rate
    tone = np.sin(2 * np.pi * 1000.0 * t)
    bins = log_energy_bins(tone, sample_rate, n_bins=20, fmin=80.0, fmax=4000.0)

    edges = np.linspace(80.0, 4000.0, 20 + 2)
    centers = edges[1:-1]
    near_tone_bin = int(np.argmin(np.abs(centers - 1000.0)))
    far_bin = int(np.argmax(np.abs(centers - 1000.0)))
    assert bins[near_tone_bin] > bins[far_bin]


def test_log_energy_bins_louder_signal_is_not_smaller():
    sample_rate = 16000
    t = np.arange(int(0.4 * sample_rate)) / sample_rate
    quiet = np.sin(2 * np.pi * 500.0 * t) * 0.1
    loud = np.sin(2 * np.pi * 500.0 * t) * 1.0
    quiet_bins = log_energy_bins(quiet, sample_rate, n_bins=20)
    loud_bins = log_energy_bins(loud, sample_rate, n_bins=20)
    assert loud_bins.sum() > quiet_bins.sum()
