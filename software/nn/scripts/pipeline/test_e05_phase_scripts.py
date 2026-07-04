#!/usr/bin/env python3
"""Tests for e05_phase00_rank.py and e05_apply_winner.py.

Stdlib unittest (no pytest needed). Drives both scripts end-to-end on synthetic
Phase 00 profiles + paraconsistent CSVs, then checks winner selection and the
injection into Phase 01 profiles.

Run:  python3 scripts/pipeline/test_e05_phase_scripts.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RANK = os.path.join(HERE, "e05_phase00_rank.py")
APPLY = os.path.join(HERE, "e05_apply_winner.py")


def hc_profile(tag, signal, wavelet, scale):
    return {
        "experiment": {"run_tag": tag, "seed": 42, "repeats": 3},
        "dataset": {"root": "/x", "results_dir": "results/phase00", "modality": signal},
        "feature_extraction": {"strategy": "handcrafted",
            "handcrafted": {"transform": "dtwpt", "scale": scale, "wavelet": wavelet,
                            "descriptors": ["energy"], "dtwpt_level": 4}},
        "paraconsistent": {"enabled": True},
        "classifier": {"enabled": False},
        "training": {"epochs": 1, "learning_rate": 1e-3, "samples_per_batch": 8,
                     "early_stop_patience": 1, "k_folds": 2, "nested_cv": True},
    }


def p01_profile(tag, modality, fusion_mode=None):
    ds = {"root": "/x", "results_dir": "results/phase01", "modality": modality}
    if fusion_mode:
        ds["fusion_mode"] = fusion_mode
    return {
        "experiment": {"run_tag": tag, "seed": 42, "repeats": 3},
        "dataset": ds,
        # placeholder extractor that must be overwritten by apply
        "feature_extraction": {"strategy": "handcrafted",
            "handcrafted": {"transform": "dtwpt", "scale": "lfcc", "wavelet": "daub4",
                            "descriptors": ["energy"], "dtwpt_level": 4}},
        "paraconsistent": {"enabled": False},
        "classifier": {"type": "dsnn",
            "layer_spec": ["linear:128:relu", "residual:2", "linear:N_speakers:identity"],
            "text_mode": "independent", "enabled": True},
        "training": {"epochs": 1, "learning_rate": 1e-3, "samples_per_batch": 8,
                     "early_stop_patience": 1, "k_folds": 2, "nested_cv": True},
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_para_csv(path, d_truth, label="handcrafted"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("label,alpha,beta,g1,g2,d_truth\n")
        f.write(f"{label},0.8,0.1,0.7,-0.1,{d_truth}\n")


class PhaseScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p00 = os.path.join(self.tmp, "phase00")
        self.p01 = os.path.join(self.tmp, "phase01")
        self.res = os.path.join(self.tmp, "results")
        os.makedirs(self.p00)
        os.makedirs(self.p01)
        os.makedirs(self.res)

        # Phase 00 profiles: two voice, two eeg combos.
        combos = [
            ("e05_p00_hc_haar_bark_voice", "voice", "haar", "bark", 0.20),
            ("e05_p00_hc_daub4_lfcc_voice", "voice", "daub4", "lfcc", 0.35),
            ("e05_p00_hc_daub20_mel_eeg", "eeg", "daub20", "mel", 0.15),
            ("e05_p00_hc_daub4_lfcc_eeg", "eeg", "daub4", "lfcc", 0.40),
        ]
        for tag, sig, wav, scale, d in combos:
            write_json(os.path.join(self.p00, f"{tag}.json"),
                       hc_profile(tag, sig, wav, scale))
            # repeats=3 → run_tag gets _repK; output file = "e05_" + run_tag (double e05_).
            for k in range(3):
                write_para_csv(
                    os.path.join(self.res, f"e05_{tag}_rep{k}_paraconsistent.csv"), d)

        # Phase 01 profiles: one per source.
        write_json(os.path.join(self.p01, "p01_dsnn_voice_indep_nested.json"),
                   p01_profile("e05_p01_dsnn_voice_indep_nested", "voice"))
        write_json(os.path.join(self.p01, "p01_dsnn_eeg_indep_nested.json"),
                   p01_profile("e05_p01_dsnn_eeg_indep_nested", "eeg"))
        write_json(os.path.join(self.p01, "p01_dsnn_fused-early_indep_nested.json"),
                   p01_profile("e05_p01_dsnn_fused_early", "fused", "early"))

        self.winners = os.path.join(self.tmp, "winners.json")

    def run_rank(self):
        subprocess.run(
            [sys.executable, RANK, "--profiles-dir", self.p00,
             "--results-dir", self.res, "--out", self.winners],
            check=True, capture_output=True, text=True)
        with open(self.winners) as f:
            return json.load(f)

    def test_rank_picks_min_d_truth(self):
        winners = self.run_rank()
        self.assertEqual(winners["voice"]["feature_extraction"]["handcrafted"]["wavelet"], "haar")
        self.assertEqual(winners["voice"]["feature_extraction"]["handcrafted"]["scale"], "bark")
        self.assertEqual(winners["eeg"]["feature_extraction"]["handcrafted"]["wavelet"], "daub20")
        self.assertAlmostEqual(winners["voice"]["d_truth"], 0.20, places=6)
        self.assertAlmostEqual(winners["eeg"]["d_truth"], 0.15, places=6)

    def test_apply_injects_winner_per_source(self):
        self.run_rank()
        subprocess.run(
            [sys.executable, APPLY, "--winners", self.winners,
             "--profiles-dir", self.p01, "--fused", "voice"],
            check=True, capture_output=True, text=True)

        def hc(name):
            with open(os.path.join(self.p01, name)) as f:
                return json.load(f)["feature_extraction"]["handcrafted"]

        self.assertEqual(hc("p01_dsnn_voice_indep_nested.json")["wavelet"], "haar")
        self.assertEqual(hc("p01_dsnn_eeg_indep_nested.json")["wavelet"], "daub20")
        # fused → voice winner (default)
        self.assertEqual(hc("p01_dsnn_fused-early_indep_nested.json")["wavelet"], "haar")

    def test_apply_is_idempotent(self):
        self.run_rank()
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, APPLY, "--winners", self.winners,
                 "--profiles-dir", self.p01, "--fused", "voice"],
                check=True, capture_output=True, text=True)
        # second run should report 0 changes
        self.assertIn("updated 0/", out.stdout)

    def test_apply_fused_eeg_option(self):
        self.run_rank()
        subprocess.run(
            [sys.executable, APPLY, "--winners", self.winners,
             "--profiles-dir", self.p01, "--fused", "eeg"],
            check=True, capture_output=True, text=True)
        with open(os.path.join(self.p01, "p01_dsnn_fused-early_indep_nested.json")) as f:
            hc = json.load(f)["feature_extraction"]["handcrafted"]
        self.assertEqual(hc["wavelet"], "daub20")  # eeg winner

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
