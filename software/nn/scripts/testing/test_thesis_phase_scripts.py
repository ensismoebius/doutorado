#!/usr/bin/env python3
"""Tests for the thesis pipeline scripts: 01_thesis_phase00_rank.py,
02_thesis_apply_winner.py, thesis_build_phase00_paraconsistent_tables.py,
thesis_build_phase01_auth_tables.py and thesis_cross_profile_significance.py.

Stdlib unittest (no pytest needed). Drives both scripts end-to-end on synthetic
Phase 00 profiles + paraconsistent CSVs, then checks winner selection and the
injection into Phase 01 profiles.

Run:  python3 scripts/testing/test_thesis_phase_scripts.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(os.path.dirname(HERE), "pipeline", "thesis")
RANK = os.path.join(PIPELINE_DIR, "01_thesis_phase00_rank.py")
APPLY = os.path.join(PIPELINE_DIR, "02_thesis_apply_winner.py")
TABLES = os.path.join(PIPELINE_DIR, "thesis_build_phase00_paraconsistent_tables.py")
AUTH_TABLES = os.path.join(PIPELINE_DIR, "thesis_build_phase01_auth_tables.py")
SIGNIFICANCE = os.path.join(PIPELINE_DIR, "thesis_cross_profile_significance.py")

DESCRIPTORS = ["energy", "zcr", "entropy", "teager", "jitter", "shimmer"]


def handcrafted(wavelet, scale):
    return {"strategy": "handcrafted",
            "handcrafted": {"transform": "dtwpt", "scale": scale, "wavelet": wavelet,
                            "cepstral": False, "descriptors": DESCRIPTORS, "dtwpt_level": 4}}


def hc_profile(tag, signal, wavelet, scale):
    """A Phase 00 profile in the strict schema: every key the run reads, nothing else."""
    return {
        "experiment": {"run_tag": tag, "seed": 42, "repeats": 3, "seed_deterministic": False},
        "dataset": {"root": "/x", "results_dir": "results/thesis/phase00", "modality": signal,
                    "max_samples": 0},
        "feature_extraction": handcrafted(wavelet, scale),
        "paraconsistent": {"enabled": True},
        "classifier": {"enabled": False},
    }


def p01_profile(tag, modality, fusion_mode=None):
    ds = {"root": "/x", "results_dir": "results/thesis/phase01", "modality": modality}
    if fusion_mode:
        ds["fusion_mode"] = fusion_mode
    ds["max_samples"] = 0
    return {
        "experiment": {"run_tag": tag, "seed": 42, "repeats": 3, "seed_deterministic": False},
        "dataset": ds,
        # placeholder extractor that must be overwritten by apply
        "feature_extraction": handcrafted("daub4", "lfcc"),
        "paraconsistent": {"enabled": False},
        "classifier": {"type": "dsnn",
            "layer_spec": ["linear:128:relu", "residual:2", "linear:N_speakers:identity"],
            "text_mode": "independent", "enabled": True},
        "training": {"epochs": 1, "learning_rate": 1e-3, "samples_per_batch": 8,
                     "k_folds": 2, "nested_cv": True, "early_stop_patience": 1,
                     "standardize_features": True, "optimizer_type": "adam",
                     "gradient_clip_norm": 0.0, "weight_decay": 0.0,
                     "firing_rate_reg_lambda": 0.0, "batch_normalization": "none"},
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_para_csv(path, d_truth, label="handcrafted"):
    """The CSV thesis writes (write_paraconsistent_csv). g2 = 0, so d_penalized == d_truth."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("label,alpha,beta,g1,g2,d_truth,d_penalized\n")
        f.write(f"{label},0.8,0.2,0.6,0.0,{d_truth},{d_truth}\n")


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

    def rank(self):
        return subprocess.run(
            [sys.executable, RANK, "--profiles-dir", self.p00,
             "--results-dir", self.res, "--out", self.winners],
            check=False, capture_output=True, text=True)

    def run_rank(self):
        r = self.rank()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.winners) as f:
            return json.load(f)

    def test_rank_refuses_a_missing_repetition(self):
        # 2 of 3 repetitions: averaging what is there would rank a different estimate.
        os.remove(os.path.join(self.res, "e05_e05_p00_hc_haar_bark_voice_rep2_paraconsistent.csv"))
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("rep2_paraconsistent.csv", r.stderr)
        self.assertFalse(os.path.exists(self.winners))

    def test_rank_ignores_a_tag_that_only_shares_the_prefix(self):
        # e05_<tag>* used to glob ..._voice_extra_* into ..._voice's mean.
        write_para_csv(os.path.join(
            self.res, "e05_e05_p00_hc_haar_bark_voice_extra_rep0_paraconsistent.csv"), 9.0)
        winners = self.run_rank()
        self.assertAlmostEqual(winners["voice"]["d_penalized"], 0.20, places=6)

    def test_rank_refuses_a_csv_without_d_penalized(self):
        path = os.path.join(self.res, "e05_e05_p00_hc_haar_bark_voice_rep0_paraconsistent.csv")
        with open(path, "w") as f:
            f.write("label,alpha,beta,g1,g2,d_truth\nhandcrafted,0.8,0.2,0.6,0.0,0.2\n")
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("d_penalized", r.stderr)

    def test_rank_refuses_more_than_one_scored_row(self):
        path = os.path.join(self.res, "e05_e05_p00_hc_haar_bark_voice_rep0_paraconsistent.csv")
        with open(path, "a") as f:
            f.write("other,0.8,0.2,0.6,0.0,0.1,0.1\n")
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("2 data rows", r.stderr)

    def test_rank_requires_every_argument(self):
        r = subprocess.run([sys.executable, RANK, "--profiles-dir", self.p00,
                            "--results-dir", self.res],
                           check=False, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--out", r.stderr)

    def test_apply_requires_the_fused_choice(self):
        self.run_rank()
        r = subprocess.run([sys.executable, APPLY, "--winners", self.winners,
                            "--profiles-dir", self.p01],
                           check=False, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--fused", r.stderr)

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
        # fused → the declared --fused voice
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


class Phase00Tables(unittest.TestCase):
    """The tables are defined by the profile set, not by whatever sits in results/."""

    # One profile per typeset group: {ae, hc, pca} x {voice, eeg}. repeats = 1 here, so
    # the summary carries the bare run_tag (no _rep<k>).
    STEMS = ("p00_ae_ann_tiny", "p00_hc_haar_lfcc_c1", "p00_pca_k8")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p00 = os.path.join(self.tmp, "phase00")
        self.res = os.path.join(self.tmp, "results")
        self.tables = os.path.join(self.tmp, "tables")
        d = 0.1
        for stem in self.STEMS:
            for sig in ("voice", "eeg"):
                tag = f"e05_{stem}_{sig}"
                prof = hc_profile(tag, sig, "haar", "lfcc")
                prof["experiment"] = {"run_tag": tag, "seed": 42, "repeats": 1}
                write_json(os.path.join(self.p00, f"{stem}_{sig}.json"), prof)
                write_json(os.path.join(self.res, f"e05_{tag}_summary.json"),
                           {"modality": sig, "best_d_truth": d, "best_d_penalized": d,
                            "best_alpha": 0.8, "best_beta": 0.2})
                d += 0.1

    def run_tables(self):
        return subprocess.run(
            [sys.executable, TABLES, "--profiles-dir", self.p00, "--results-dir", self.res,
             "--tables-dir", self.tables], check=False, capture_output=True, text=True)

    def test_writes_every_group_when_complete(self):
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        for key in ("ae_eeg", "ae_voice", "hc_eeg", "hc_voice", "pca_eeg", "pca_voice"):
            self.assertTrue(os.path.isfile(os.path.join(self.tables, f"phase00_{key}.csv")), key)
        with open(os.path.join(self.tables, "phase00_pca_voice.csv"), newline="") as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row["configuration"], "PCA (k = 8)")
        self.assertEqual(row["std_d_truth"], "—")  # repeats = 1: no spread, not 0

    def test_refuses_a_missing_summary(self):
        os.remove(os.path.join(self.res, "e05_e05_p00_pca_k8_eeg_summary.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("e05_e05_p00_pca_k8_eeg_summary.json", r.stderr)

    def test_refuses_an_empty_group(self):
        os.remove(os.path.join(self.p00, "p00_pca_k8_eeg.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("pca_eeg", r.stderr)

    def test_a_result_without_a_profile_is_not_read(self):
        write_json(os.path.join(self.res, "e05_e05_p00_hc_haar_bark_c1_eeg_summary.json"),
                   {"modality": "eeg", "best_d_truth": 0.0, "best_d_penalized": 0.0,
                    "best_alpha": 1.0, "best_beta": 0.0})
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.tables, "phase00_hc_eeg.csv")) as f:
            self.assertNotIn("BARK", f.read())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


class Phase01Tables(unittest.TestCase):
    """The Phase 01 table is defined by the profiles; labels come from the declared config."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p01 = os.path.join(self.tmp, "phase01")
        self.res = os.path.join(self.tmp, "results")
        self.tables = os.path.join(self.tmp, "tables")
        # (file stem, modality, fusion_mode, per-repetition EER)
        for stem, mod, fm, eers in (("a_voice", "voice", None, (0.10, 0.20)),
                                    ("b_fused", "fused", "late", (0.05, 0.07))):
            prof = p01_profile(f"e05_{stem}", mod, fm)
            prof["experiment"]["repeats"] = 2
            write_json(os.path.join(self.p01, f"{stem}.json"), prof)
            for k, eer in enumerate(eers):
                self.write_summary(f"e05_e05_{stem}_rep{k}_summary.json", eer)

    def write_summary(self, name, eer, auc=0.9):
        write_json(os.path.join(self.res, name),
                   {"results": [{"mean_eer": eer, "mean_auc": auc}]})

    def run_tables(self):
        return subprocess.run(
            [sys.executable, AUTH_TABLES, "--profiles-dir", self.p01, "--results-dir", self.res,
             "--tables-dir", self.tables], check=False, capture_output=True, text=True)

    def read_table(self):
        with open(os.path.join(self.tables, "phase01_auth.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_ranks_by_eer_with_labels_from_the_config(self):
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = self.read_table()
        self.assertEqual([row["configuration"] for row in rows], [
            "Fusão tardia · independente de texto · $k$-fold aninhado · padronizado",
            "Voz · independente de texto · $k$-fold aninhado · padronizado"])
        self.assertEqual(rows[1]["mean_eer"], "0.1500")

    def test_refuses_a_missing_summary(self):
        os.remove(os.path.join(self.res, "e05_e05_a_voice_rep1_summary.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("e05_e05_a_voice_rep1_summary.json", r.stderr)

    def test_refuses_a_non_finite_eer(self):
        self.write_summary("e05_e05_a_voice_rep1_summary.json", float("nan"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("mean_eer is nan", r.stderr)

    def test_refuses_two_profiles_with_one_label(self):
        prof = p01_profile("e05_c_voice", "voice")
        prof["experiment"]["repeats"] = 2
        write_json(os.path.join(self.p01, "c_voice.json"), prof)
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("share the label", r.stderr)

    def test_a_single_repetition_has_no_spread_rather_than_zero(self):
        path = os.path.join(self.p01, "a_voice.json")
        with open(path) as f:
            prof = json.load(f)
        prof["experiment"] = {"run_tag": "e05_a_voice", "seed": 42, "repeats": 1}
        write_json(path, prof)
        self.write_summary("e05_e05_a_voice_summary.json", 0.10)
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        voice = next(x for x in self.read_table() if x["configuration"].startswith("Voz"))
        self.assertEqual((voice["std_eer"], voice["std_auc"]), ("—", "—"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


METRICS_HEADER = ("feature_set,classifier,text_mode,fold,accuracy,f1,precision,recall,"
                  "specificity,eer,auc,train_ms,infer_ms,model_path\n")


class CrossProfileSignificance(unittest.TestCase):
    """Every choice is declared; paired tests only between profiles that share the folds."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p01 = os.path.join(self.tmp, "phase01")
        self.res = os.path.join(self.tmp, "results")
        # Two repetitions x 3 folds each. a is best on EER.
        self.add("a", "voice", "independent", [0.10, 0.12, 0.11, 0.13, 0.10, 0.12])
        self.add("b", "eeg", "independent", [0.30, 0.28, 0.33, 0.31, 0.29, 0.35])

    def add(self, name, modality, text_mode, eers, accuracy="0.5"):
        prof = p01_profile(f"e05_{name}", modality)
        prof["experiment"]["repeats"] = 2
        prof["classifier"]["text_mode"] = text_mode
        write_json(os.path.join(self.p01, f"{name}.json"), prof)
        for k in range(2):
            path = os.path.join(self.res, f"e05_e05_{name}_rep{k}_metrics.csv")
            os.makedirs(self.res, exist_ok=True)
            with open(path, "w") as f:
                f.write(METRICS_HEADER)
                f.writelines(f"hc,dsnn,{text_mode},{fold},{accuracy},0.5,0.5,0.5,0.5,"
                             f"{eers[3 * k + fold]},0.9,1.0,1.0,m\n" for fold in range(3))

    def run_sig(self, *extra, pairing="independent", reference="best", metric="eer"):
        args = [sys.executable, SIGNIFICANCE, "--profiles-dir", self.p01,
                "--results-dir", self.res, *extra]
        for flag, value in (("--metric", metric), ("--reference", reference),
                            ("--pairing", pairing)):
            if value is not None:
                args += [flag, value]
        return subprocess.run(args, check=False, capture_output=True, text=True)

    def test_independent_against_the_best(self):
        out = os.path.join(self.tmp, "sig.csv")
        r = self.run_sig("--out", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(x["reference"], x["condition"], x["test"]) for x in rows],
                         [("e05_a", "e05_b", "mannwhitneyu(indep)")])
        self.assertLess(float(rows[0]["p_value"]), 0.05)

    def test_holm_adjusts_over_the_tested_family(self):
        # Two conditions worse than the reference a: a Holm family of m = 2.
        self.add("c", "eeg", "independent", [0.20, 0.22, 0.25, 0.21, 0.24, 0.26])
        out = os.path.join(self.tmp, "sig.csv")
        r = self.run_sig("--out", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, newline="") as f:
            rows = {x["condition"]: x for x in csv.DictReader(f)}
        raw = sorted(float(x["p_value"]) for x in rows.values())
        expected = [min(1.0, 2 * raw[0])]
        expected.append(max(expected[0], min(1.0, raw[1])))
        got = sorted(float(x["p_holm"]) for x in rows.values())
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=12)
        self.assertEqual({x["holm_family_size"] for x in rows.values()}, {"2"})
        for x in rows.values():
            self.assertEqual(x["significant_0.05"], str(int(float(x["p_holm"]) < 0.05)))
        self.assertIn("biased towards a difference", r.stdout)

    def test_holm_reference_values(self):
        sys.path.insert(0, PIPELINE_DIR)
        try:
            import thesis_cross_profile_significance as sig
        finally:
            sys.path.remove(PIPELINE_DIR)
        got = sig.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        # sorted 0.01, 0.03, 0.04 -> 3*0.01, 2*0.03, max(0.06, 1*0.04)
        self.assertAlmostEqual(got["a"], 0.03)
        self.assertAlmostEqual(got["c"], 0.06)
        self.assertAlmostEqual(got["b"], 0.06)

    def test_paired_between_profiles_that_share_the_folds(self):
        r = self.run_sig(pairing="paired")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("wilcoxon(paired)", r.stdout)

    def test_paired_refused_when_the_folds_differ(self):
        self.add("c", "voice", "dependent", [0.2] * 6)
        r = self.run_sig(pairing="paired")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("classifier.text_mode", r.stderr)

    def test_every_choice_is_required(self):
        for missing in ("pairing", "reference", "metric"):
            r = self.run_sig(**{missing: None})
            self.assertNotEqual(r.returncode, 0, missing)
            self.assertIn(f"--{missing}", r.stderr)

    def test_unknown_reference_refused(self):
        r = self.run_sig(reference="e05_nope")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("neither 'best' nor a profile run_tag", r.stderr)

    def test_missing_metrics_file_refused(self):
        os.remove(os.path.join(self.res, "e05_e05_b_rep1_metrics.csv"))
        r = self.run_sig()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("e05_e05_b_rep1_metrics.csv is missing", r.stderr)

    def test_nan_metric_refused_not_dropped(self):
        self.add("c", "voice", "independent", [0.2] * 6, accuracy="nan")
        r = self.run_sig(metric="accuracy")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("compare on eer or auc", r.stderr)

    def test_tied_values_have_no_p_value_instead_of_a_stand_in(self):
        self.add("c", "voice", "independent", [0.4] * 6)
        self.add("d", "voice", "independent", [0.4] * 6)
        out = os.path.join(self.tmp, "sig.csv")
        r = self.run_sig("--out", out, reference="e05_c")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, newline="") as f:
            row = next(x for x in csv.DictReader(f) if x["condition"] == "e05_d")
        self.assertEqual(row["p_value"], "")
        self.assertEqual(row["p_holm"], "")
        self.assertEqual(row["cohens_d"], "")
        self.assertIn("all values tied", row["note"])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
