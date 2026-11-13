#!/usr/bin/env python3
"""Tests for the thesis pipeline scripts: 01_thesis_phase00_rank.py,
02_thesis_apply_winner.py, thesis_build_phase00_paraconsistent_tables.py,
thesis_build_phase01_auth_tables.py and thesis_cross_profile_significance.py.

Stdlib unittest (no pytest needed). Drives the scripts end-to-end on the profiles
00_thesis_generate_profiles.py generates (into a temporary directory, never the tree) and
synthetic results in the format thesis writes: winner selection, ranking stability, the
injection into Phase 01 profiles, both table builders and the significance tests.

Run:  python3 scripts/testing/test_thesis_phase_scripts.py
"""
import csv
import importlib.util
import json
import os
import shutil
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
BIOMEX_TABLES = os.path.join(PIPELINE_DIR, "thesis_build_biomex_tables.py")

GENERATE = os.path.join(PIPELINE_DIR, "00_thesis_generate_profiles.py")
SIGNALS = ("voice", "eegpron", "eegimag")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load(GENERATE, "thesis_generate_profiles")


def generated(phase):
    """{stem: profile} exactly as 00_thesis_generate_profiles.py writes them, so the fixtures
    carry the real names and the real schema (a hand-written fixture drifted from both and
    left 25 of these tests failing against scripts that were right)."""
    rows = {"phase00": GEN.phase00, "phase01": GEN.phase01, "biomex": GEN.biomex}[phase]()
    return {os.path.basename(rel)[:-len(".json")]: prof for rel, prof in rows}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def rep_tags(prof):
    """thesis.cpp appends _rep<k> to the run_tag only when repeats > 1; files add e05_."""
    tag, reps = prof["experiment"]["run_tag"], prof["experiment"]["repeats"]
    return [f"e05_{tag}"] if reps == 1 else [f"e05_{tag}_rep{k}" for k in range(reps)]


# Speaker subsets as thesis writes them: leave-one-speaker-out and Phase 01 training folds.
SUBSETS = ("loso:1", "loso:2", "loso:3", "fold:0:0", "fold:0:1")


def write_para(res, prof, d, subset_d=None, label="x"):
    """The two CSVs a Phase 00 run writes per repetition (write_paraconsistent_csv and the
    subsets file). g2 = 0, so d_penalized == d_truth."""
    for t in rep_tags(prof):
        os.makedirs(res, exist_ok=True)
        with open(os.path.join(res, f"{t}_paraconsistent.csv"), "w") as f:
            f.write("label,alpha,beta,g1,g2,d_truth,d_penalized\n")
            f.write(f"{label},0.8,0.2,0.6,0.0,{d},{d}\n")
        with open(os.path.join(res, f"{t}_paraconsistent_subsets.csv"), "w") as f:
            f.write("subset,label,alpha,beta,d_truth,d_penalized\n")
            for i, sub in enumerate(SUBSETS):
                v = (subset_d or {}).get(sub, d + 0.001 * i)
                f.write(f"{sub},{label},0.8,0.2,{v},{v}\n")


# Per signal: the handcrafted winner, a handcrafted loser, and the best of the other two
# families (the ranking needs one per family). D_penalized: lower is better.
PHASE00_FIXTURE = {"p00_hc_haar_lfcc_c1": 0.20, "p00_hc_daub10_lfcc_c1": 0.35,
                   "p00_ae_ann_tiny": 0.50, "p00_pca_k8": 0.45}
# Different winners per signal, so the apply test can tell which winner went where.
WINNER = {"voice": "p00_hc_haar_lfcc_c1", "eegpron": "p00_hc_daub10_lfcc_c1",
          "eegimag": "p00_hc_haar_lfcc_c1"}


def phase00_fixture(p00, res):
    all00 = generated("phase00")
    for sig in SIGNALS:
        for stem, d in PHASE00_FIXTURE.items():
            if stem.startswith("p00_hc_"):
                d = 0.20 if stem == WINNER[sig] else 0.35
            name = f"{stem}_{sig}"
            write_json(os.path.join(p00, f"{name}.json"), all00[name])
            write_para(res, all00[name], d)
    return all00


class PhaseScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p00 = os.path.join(self.tmp, "phase00")
        self.p01 = os.path.join(self.tmp, "phase01")
        self.res = os.path.join(self.tmp, "results")
        self.all00 = phase00_fixture(self.p00, self.res)
        all01 = generated("phase01")
        for name in ("p01_dsnn_voice_indep_flat_std", "p01_dsnn_eegpron_indep_flat_std",
                     "p01_dsnn_eegimag_indep_flat_std", "p01_dsnn_earlypron_indep_flat_std",
                     "p01_cmp-pca_voice_dep"):
            write_json(os.path.join(self.p01, f"{name}.json"), all01[name])
        self.winners = os.path.join(self.tmp, "winners.json")
        self.stability = os.path.join(self.tmp, "winner_stability.csv")

    def rank(self):
        return subprocess.run(
            [sys.executable, RANK, "--profiles-dir", self.p00, "--results-dir", self.res,
             "--out", self.winners, "--stability-out", self.stability],
            check=False, capture_output=True, text=True)

    def run_rank(self):
        r = self.rank()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.winners) as f:
            return json.load(f)

    def para(self, stem, rep=0):
        return os.path.join(self.res, f"e05_e05_{stem}_rep{rep}_paraconsistent.csv")

    def test_rank_refuses_a_missing_repetition(self):
        # 2 of 3 repetitions: averaging what is there would rank a different estimate.
        os.remove(self.para("p00_hc_haar_lfcc_c1_voice", 2))
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("rep2_paraconsistent.csv", r.stderr)
        self.assertFalse(os.path.exists(self.winners))

    def test_rank_refuses_a_missing_subsets_file(self):
        os.remove(self.para("p00_pca_k8_eegpron", 1).replace(".csv", "_subsets.csv"))
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("rep1_paraconsistent_subsets.csv", r.stderr)

    def test_rank_ignores_a_tag_that_only_shares_the_prefix(self):
        # e05_<tag>* used to glob ..._voice_extra_* into ..._voice's mean.
        stray = dict(self.all00["p00_hc_haar_lfcc_c1_voice"])
        stray["experiment"] = dict(stray["experiment"],
                                   run_tag="e05_p00_hc_haar_lfcc_c1_voice_extra")
        write_para(self.res, stray, 9.0)
        winners = self.run_rank()
        self.assertAlmostEqual(winners["voice"]["d_penalized"], 0.20, places=6)

    def test_rank_refuses_a_csv_without_d_penalized(self):
        with open(self.para("p00_hc_haar_lfcc_c1_voice"), "w") as f:
            f.write("label,alpha,beta,g1,g2,d_truth\nhandcrafted,0.8,0.2,0.6,0.0,0.2\n")
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("d_penalized", r.stderr)

    def test_rank_refuses_more_than_one_scored_row(self):
        with open(self.para("p00_hc_haar_lfcc_c1_voice"), "a") as f:
            f.write("other,0.8,0.2,0.6,0.0,0.1,0.1\n")
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("2 data rows", r.stderr)

    def test_rank_requires_every_argument(self):
        for drop in ("--out", "--stability-out"):
            args = {"--profiles-dir": self.p00, "--results-dir": self.res,
                    "--out": self.winners, "--stability-out": self.stability}
            del args[drop]
            r = subprocess.run([sys.executable, RANK] + [x for kv in args.items() for x in kv],
                               check=False, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, drop)
            self.assertIn(drop, r.stderr)

    def test_rank_refuses_a_signal_without_one_family(self):
        os.remove(os.path.join(self.p00, "p00_pca_k8_eegimag.json"))
        r = self.rank()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no pca profile", r.stderr)

    def test_rank_picks_min_d_penalized_per_signal_and_family(self):
        winners = self.run_rank()
        for sig in SIGNALS:
            want = self.all00[f"{WINNER[sig]}_{sig}"]["feature_extraction"]
            self.assertEqual(winners[sig]["feature_extraction"], want, sig)
            self.assertAlmostEqual(winners[sig]["d_penalized"], 0.20, places=6)
            self.assertEqual(winners[sig]["by_family"]["pca"]["label"], "pca/k8")
            self.assertEqual(winners[sig]["by_family"]["ae"]["label"], "autoencoder/ann-ae/latent8")

    def test_stability_records_every_subset_of_every_signal(self):
        self.run_rank()
        with open(self.stability, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(sorted((r["signal"], r["subset"]) for r in rows),
                         sorted((s, sub) for s in SIGNALS for sub in SUBSETS))
        self.assertTrue(all(r["full_winner_rank"] == "1" for r in rows))

    def test_apply_requires_the_fused_choice(self):
        self.run_rank()
        r = subprocess.run([sys.executable, APPLY, "--winners", self.winners,
                            "--profiles-dir", self.p01],
                           check=False, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--fused", r.stderr)

    def fe(self, name):
        with open(os.path.join(self.p01, f"{name}.json")) as f:
            return json.load(f)["feature_extraction"]

    def test_apply_injects_winner_per_source(self):
        winners = self.run_rank()
        subprocess.run([sys.executable, APPLY, "--winners", self.winners,
                        "--profiles-dir", self.p01, "--fused", "voice"],
                       check=True, capture_output=True, text=True)
        for sig in SIGNALS:
            self.assertEqual(self.fe(f"p01_dsnn_{sig}_indep_flat_std"),
                             winners[sig]["feature_extraction"], sig)
        # fused <- the declared --fused voice; a comparison profile <- its family's best
        self.assertEqual(self.fe("p01_dsnn_earlypron_indep_flat_std"),
                         winners["voice"]["feature_extraction"])
        self.assertEqual(self.fe("p01_cmp-pca_voice_dep"),
                         winners["voice"]["by_family"]["pca"]["feature_extraction"])

    def test_apply_is_idempotent(self):
        self.run_rank()
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, APPLY, "--winners", self.winners,
                 "--profiles-dir", self.p01, "--fused", "voice"],
                check=True, capture_output=True, text=True)
        self.assertIn("updated 0/", out.stdout)

    def test_apply_fused_eeg_option(self):
        winners = self.run_rank()
        subprocess.run([sys.executable, APPLY, "--winners", self.winners,
                        "--profiles-dir", self.p01, "--fused", "eeg"],
                       check=True, capture_output=True, text=True)
        # early fusion of pronounced EEG <- the pronounced-EEG winner, not the voice one
        self.assertEqual(self.fe("p01_dsnn_earlypron_indep_flat_std"),
                         winners["eegpron"]["feature_extraction"])
        self.assertNotEqual(winners["eegpron"]["feature_extraction"],
                            winners["voice"]["feature_extraction"])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Phase00Tables(unittest.TestCase):
    """The tables are defined by the profile set, not by whatever sits in results/."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p00 = os.path.join(self.tmp, "phase00")
        self.res = os.path.join(self.tmp, "results")
        self.tables = os.path.join(self.tmp, "tables")
        self.all00 = phase00_fixture(self.p00, self.res)
        for name in os.listdir(self.p00):
            prof = self.all00[name[:-len(".json")]]
            d = 0.1 + 0.01 * len(name)
            for t in rep_tags(prof):
                self.summary(t, d)

    def summary(self, tag, d):
        write_json(os.path.join(self.res, f"{tag}_summary.json"),
                   {"best_d_truth": d, "best_d_penalized": d, "best_alpha": 0.8,
                    "best_beta": 0.2})

    def run_tables(self):
        return subprocess.run(
            [sys.executable, TABLES, "--profiles-dir", self.p00, "--results-dir", self.res,
             "--tables-dir", self.tables], check=False, capture_output=True, text=True)

    def test_writes_every_group_when_complete(self):
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        for fam in ("ae", "hc", "pca"):
            for sig in SIGNALS:
                self.assertTrue(os.path.isfile(os.path.join(self.tables,
                                                            f"phase00_{fam}_{sig}.csv")))
        with open(os.path.join(self.tables, "phase00_pca_voice.csv"), newline="") as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row["configuration"], "PCA (k = 8)")
        self.assertEqual(row["std_d_penalized"], "—")  # deterministic: no spread, not 0
        self.assertEqual(row["fold_wins"].split("/")[1], "2")   # the two fold: subsets

    def test_refuses_a_missing_summary(self):
        os.remove(os.path.join(self.res, "e05_e05_p00_pca_k8_eegpron_rep2_summary.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("e05_e05_p00_pca_k8_eegpron_rep2_summary.json", r.stderr)

    def test_refuses_an_empty_group(self):
        os.remove(os.path.join(self.p00, "p00_pca_k8_eegimag.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("pca_eegimag", r.stderr)

    def test_a_result_without_a_profile_is_not_read(self):
        self.summary("e05_e05_p00_hc_haar_bark_c1_eegpron_rep0", 0.0)
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.tables, "phase00_hc_eegpron.csv")) as f:
            self.assertNotIn("Bark", f.read())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


ENERGY = {"nj": 100.0, "nj_as_executed": 120.0, "nj_sram8k": 90.0, "nj_sram32k": 110.0,
          "nj_sram1m": 300.0, "ops_nj": 10.0, "spikes": 50.0, "mem_nj": 90.0,
          "accesses": 1000.0, "fp_adds": 100.0, "fp_macs": 10.0}


class Phase01Tables(unittest.TestCase):
    """The Phase 01 tables are defined by the generated profile set; labels come from the
    declared config, and every cell the tables print must have its results."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p01 = os.path.join(self.tmp, "phase01")
        self.res = os.path.join(self.tmp, "results")
        self.tables = os.path.join(self.tmp, "tables")
        self.all01 = generated("phase01")
        for i, (name, prof) in enumerate(sorted(self.all01.items())):
            write_json(os.path.join(self.p01, f"{name}.json"), prof)
            eer = 0.30 + 0.001 * i
            if name == "p01_dsnn_latepron_dep_flat_std":
                eer = 0.05            # the best cell, by construction
            for k, t in enumerate(rep_tags(prof)):
                self.summary(t, prof, eer + 0.01 * k)

    def summary(self, tag, prof, eer, auc=0.9):
        r = {"mean_eer": eer, "mean_auc": auc}
        if prof["classifier"]["type"] != "cosine":
            r["energy"] = ENERGY
        write_json(os.path.join(self.res, f"{tag}_summary.json"), {"results": [r]})

    def run_tables(self):
        return subprocess.run(
            [sys.executable, AUTH_TABLES, "--profiles-dir", self.p01, "--results-dir", self.res,
             "--tables-dir", self.tables], check=False, capture_output=True, text=True)

    def read(self, table):
        with open(os.path.join(self.tables, f"phase01_{table}.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_ranks_by_eer_with_labels_from_the_config(self):
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = self.read("auth")
        self.assertEqual(rows[0]["configuration"],
                         "Fusão tardia (pronunciada) · frases vistas · $k$-fold plano · padronizado")
        self.assertEqual(rows[0]["mean_eer"], "0.0600")     # mean of 0.05, 0.06, 0.07
        self.assertEqual(rows[0]["std_eer"], "0.0082")      # population sd of the three
        for t in ("controls", "dysphonia", "extractors", "energy"):
            self.assertTrue(self.read(t), t)

    def test_refuses_a_missing_summary(self):
        os.remove(os.path.join(self.res, "e05_e05_p01_ann_voice_dep_flat_std_rep1_summary.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("e05_e05_p01_ann_voice_dep_flat_std_rep1_summary.json", r.stderr)

    def test_refuses_a_non_finite_eer(self):
        prof = self.all01["p01_dsnn_voice_dep_flat_std"]
        self.summary("e05_e05_p01_dsnn_voice_dep_flat_std_rep1", prof, float("nan"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("mean_eer is nan", r.stderr)

    def test_refuses_a_profile_outside_the_generated_sets(self):
        write_json(os.path.join(self.p01, "a_voice.json"), self.all01["p01_dsnn_voice_dep_flat_std"])
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("belongs to no Phase 01 set", r.stderr)

    def test_refuses_a_name_its_config_contradicts(self):
        prof = json.loads(json.dumps(self.all01["p01_dsnn_voice_dep_flat_std"]))
        prof["classifier"]["text_mode"] = "independent"
        write_json(os.path.join(self.p01, "p01_dsnn_voice_dep_flat_std.json"), prof)
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("its name says text mode", r.stderr)

    def test_refuses_a_missing_control_cell(self):
        os.remove(os.path.join(self.p01, "p01_cosine_eegimag_indep_flat_std.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("controls (cosine)", r.stderr)

    def test_a_single_repetition_has_no_spread_rather_than_zero(self):
        name = "p01_dsnn_voice_dep_flat_std"
        prof = json.loads(json.dumps(self.all01[name]))
        prof["experiment"]["repeats"] = 1
        write_json(os.path.join(self.p01, f"{name}.json"), prof)
        self.summary(f"e05_{prof['experiment']['run_tag']}", prof, 0.10)
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        voice = next(x for x in self.read("auth") if x["configuration"].startswith(
            "Voz · frases vistas · $k$-fold plano · padronizado"))
        self.assertEqual((voice["mean_eer"], voice["std_eer"], voice["std_auc"]),
                         ("0.1000", "—", "—"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class BiomexTables(unittest.TestCase):
    """Every BIOMEX-DB cell is printed next to the same Coretto cell, and only then."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bx, self.co = (os.path.join(self.tmp, d) for d in ("biomex", "phase01"))
        self.bxres, self.cores = (os.path.join(self.tmp, d) for d in ("bxres", "cores"))
        self.tables = os.path.join(self.tmp, "tables")
        self.allbx, self.allco = generated("biomex"), generated("phase01")
        for d, res, profs, base in ((self.bx, self.bxres, self.allbx, 0.20),
                                    (self.co, self.cores, self.allco, 0.30)):
            for i, (name, prof) in enumerate(sorted(profs.items())):
                write_json(os.path.join(d, f"{name}.json"), prof)
                for t in rep_tags(prof):
                    self.summary(res, t, prof, base + 0.001 * i)

    def summary(self, res, tag, prof, eer):
        r = {"mean_eer": eer, "mean_auc": 0.9}
        if prof["classifier"]["type"] != "cosine":
            r["energy"] = ENERGY
        write_json(os.path.join(res, f"{tag}_summary.json"), {"results": [r]})

    def run_tables(self):
        return subprocess.run(
            [sys.executable, BIOMEX_TABLES, "--profiles-dir", self.bx, "--results-dir", self.bxres,
             "--coretto-profiles-dir", self.co, "--coretto-results-dir", self.cores,
             "--tables-dir", self.tables], check=False, capture_output=True, text=True)

    def read(self, table):
        with open(os.path.join(self.tables, f"biomex_{table}.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_each_row_carries_the_same_coretto_cell(self):
        r = self.run_tables()
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = self.read("auth")
        self.assertEqual(len(rows), 8)      # 4 sources x 2 text modes
        names = sorted(self.allco)
        late = next(x for x in rows if x["source"] == "Fusão tardia (pronunciada)"
                    and x["text"] == "frases vistas")
        want = 0.30 + 0.001 * names.index("p01_dsnn_latepron_dep_flat_std")
        self.assertEqual(late["coretto_dsnn_eer"], f"{want:.4f}")
        bx_names = sorted(self.allbx)
        self.assertEqual(late["dsnn_eer"],
                         f"{0.20 + 0.001 * bx_names.index('bx01_dsnn_latepron_dep_flat_std'):.4f}")
        self.assertEqual(late["dsnn_over_ann"], "1.00")
        dys = self.read("dysphonia")
        self.assertEqual(len(dys), 6)       # voice, early, late fusion x two scopes
        voice = next(x for x in dys if x["source"] == "Voz" and x["scope"] == "só sondas")
        self.assertAlmostEqual(float(voice["rise"]),
                               float(voice["eer_l100"]) - float(voice["eer_l0"]), places=4)

    def test_refuses_a_missing_biomex_summary(self):
        os.remove(os.path.join(self.bxres, "e05_e05_bx01_cosine_voice_dep_flat_std_rep2_summary.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("bx01_cosine_voice_dep_flat_std_rep2_summary.json", r.stderr)

    def test_refuses_a_profile_of_the_other_base(self):
        write_json(os.path.join(self.bx, "bx01_dsnn_voice_dep_flat_std.json"),
                   self.allco["p01_dsnn_voice_dep_flat_std"])
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not biomex2021", r.stderr)

    def test_refuses_when_the_coretto_cell_is_missing(self):
        os.remove(os.path.join(self.co, "p01_dsnn_earlypron_indep_flat_std.json"))
        os.remove(os.path.join(self.cores, "e05_e05_p01_dsnn_earlypron_indep_flat_std_rep0_summary.json"))
        r = self.run_tables()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Coretto", r.stderr + r.stdout)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def p01_profile(tag, modality, fusion_mode=None):
    """A Phase 01 profile for the significance tests: the run_tag, the repetitions, and every
    block the paired test compares to decide whether two profiles hold out the same folds."""
    ds = {"root": "/x", "results_dir": "results/thesis/phase01", "modality": modality}
    if fusion_mode:
        ds["fusion_mode"] = fusion_mode
    ds["max_samples"] = 0
    return {
        "experiment": {"run_tag": tag, "seed": 42, "repeats": 3, "seed_deterministic": False},
        "dataset": ds,
        "feature_extraction": {"strategy": "handcrafted", "handcrafted": {
            "transform": "dtwpt", "scale": "lfcc", "wavelet": "haar", "cepstral": False,
            "descriptors": ["energy", "zcr", "entropy", "teager", "jitter", "shimmer"],
            "dtwpt_level": 4}},
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

    def run_sig(self, *extra, pairing="independent", reference="best", metric="eer",
                select="^e05_"):
        args = [sys.executable, SIGNIFICANCE, "--profiles-dir", self.p01,
                "--results-dir", self.res, *extra]
        for flag, value in (("--metric", metric), ("--reference", reference),
                            ("--pairing", pairing), ("--select", select)):
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
        for missing in ("pairing", "reference", "metric", "select"):
            r = self.run_sig(**{missing: None})
            self.assertNotEqual(r.returncode, 0, missing)
            self.assertIn(f"--{missing}", r.stderr)

    def test_select_keeps_only_the_declared_family(self):
        self.add("c", "eeg", "independent", [0.20, 0.22, 0.25, 0.21, 0.24, 0.26])
        out = os.path.join(self.tmp, "sig.csv")
        r = self.run_sig("--out", out, reference="e05_a", select="^e05_[ab]$")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, newline="") as f:
            self.assertEqual([x["condition"] for x in csv.DictReader(f)], ["e05_b"])
        r = self.run_sig(reference="e05_a", select="^e05_a$")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("a comparison needs 2", r.stderr)

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
        shutil.rmtree(self.tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
