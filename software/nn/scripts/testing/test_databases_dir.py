#!/usr/bin/env python3
"""The databases directory is written in four places; this test keeps them one.

    scripts/data/databases_dir.py                       DATABASES_REL
    scripts/pipeline/thesis/00_thesis_generate_profiles.py   DATABASES (the profiles' dataset.root)
    scripts/pipeline/meeting01/gridunesp_deploy.sh      DATABASES_REL
    scripts/pipeline/meeting01/ensure_datasets.sh       DATASETS_ROOT default

A profile whose dataset.root named another directory than the script that downloads or converts
the base would read nothing, or a stale copy. It also checks that every tracked thesis and
paraconsistentGA profile reads its base from that directory.

Stdlib unittest. Run:  python3 scripts/testing/test_databases_dir.py
"""
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
NN = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(NN, "scripts", "data"))
import databases_dir  # noqa: E402


def _read(*parts):
    with open(os.path.join(NN, *parts), encoding="utf-8") as f:
        return f.read()


class DatabasesDir(unittest.TestCase):
    def test_generator_names_the_same_directory(self):
        spec = importlib.util.spec_from_file_location(
            "gen", os.path.join(NN, "scripts", "pipeline", "thesis", "00_thesis_generate_profiles.py"))
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        self.assertEqual(gen.DATABASES, "~/" + databases_dir.DATABASES_REL)
        self.assertEqual(gen.BASES["coretto2017"], gen.DATABASES + "/database.sqlite")
        self.assertEqual(gen.BASES["biomex2021"], gen.DATABASES + "/biomex.sqlite")

    def test_deploy_script_names_the_same_directory(self):
        text = _read("scripts", "pipeline", "meeting01", "gridunesp_deploy.sh")
        m = re.search(r'^DATABASES_REL="([^"]+)"$', text, re.M)
        self.assertIsNotNone(m, "gridunesp_deploy.sh has no DATABASES_REL")
        self.assertEqual(m.group(1), databases_dir.DATABASES_REL)

    def test_ensure_datasets_downloads_into_it(self):
        text = _read("scripts", "pipeline", "meeting01", "ensure_datasets.sh")
        m = re.search(r'^DATASETS_ROOT="\$\{DATASETS_ROOT:-([^}]+)\}"$', text, re.M)
        self.assertIsNotNone(m, "ensure_datasets.sh has no DATASETS_ROOT default")
        self.assertTrue(m.group(1).endswith("/" + databases_dir.DATABASES_REL), m.group(1))

    def test_every_profile_reads_its_base_from_it(self):
        want = {"coretto2017": "~/" + databases_dir.DATABASES_REL + "/database.sqlite",
                "biomex2021": "~/" + databases_dir.DATABASES_REL + "/biomex.sqlite"}
        files = subprocess.run(["git", "ls-files", "src/experiments/*/profiles/*.json",
                                "src/experiments/*/profiles/**/*.json"], cwd=NN, text=True,
                               capture_output=True, check=True).stdout.split()
        checked = 0
        for rel in sorted(set(files)):
            with open(os.path.join(NN, rel), encoding="utf-8") as f:
                ds = json.load(f).get("dataset", {})
            root = ds.get("root", "")
            if not root.endswith(".sqlite"):
                continue  # meeting01 profiles name directories of corpora, checked elsewhere
            checked += 1
            self.assertEqual(root, want[ds.get("source", "coretto2017")], rel)
        self.assertGreater(checked, 900, "found suspiciously few thesis/GA profiles")

    def test_scripts_default_into_it(self):
        for rel in ("biomex_download.py", "biomex_to_sqlite.py", "biomex_verify.py",
                    "biomex_inventory.py", "coretto_verify_mirror.py"):
            # a wide terminal, so argparse does not wrap the path across two lines
            out = subprocess.run([sys.executable, os.path.join(NN, "scripts", "data", rel), "--help"],
                                 capture_output=True, text=True, check=True,
                                 env={**os.environ, "COLUMNS": "400"}).stdout
            self.assertIn(databases_dir.databases_dir(), out, rel)


if __name__ == "__main__":
    unittest.main()
