"""Unit tests for snn_hyperparam_search grid and CSV writing logic.

We test the grid construction and CSV writing in isolation — no dataset loading,
no model training.  The patterns mirror those in run_hyper_search.py.
"""

import csv
import io
import itertools


# ---- helpers replicated from run_hyper_search.py (pure logic, no imports needed) ----

def build_grid_combinations(grid: dict) -> list[dict]:
    """Produce all combinations from a hyperparameter grid dict."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def write_results_csv(fileobj, fieldnames: list[str], rows: list[dict]) -> int:
    """Write rows to a CSV file-like object.  Returns number of rows written."""
    writer = csv.DictWriter(fileobj, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return len(rows)


# ---- tests ----

class TestGridConstruction:
    def test_total_combinations(self):
        grid = {
            "lr": [1e-3, 5e-4],
            "hidden": [32, 64],
            "loss_mode": ["rate", "membrane", "van_rossum"],
            "num_passes": [1, 2],
        }
        combos = build_grid_combinations(grid)
        # 2 * 2 * 3 * 2 = 24
        assert len(combos) == 24

    def test_each_combo_has_all_keys(self):
        grid = {"lr": [1e-3], "hidden": [32, 64]}
        keys = set(grid.keys())
        for combo in build_grid_combinations(grid):
            assert set(combo.keys()) == keys

    def test_single_value_grid(self):
        grid = {"lr": [0.01], "epochs": [5]}
        combos = build_grid_combinations(grid)
        assert len(combos) == 1
        assert combos[0] == {"lr": 0.01, "epochs": 5}

    def test_empty_grid_gives_one_empty_combo(self):
        # itertools.product of no iterables gives one empty tuple
        combos = build_grid_combinations({})
        assert len(combos) == 1
        assert combos[0] == {}

    def test_all_lr_values_present(self):
        lr_values = [1e-3, 5e-4, 1e-4]
        grid = {"lr": lr_values, "hidden": [32]}
        combos = build_grid_combinations(grid)
        found_lrs = {c["lr"] for c in combos}
        assert found_lrs == set(lr_values)


class TestCsvWriting:
    FIELDNAMES = ["lr", "hidden", "loss_mode", "num_passes", "val_acc", "elapsed"]

    def _make_fake_rows(self, n: int = 3) -> list[dict]:
        return [
            {
                "lr": 1e-3,
                "hidden": 32,
                "loss_mode": "rate",
                "num_passes": 1,
                "val_acc": 0.5 + i * 0.1,
                "elapsed": float(i),
            }
            for i in range(n)
        ]

    def test_rows_written_count(self):
        buf = io.StringIO()
        rows = self._make_fake_rows(5)
        n = write_results_csv(buf, self.FIELDNAMES, rows)
        assert n == 5

    def test_csv_has_header(self):
        buf = io.StringIO()
        write_results_csv(buf, self.FIELDNAMES, self._make_fake_rows(2))
        buf.seek(0)
        reader = csv.DictReader(buf)
        assert reader.fieldnames == self.FIELDNAMES

    def test_csv_values_roundtrip(self):
        rows = self._make_fake_rows(3)
        buf = io.StringIO()
        write_results_csv(buf, self.FIELDNAMES, rows)
        buf.seek(0)
        reader = csv.DictReader(buf)
        read_rows = list(reader)
        assert len(read_rows) == 3
        for orig, read in zip(rows, read_rows):
            assert abs(float(read["val_acc"]) - orig["val_acc"]) < 1e-6

    def test_csv_empty_rows(self):
        buf = io.StringIO()
        n = write_results_csv(buf, self.FIELDNAMES, [])
        assert n == 0
        buf.seek(0)
        content = buf.read()
        # Header only (plus newline)
        assert ",".join(self.FIELDNAMES) in content
