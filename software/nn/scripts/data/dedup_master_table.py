#!/usr/bin/env python3
"""dedup_master_table.py — Deduplicate grid-search CSV by profile signature.

Reads analysis/master_comparison.csv (output of Experiment 03 grid search),
strips the YYYYMMDD_HHMMSS timestamp prefix from profile names, groups rows by
the remaining profile signature, and keeps only the first occurrence of each.
Output written to analysis/master_comparison_dedup_by_profile.csv.

Usage:
    python scripts/data/dedup_master_table.py

Input/output paths are hardcoded; run from the repo root (software/nn/).
"""
import csv
from collections import defaultdict
from pathlib import Path

infile = Path("analysis/master_comparison.csv")
outfile = Path("analysis/master_comparison_dedup_by_profile.csv")

rows = list(csv.DictReader(infile.open(newline="")))
bucket = defaultdict(list)

for row in rows:
    profile = row.get("Profile", "")
    parts = profile.split("_")
    # Timestamped names follow: YYYYMMDD_HHMMSS_<profile-signature>
    if len(parts) > 2 and parts[0].isdigit() and len(parts[0]) == 8 and parts[1].isdigit() and len(parts[1]) == 6:
        sig = "_".join(parts[2:])
    elif "_" in profile and profile[:8].isdigit():
        sig = profile.split("_", 1)[1]
    else:
        sig = profile
    row["_sig"] = sig
    try:
        row["_mean"] = float(row.get("Mean Val Loss", "inf"))
    except Exception:
        row["_mean"] = float("inf")
    row["_ok"] = row.get("Status", "") == "Success"
    bucket[sig].append(row)

chosen = []
for sig, rs in bucket.items():
    succ = [x for x in rs if x["_ok"]]
    cand = succ if succ else rs
    cand.sort(key=lambda x: (x["_mean"], x.get("Rank", "999999")))
    chosen.append(cand[0])

chosen.sort(key=lambda x: (x["_mean"], x.get("Profile", "")))

fields = [
    "Rank",
    "Profile",
    "Profile Signature",
    "Modality",
    "Loss Type",
    "LR",
    "BS",
    "HS",
    "LS",
    "Final Train Loss",
    "Final Val Loss",
    "Mean Val Loss",
    "Best Val Loss",
    "Epochs",
    "Status",
    "Error",
    "Duplicate Runs",
]

with outfile.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, row in enumerate(chosen, 1):
        sig = row["_sig"]
        w.writerow(
            {
                "Rank": i,
                "Profile": row.get("Profile", ""),
                "Profile Signature": sig,
                "Modality": row.get("Modality", ""),
                "Loss Type": row.get("Loss Type", ""),
                "LR": row.get("LR", ""),
                "BS": row.get("BS", ""),
                "HS": row.get("HS", ""),
                "LS": row.get("LS", ""),
                "Final Train Loss": row.get("Final Train Loss", ""),
                "Final Val Loss": row.get("Final Val Loss", ""),
                "Mean Val Loss": row.get("Mean Val Loss", ""),
                "Best Val Loss": row.get("Best Val Loss", ""),
                "Epochs": row.get("Epochs", ""),
                "Status": row.get("Status", ""),
                "Error": row.get("Error", ""),
                "Duplicate Runs": len(bucket[sig]),
            }
        )

print(f"RAW_ROWS={len(rows)}")
print(f"DEDUP_ROWS={len(chosen)}")
print(f"OUT={outfile}")
