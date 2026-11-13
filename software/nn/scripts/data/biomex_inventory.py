#!/usr/bin/env python3
"""Inventory of a BIOMEX-DB extraction: what each volunteer's folder actually holds.

WHY. The paper (Moreno-Rodriguez et al., IEEE Access 9:111267, 2021, Table 2) describes two
EEG files and twenty voice files per volunteer: one 10-digit session (01G10) and one 4-digit
session (01G04). The distributed folders do not all look like that: some hold a second
session on another day (02G10 / 02G04), some hold two complete recordings of one session,
some files carry another volunteer's name. Before any rule decides which recording is a
volunteer's, this lists every one of them, so the rule is chosen on the whole base and not
case by case.

For each volunteer, one line per EDF: the session its name gives, the name's own volunteer
id, the subject field of its CSV export (if any), start, length, the digit runs it holds,
and whether those runs are the complete protocol (10 runs of 10 or of 4 digits). Then the
voice files per session code. A volunteer is "regular" if it holds exactly one complete EDF
for 01G10 and one for 01G04, both named for it, and 20 voice series for those sessions only.

Usage (from software/nn):
    python3 scripts/data/biomex_inventory.py --src ~/biomex/extracted [--csv out.csv]
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biomex_to_sqlite as bx  # noqa: E402

EDF_NAME = re.compile(r"^(?P<who>[FM]\d+)-(?P<session>\w+?)-(?P<stamp>[\d.]+)\.edf$")
VOICE_NAME = re.compile(r"^(?P<who>[FM]\d+)_(?P<session>\w+?)_(?P<k>\d+)\.(?P<ext>wav|lab)$")


def csv_subject(folder, who, session):
    path = os.path.join(folder, "EEG", f"{who}-{session}.CSV")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        head = f.readline()
    fields = {k.strip(): v for k, v in (p.split(":", 1) for p in head.split(",") if ":" in p)}
    return fields.get("subject", "").strip(), fields.get("recorded", "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--csv", help="also write one row per EDF here")
    args = ap.parse_args()
    src = os.path.expanduser(args.src)
    rows = []
    irregular = []
    for vid in sorted(d for d in os.listdir(src) if bx.VOLUNTEER.match(d)):
        folder = os.path.join(src, vid)
        complete = Counter()
        notes = []
        print(f"{vid}")
        for path in sorted(glob.glob(os.path.join(folder, "EEG", "*.edf"))):
            name = os.path.basename(path)
            m = EDF_NAME.match(name)
            labels, phys, rate, start, raw = bx.read_edf(path)
            runs = bx.digit_runs(raw[labels.index("MARKER")], name)
            lens = [len(r) for r in runs]
            session = m["session"] if m else "?"
            per = {"G10": 10, "G04": 4}.get(session[-3:])
            full = per is not None and lens == [per] * 10
            exported = csv_subject(folder, m["who"] if m else vid, session)
            row = {"volunteer": vid, "file": name, "name_id": m["who"] if m else "?",
                   "session": session, "start": start, "seconds": phys.shape[1] / rate,
                   "runs": "x".join(map(str, lens)) or "-", "complete": full,
                   "csv_subject": exported[0] if exported else "",
                   "csv_recorded": exported[1] if exported else ""}
            rows.append(row)
            if full:
                complete[(session, m["who"] if m else "?")] += 1
            print(f"   {name:38} {start}  {row['seconds']:6.1f} s  runs {row['runs']:>20}  "
                  f"{'COMPLETE' if full else 'partial ':8}  csv: "
                  f"{exported[0] + ' ' + exported[1] if exported else '-'}")
        voice = Counter()
        for path in glob.glob(os.path.join(folder, "Voice", "*")):
            m = VOICE_NAME.match(os.path.basename(path))
            if m and m["ext"] == "wav":
                voice[(m["who"], m["session"])] += 1
            elif not m:
                notes.append(f"voice file {os.path.basename(path)}")
        print(f"   voice series: {dict(sorted(voice.items()))}")
        own = [r for r in rows if r["volunteer"] == vid]
        if (len(own) == 2 and all(r["complete"] for r in own)
                and complete == Counter({("01G10", vid): 1, ("01G04", vid): 1})
                and voice == Counter({(vid, "01G10"): 10, (vid, "01G04"): 10}) and not notes):
            continue
        irregular.append(vid)
        if notes:
            print(f"   other: {notes}")
    print(f"\n{len(irregular)} irregular volunteer(s): {', '.join(irregular)}")
    if args.csv:
        with open(os.path.expanduser(args.csv), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
