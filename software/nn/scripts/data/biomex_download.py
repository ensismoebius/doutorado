#!/usr/bin/env python3
"""Download BIOMEX-DB from Mendeley Data and keep only its EEG and voice.

WHY. The base is 51 RAR archives, one per volunteer, 8.1 GB in all; almost all of it is
video, which the thesis does not use (the 12 volunteers without video are ~8 MB each).
This script fetches each archive, checks its size against the listing, extracts the EEG and
Voice folders, and deletes the archive unless --keep-archives is given. A volunteer whose
folders were extracted completely is skipped on a rerun, so an interrupted download resumes.

WHERE THE FILES ARE. The dataset's version root is empty (that is why its web page shows no
files and its size as 0); the archives sit in one folder, BIOMEX:
    public-api/datasets/s7chktmb6x/folders/1            -> folder id of "BIOMEX"
    public-api/datasets/s7chktmb6x/files?folder_id=<id>&version=1  -> 51 files

Licence: CC BY 4.0. Cite Moreno-Rodriguez et al., IEEE Access 9:111267 (2021) and the data
DOI 10.17632/s7chktmb6x.1, and state the changes (scripts/data/biomex_to_sqlite.py).

Usage (from software/nn):
    python3 scripts/data/biomex_download.py --dest ~/biomex
then
    python3 scripts/data/biomex_to_sqlite.py --src ~/biomex/extracted --out ~/biomex.sqlite
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://data.mendeley.com/public-api/datasets/s7chktmb6x"
VERSION = 1
FOLDER = "BIOMEX"
EXPECTED_FILES = 51
DONE = ".extracted"
UA = {"User-Agent": "thesis-dataset-fetch/1.0"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def listing():
    folders = [f for f in get_json(f"{API}/folders/{VERSION}") if f["name"] == FOLDER]
    if len(folders) != 1:
        raise SystemExit(f"expected one folder named {FOLDER}, found {len(folders)}")
    files = get_json(f"{API}/files?folder_id={folders[0]['id']}&version={VERSION}")
    if len(files) != EXPECTED_FILES:
        raise SystemExit(f"expected {EXPECTED_FILES} archives, the listing has {len(files)}")
    return files


ATTEMPTS = 5


def fetch(url, tmp):
    """Fetch url into tmp. Mendeley's gateway answers 502/503 now and then mid-run (seen at
    archive 7 of 51); such a reply, or a dropped connection, is retried a bounded number of
    times, each retry printed. Any 4xx, or the last failure, is raised."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            return
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == ATTEMPTS:
                raise
            why = f"HTTP {e.code}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt == ATTEMPTS:
                raise
            why = type(e).__name__
        wait = 30 * attempt
        print(f"[biomex]   {why}; retry {attempt}/{ATTEMPTS - 1} in {wait} s", flush=True)
        time.sleep(wait)


def download(url, path, size):
    tmp = path + ".partial"
    fetch(url, tmp)
    got = os.path.getsize(tmp)
    if got != size:
        os.remove(tmp)
        raise SystemExit(f"{os.path.basename(path)}: {got} bytes, the listing says {size}")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", required=True, help="archives/ and extracted/ go here")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep each RAR after extraction (8.1 GB in all)")
    args = ap.parse_args()
    if shutil.which("bsdtar") is None:
        raise SystemExit("bsdtar (libarchive) is required to extract the RAR archives")
    dest = os.path.expanduser(args.dest)
    arch_dir = os.path.join(dest, "archives")
    out_dir = os.path.join(dest, "extracted")
    os.makedirs(arch_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    files = listing()
    total = sum(f["size"] for f in files)
    print(f"[biomex] {len(files)} archives, {total / 1e9:.2f} GB")
    for n, f in enumerate(sorted(files, key=lambda f: f["filename"]), 1):
        name = f["filename"]
        vid = os.path.splitext(name)[0]
        marker = os.path.join(out_dir, vid, DONE)
        if os.path.exists(marker):
            continue
        rar = os.path.join(arch_dir, name)
        if not (os.path.exists(rar) and os.path.getsize(rar) == f["size"]):
            print(f"[biomex] {n}/{len(files)} {name} ({f['size'] / 1e6:.0f} MB)", flush=True)
            download(f["content_details"]["download_url"], rar, f["size"])
        shutil.rmtree(os.path.join(out_dir, vid), ignore_errors=True)
        subprocess.run(["bsdtar", "-xf", rar, "-C", out_dir, f"{vid}/EEG/*", f"{vid}/Voice/*"],
                       check=True)
        for sub in ("EEG", "Voice"):
            if not os.listdir(os.path.join(out_dir, vid, sub)):
                raise SystemExit(f"{name}: {sub}/ extracted empty")
        open(marker, "w").close()
        if not args.keep_archives:
            os.remove(rar)
    print(f"[biomex] done -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
