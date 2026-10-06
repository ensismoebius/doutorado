#!/usr/bin/env python3
"""collect_ga_local.py — one-shot SSH collector for GA cache data from GridUnesp.

SSHes into GridUnesp, runs collect_ga() from gridunesp_status_remote.py inside
the remote conda env, and appends the result to a local JSONL file the dashboard
can read.

Usage:
    python3 collect_ga_local.py [--run-tag meeting01_loso] [--results-dir results/meeting01]
                                [--output-dir DIR] [--interval 0]

Output file: <output_dir>/<run_tag>_ga_remote.jsonl
Each line: {"cells": [...], "ts": <unix>, "source": "remote"}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from gridunesp_config import GridUnespConfig, load_config

# Bound on how long one SSH round-trip may take.  A login-node glob over the
# results dir is fast, but the SSH handshake plus a cold conda activate is not.
SSH_TIMEOUT_S = 60


def _finite_or_none(v: Any) -> Any:
    """NaN/Inf -> None, recursively.

    Mirrors monitor.py's guard.  Without it json.dumps emits bare `NaN` tokens,
    which are invalid JSON and make JSONResponse raise on the way out.
    """
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite_or_none(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite_or_none(x) for x in v]
    return v


def build_remote_cmd(cfg: GridUnespConfig, results_dir: str, run_tag: str) -> str:
    """Build the remote shell command that runs collect_ga().

    Four things this has to get right, each of which is a documented GridUnesp
    trap (see Guides/GridUnesp-Deployment.md):

    1. ``module load`` + ``eval "$(conda shell.bash hook)"`` + ``conda activate``
       — without the middle line ``conda activate`` fails outright, and without
       either the system python3 (3.6.8) runs instead, which cannot parse
       ``from __future__ import annotations``.
    2. The Python is fed on **stdin** via a quoted heredoc, not ``python3 -c``
       with nested quotes — one less layer of shell/Python/string quoting to get
       wrong, and the payload is not exposed in the remote process's argv.
    3. Exactly one quoting context per interpolated value.  ``cfg.remote_dir``
       lands in *shell* position (``cd <root>``) and needs shlex.quote; the
       results path and run tag land inside the *Python* body and need repr().
       Running shlex.quote over a value that repr() then re-quotes yields a
       string with literal quote characters embedded in it — a silently wrong
       path, not a loud failure.
    4. Because the heredoc delimiter is quoted (``<<'EOF'``), the shell performs
       no expansion inside the Python body, so repr() alone makes the
       caller-supplied ``results_dir`` (an HTTP query string) shell-inert.
    """
    # Shell context — appears as a bare word in the command line.
    remote_root = shlex.quote(cfg.remote_dir)
    # Python context — inside the literal heredoc body, so repr() is the only
    # quoting needed and the quoted delimiter neutralises the shell.
    remote_results = os.path.join(cfg.remote_dir, results_dir)
    py = (
        "import json, sys\n"
        "sys.path.insert(0, 'scripts/pipeline/meeting01')\n"
        "from gridunesp_status_remote import collect_ga\n"
        f"json.dump(collect_ga({remote_results!r}, {run_tag!r}), sys.stdout)\n"
    )
    return (
        f"cd {remote_root} && {cfg.remote_activate()} && "
        f"python3 - <<'MEETING01_PY_EOF'\n{py}MEETING01_PY_EOF"
    )


def _parse_remote_stdout(stdout: str) -> dict[str, Any]:
    """Extract the JSON object from SSH stdout.

    Scans lines from the end and returns the first that parses as a JSON
    object, so a login-node MOTD, a module-load banner or a conda warning
    printed ahead of the payload does not break parsing.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            return rec
    raise ValueError("no JSON object found in remote stdout")


def _ssh_error_tail(stderr: str, limit: int = 900) -> str:
    """Extract the part of ssh's stderr a human actually needs.

    Cluster SSH prepends ~500 bytes of fixed noise to *every* connection: an
    OpenSSH post-quantum-key-exchange warning and a `====` banner around the
    `module load` conda notice.  A naive head-truncation therefore spends the
    whole budget on noise and hides the real failure — which, when collecting
    remotely, is always a Python traceback at the very *end*.  So: drop the known
    boilerplate, then keep the tail.
    """
    noise = (
        "post-quantum key exchange",
        "store now, decrypt later",
        "openssh.com/pq.html",
        "You are using Miniconda with libmamba-server",
        "Installation proccess",
        "is supposed to be faster",
        "After creation of environment",
        "source activate evironment_name",
        "conda activate evironment_name",
    )
    kept = []
    for line in stderr.splitlines():
        s = line.strip()
        if not s or set(s) == {"="}:
            continue
        if any(n in s for n in noise):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()
    return text[-limit:] if len(text) > limit else text


def collect_once(
    cfg: GridUnespConfig, results_dir: str, run_tag: str
) -> Optional[dict[str, Any]]:
    """SSH into GridUnesp and run the GA collection.  Returns None on failure."""
    remote_cmd = build_remote_cmd(cfg, results_dir, run_tag)
    try:
        argv = cfg.ssh_argv(remote_cmd)
    except RuntimeError as exc:
        print(f"[collect_ga] {exc}", file=sys.stderr)
        return None

    try:
        out = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT_S,
            env=cfg.ssh_env(),
        )
    except subprocess.TimeoutExpired:
        print(f"[collect_ga] SSH timeout after {SSH_TIMEOUT_S}s", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"[collect_ga] failed to launch ssh: {exc}", file=sys.stderr)
        return None

    if out.returncode != 0:
        detail = _ssh_error_tail(out.stderr)
        print(
            f"[collect_ga] SSH error (rc={out.returncode})"
            + (f":\n{detail}" if detail else " (no stderr; check host/port and "
               "that BatchMode/password auth is configured)"),
            file=sys.stderr,
        )
        return None

    try:
        data = _parse_remote_stdout(out.stdout)
    except ValueError as exc:
        print(f"[collect_ga] {exc}", file=sys.stderr)
        print(f"[collect_ga] stdout was: {out.stdout.strip()[-400:]!r}", file=sys.stderr)
        return None

    data["ts"] = time.time()
    data["source"] = "remote"
    return _finite_or_none(data)


def append_record(output_path: str, data: dict[str, Any]) -> None:
    """Append one record as a single line, creating the file if needed.

    Opened in "a" mode with a single write() so a record is never interleaved or
    split across a crash — the dashboard reads the last *complete* line.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-tag", default="meeting01_loso")
    ap.add_argument("--results-dir", default="results/meeting01",
                    help="Results dir path, relative to the REMOTE checkout root")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the local JSONL (default: --results-dir)")
    ap.add_argument("--interval", type=float, default=0,
                    help="0 = one-shot; >0 = poll every N seconds")
    ap.add_argument("--print-cmd", action="store_true",
                    help="Print the remote command and exit (no SSH)")
    args = ap.parse_args()

    output_dir = args.output_dir or args.results_dir
    output_path = os.path.join(output_dir, f"{args.run_tag}_ga_remote.jsonl")

    try:
        cfg = load_config()
    except SystemExit as exc:
        print(f"[collect_ga] configuration error: {exc}", file=sys.stderr)
        return 1

    if args.print_cmd:
        print(build_remote_cmd(cfg, args.results_dir, args.run_tag))
        return 0

    print(f"[collect_ga] target: {cfg.ssh_target}  output: {output_path}")

    while True:
        data = collect_once(cfg, args.results_dir, args.run_tag)
        if data:
            append_record(output_path, data)
            print(f"[collect_ga] {len(data.get('cells', []))} cells "
                  f"written at {data['ts']:.0f}")
        else:
            print("[collect_ga] no data this cycle")

        if args.interval <= 0:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
