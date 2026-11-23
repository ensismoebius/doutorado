#!/usr/bin/env python3
"""thesis_grid_status.py -- live progress of the thesis Slurm arrays on GridUnesp.

Why. A thesis phase (thesis_gridunesp.sbatch) is an array of up to 150 one-CPU tasks, each
running a few "units" (a profile, or a profile's repetition) in a row. squeue shows tasks, not
units, and says nothing about which unit a task is on, how long units take, how close a task is
to the partition's time limit, or how much memory it uses. This script puts that on one screen:

    pfworld  job 5345042  revision 271cbb5b/6f3ed3f0bc66 threads=1
      tasks   26 running   124 pending (Resources)   0 finished
      units   [######..............................]  12/150 done (8%)   0 failed
              mean 14.9 h per unit (12 measured)   ~3.1 d left at 26 in parallel (estimate)
      task  unit                                  unit time   task time / limit    memory
         0  pf01w_earlyimag_probes_aphonic-r05      2:13:05   2:13:05 / 1-00:00:00   3.1 GB
        ...

It is written for the cluster's own python3 (3.6, no third-party module) and is normally not
run by hand: thesis_grid_watch.sh pipes it into ONE ssh session, so the refresh loop runs on the
login node. GridUnesp's Fail2Ban locks repeated logins for 15 minutes; a local loop that
reconnects every refresh would trip it (.wiki/Guides/GridUnesp-Deployment.md).

What it reads, from ~/software/nn on the cluster (nothing is written):
  squeue -r            every task of the user's thesis-run / thesis-warm jobs, one row each
  sstat                the peak memory (MaxRSS) of every running task's batch step
  sacct                the end state and peak memory of tasks that finished (COMPLETED,
                       OUT_OF_MEMORY, TIMEOUT, ...), over the last --since-hours
  thesis_run_<job>_<task>.out, thesis_warm_<job>_<task>.out
                       the phase (header line), the unit a task is on ("start <unit>") and
                       how long finished units took ("done <unit> (<s> s)")
  results/thesis/<phase>/.done, .failed
                       the units of record: a .done marker is what thesis_gridunesp.sbatch
                       writes after a unit exited 0 with every expected file
  src/experiments/thesis/profiles/<phase>, thesis_gridunesp.sbatch
                       how many units a phase has: one per profile, or one per repetition for
                       the phases the sbatch splits (its SPLIT_REPEATS case is parsed, not
                       copied, so the two cannot disagree)

Nothing is guessed: a job whose .out files do not name a phase is shown as "phase unknown"
with its tasks, and a figure that cannot be computed (no unit finished yet) prints "-".

Usage (on the cluster; normally through thesis_grid_watch.sh):
  python3 scripts/pipeline/thesis/thesis_grid_status.py [--interval 60] [--once]
      [--since-hours 48] [--jobs 5345042,5338456] [--max-rows 40]
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

NAMES = ("thesis-run", "thesis-warm")
NEAR_LIMIT = 0.85  # flag a task past this share of its time limit
HEADER = re.compile(r"^\[thesis-(grid|warm)\] (\S+) task (\d+)/(\d+), revision (.*?):")
WARM_HEADER = re.compile(r"^\[thesis-warm\] \S+ \S+ task (\d+)/(\d+): (\S+) revision (\S+)")
START = re.compile(r"^\[thesis-grid\] (\S+ \S+) start (\S+)$")
DONE = re.compile(r"^\[thesis-(?:grid|warm)\] (\S+ \S+) done (\S+) \((\d+) s\)$")


def run(cmd):
    """stdout of a command; a non-zero exit is reported, not hidden."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    if p.returncode != 0:
        raise RuntimeError("{} failed ({}): {}".format(" ".join(cmd), p.returncode,
                                                         p.stderr.strip()))
    return p.stdout


def slurm_seconds(text):
    """'1-02:03:04', '02:03:04', '03:04' -> seconds; '' or UNLIMITED -> None."""
    if not text or text in ("UNLIMITED", "INVALID", "Partition_Limit"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(float(x)) for x in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def rss_bytes(text):
    """sacct/sstat MaxRSS ('3255088K', '1.2G') -> bytes; '' -> None."""
    if not text:
        return None
    units = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def hms(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    d, rest = divmod(seconds, 86400)
    h, rest = divmod(rest, 3600)
    m, s = divmod(rest, 60)
    return "{}-{:02d}:{:02d}:{:02d}".format(d, h, m, s) if d else "{}:{:02d}:{:02d}".format(h, m, s)


def human_span(seconds):
    if seconds is None:
        return "-"
    if seconds >= 86400:
        return "{:.1f} d".format(seconds / 86400)
    if seconds >= 3600:
        return "{:.1f} h".format(seconds / 3600)
    return "{:.0f} min".format(seconds / 60)


def gb(b):
    return "-" if b is None else "{:.1f} GB".format(b / 1024 ** 3)


# ── collection ────────────────────────────────────────────────────────────────────────────────

def queue():
    """{array_job: {task: row}} of the user's thesis jobs still in the queue."""
    out = run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r",
               "-o", "%F|%K|%A|%j|%T|%M|%l|%R"])
    jobs = {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) != 8 or f[3] not in NAMES:
            continue
        array, task, raw, name, state, elapsed, limit, reason = f
        task = int(task) if task.isdigit() else 0
        jobs.setdefault(array, {})[task] = {
            "raw": raw, "name": name, "state": state, "elapsed": slurm_seconds(elapsed),
            "limit": slurm_seconds(limit), "reason": reason}
    return jobs


def running_rss(raws):
    """{raw job id: MaxRSS bytes} of running batch steps (one sstat call)."""
    if not raws:
        return {}
    out = run(["sstat", "-n", "-P", "-a", "-o", "JobID,MaxRSS",
               "-j", ",".join(r + ".batch" for r in raws)])
    rss = {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) == 2 and f[0].endswith(".batch"):
            rss[f[0][:-len(".batch")]] = rss_bytes(f[1])
    return rss


def accounting(since_hours):
    """{array_job: {task: {state, elapsed, limit, rss, name}}} of thesis tasks since then."""
    out = run(["sacct", "-u", os.environ.get("USER", ""), "-n", "-P",
               "-S", "now-{}hours".format(since_hours),
               "-o", "JobID,JobName,State,Elapsed,Timelimit,MaxRSS"])
    tasks, rss = {}, {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) != 6:
            continue
        jid, name, state, elapsed, limit, maxrss = f
        if jid.endswith(".batch"):
            rss[jid[:-len(".batch")]] = rss_bytes(maxrss)
            continue
        if "." in jid or name not in NAMES or "_" not in jid or "[" in jid:
            continue
        array, task = jid.split("_", 1)
        tasks.setdefault(array, {})[int(task)] = {
            "name": name, "state": state.split()[0], "elapsed": slurm_seconds(elapsed),
            "limit": slurm_seconds(limit)}
    for array, by_task in tasks.items():
        for task, row in by_task.items():
            row["rss"] = rss.get("{}_{}".format(array, task))
    return tasks


def read_logs(array, name):
    """Per task of one array job: phase, revision, current unit + its start, finished units."""
    prefix = "thesis_run" if name == "thesis-run" else "thesis_warm"
    logs = {}
    for path in glob.glob("{}_{}_*.out".format(prefix, array)):
        m = re.search(r"_(\d+)\.out$", path)
        if not m:
            continue
        task = int(m.group(1))
        info = {"phase": None, "revision": None, "current": None, "started": None, "done": []}
        try:
            with open(path, errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        for line in lines:
            h = HEADER.match(line)
            if h and h.group(1) == "grid":
                info["phase"], info["revision"] = h.group(2), h.group(5)
                continue
            w = WARM_HEADER.match(line)
            if w:
                info["phase"], info["revision"] = "warm", w.group(4)
                info["current"] = os.path.basename(w.group(3))
                continue
            s = START.match(line)
            if s:
                info["current"] = s.group(2)
                info["started"] = time.mktime(time.strptime(s.group(1), "%Y-%m-%d %H:%M:%S"))
                continue
            d = DONE.match(line)
            if d:
                info["done"].append((d.group(2), int(d.group(3))))
                if d.group(2) == info["current"] or info["phase"] == "warm":
                    info["current"], info["started"] = None, None
        logs[task] = info
    return logs


def split_phases(sbatch="scripts/pipeline/thesis/thesis_gridunesp.sbatch"):
    """{phase: True if split by repetition} from the sbatch's own SPLIT_REPEATS case."""
    split = {}
    with open(sbatch) as fh:
        for line in fh:
            m = re.match(r"^\s*([a-z0-9|]+)\) SPLIT_REPEATS=([01]) ;;", line)
            if m:
                for phase in m.group(1).split("|"):
                    split[phase] = m.group(2) == "1"
    if not split:
        raise RuntimeError(sbatch + ": no SPLIT_REPEATS case found; the sbatch changed shape")
    return split


def phase_units(phase, split):
    """How many units a phase has, from its profiles; None if it cannot be told."""
    profiles = sorted(glob.glob("src/experiments/thesis/profiles/{}/*.json".format(phase)))
    if not profiles or phase not in split:
        return None
    if not split[phase]:
        return len(profiles)
    total = 0
    for path in profiles:
        with open(path) as fh:
            total += int(json.load(fh)["experiment"]["repeats"])
    return total


def markers(phase):
    """(done count, failed unit names, the revisions the .done markers hold)."""
    root = "results/thesis/{}".format(phase)
    done = glob.glob(root + "/.done/*.done")
    failed = sorted(os.path.basename(p)[:-len(".failed")]
                    for p in glob.glob(root + "/.failed/*.failed"))
    revisions = set()
    for path in done:
        with open(path) as fh:
            revisions.add(fh.read().strip())
    return len(done), failed, revisions


# ── one frame ─────────────────────────────────────────────────────────────────────────────────

def bar(done, total, width=36):
    if not total:
        return "[" + "?" * width + "]"
    filled = int(round(width * min(done, total) / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def frame(args, split):
    q = queue()
    acct = accounting(args.since_hours)
    wanted = set(args.jobs.split(",")) if args.jobs else set(q) | set(acct)
    running_raws = [row["raw"] for by_task in q.values() for row in by_task.values()
                    if row["state"] == "RUNNING"]
    rss_now = running_rss(running_raws)

    lines = ["thesis jobs on GridUnesp  --  {}  (refresh {} s; Ctrl-C to quit)".format(
        time.strftime("%Y-%m-%d %H:%M:%S"), args.interval if not args.once else "-")]
    if not wanted:
        lines.append("")
        lines.append("no thesis-run / thesis-warm job queued or finished in the last {} h"
                     .format(args.since_hours))
        return "\n".join(lines)

    for array in sorted(wanted, key=lambda j: int(j) if j.isdigit() else 0, reverse=True):
        queued = q.get(array, {})
        finished = acct.get(array, {})
        names = {r["name"] for r in list(queued.values()) + list(finished.values())}
        name = sorted(names)[0] if names else "thesis-run"
        logs = read_logs(array, name)
        phases = {i["phase"] for i in logs.values() if i["phase"]}
        revisions = {i["revision"] for i in logs.values() if i["revision"]}
        phase = phases.pop() if len(phases) == 1 else None
        lines.append("")
        lines.append("{}  job {}  {}".format(
            phase or ("phase unknown" if not phases else "phases " + ",".join(sorted(phases))),
            array, "revision " + ", ".join(sorted(revisions)) if revisions else ""))

        states = {}
        for task, row in finished.items():
            if task not in queued:
                states[row["state"]] = states.get(row["state"], 0) + 1
        running = {t: r for t, r in queued.items() if r["state"] == "RUNNING"}
        pending = {t: r for t, r in queued.items() if r["state"] == "PENDING"}
        reasons = sorted({r["reason"].strip("()") for r in pending.values()})
        ended = ", ".join("{} {}".format(n, s) for s, n in sorted(states.items())) or "0"
        lines.append("  tasks   {} running   {} pending{}   finished in the last {} h: {}".format(
            len(running), len(pending), " (" + ", ".join(reasons) + ")" if reasons else "",
            args.since_hours, ended))

        durations = [s for info in logs.values() for _, s in info["done"]]
        mean = sum(durations) / len(durations) if durations else None
        done, failed, marker_revs = markers(phase) if phase and phase != "warm" else (0, [], set())
        foreign = marker_revs and revisions and not (marker_revs <= revisions)
        if phase and phase != "warm" and foreign:
            # The phase directory belongs to another deploy (this job's own directory was moved
            # aside, as the sbatch requires): its markers are not this job's progress.
            lines.append("  units   not this job's: results/thesis/{} holds markers of {}".format(
                phase, ", ".join(sorted(marker_revs))))
        elif phase and phase != "warm":
            total = phase_units(phase, split)
            pct = " ({:.0f}%)".format(100.0 * done / total) if total else ""
            lines.append("  units   {}  {}/{} done{}   {} failed".format(
                bar(done, total), done, total if total is not None else "?", pct, len(failed)))
            left = None
            if mean is not None and total is not None and running:
                left = (total - done) * mean / len(running)
            lines.append("          mean {} per unit ({} measured in this job)   {}".format(
                human_span(mean), len(durations),
                "~{} left at {} in parallel (estimate)".format(human_span(left), len(running))
                if left is not None else "time left: -"))
            if failed:
                shown = ", ".join(failed[:6]) + (" ..." if len(failed) > 6 else "")
                lines.append("  FAILED  " + shown)
        elif phase == "warm":
            lines.append("  groups  {} done in this job (mean {})".format(
                len(durations), human_span(mean)))

        peak = [r["rss"] for r in finished.values() if r.get("rss")]
        if peak:
            lines.append("  memory  peak of finished tasks {}".format(gb(max(peak))))

        if running:
            lines.append("  {:>4}  {:<44} {:>10}  {:>23}  {:>8}".format(
                "task", "unit", "unit time", "task time / limit", "memory"))
            now = time.time()
            for i, (task, row) in enumerate(sorted(running.items())):
                if i >= args.max_rows:
                    lines.append("        ... {} more running".format(len(running) - i))
                    break
                info = logs.get(task, {})
                unit = info.get("current") or "-"
                started = info.get("started")
                unit_time = hms(now - started) if started else "-"
                limit = row["limit"]
                flag = ""
                if limit and row["elapsed"] is not None and row["elapsed"] > NEAR_LIMIT * limit:
                    flag = "  NEAR LIMIT"
                lines.append("  {:>4}  {:<44} {:>10}  {:>10} / {:<10}  {:>8}{}".format(
                    task, unit[:44], unit_time, hms(row["elapsed"]), hms(limit),
                    gb(rss_now.get(row["raw"])), flag))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--interval", type=int, default=60, help="seconds between refreshes")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--since-hours", type=int, default=48,
                    help="also show jobs that finished in this many hours")
    ap.add_argument("--jobs", help="only these array job ids, comma-separated")
    ap.add_argument("--max-rows", type=int, default=40, help="running tasks listed per job")
    args = ap.parse_args()
    if args.interval < 15:
        ap.error("--interval below 15 s only loads the Slurm controller")
    if not os.path.isfile("scripts/pipeline/thesis/thesis_gridunesp.sbatch"):
        ap.error("run from ~/software/nn (thesis_gridunesp.sbatch not found here)")
    split = split_phases()
    if args.once:
        print(frame(args, split))  # an error here is the exit status, not a printed frame
        return 0
    while True:
        try:
            text = frame(args, split)
        except Exception as exc:  # shown on screen in place of the frame; the next cycle retries
            text = "ERROR while collecting (next try in {} s): {}".format(args.interval, exc)
        sys.stdout.write("\033[H\033[2J" + text + "\n")
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
