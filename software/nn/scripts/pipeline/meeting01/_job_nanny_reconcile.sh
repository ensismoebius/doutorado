#!/usr/bin/env bash
# _job_nanny_reconcile.sh -- after a GridUnesp job, make ~/software/nn/results hold what the run
# left behind, not also what it deleted. Sourced by 00_meeting01_prevalidate_gridunesp.sbatch and
# 01_meeting01_run_loso_gridunesp.sbatch, and by the start.sh they write; tested by
#   bash scripts/pipeline/meeting01/_job_nanny_reconcile.sh --self-test
#
# THE PROBLEM. job-nanny copies the run's results/ home every WAIT_CHECKPOINT seconds and once at
# the end, with `rsync -a -u` and no --delete. A file the run writes and later removes therefore
# survives at home whenever a periodic copy caught it. The run removes exactly one kind of file:
# a GA search's resume artifacts (meeting01_ga_<tag>_checkpoint.json and _cache.jsonl), deleted when
# that search finishes. Pre-validation job 5317827 (2026-11-07) left 13 of each at home, every
# checkpoint older than its search's selection manifest:
#
#   time ─►  gen 1        periodic copy      gen 2, search done,       final copy
#   WORK_DIR ckpt(gen 1)  ───────────►       ckpt removed               ───────────►  (no ckpt)
#   home                  ckpt(gen 1)                                                  ckpt(gen 1)  <- stale
#
# Silent, not loud: a later RESUME=1 stages results/ from home, so a fold that is rerun "resumes"
# a search that had finished from that old generation instead of recomputing it as
# 01_meeting01_run_loso.sh documents -- its events say "resumed" for a search that was never cut,
# and the stamp check passes, because the stale file is from this very configuration.
#
# THE FIX, in two halves:
#   job_nanny_write_listing RESULTS_DIR JOBID
#       start.sh, inside job-nanny's WORK_DIR, after the run returns (a stopped run included):
#       writes RESULTS_DIR/.job-final-files-JOBID.txt, a header line and every regular file under
#       RESULTS_DIR (path relative to it), through a temp file + rename. job-nanny's final copy
#       brings it home with everything else.
#   job_nanny_reconcile RESULTS_ROOT SCOPE_DIR JOBID [NAME_GLOB ...]
#       the sbatch, after job-nanny returns: removes each regular file under SCOPE_DIR that the
#       listing does not name, is not newer than the listing, and whose name matches a NAME_GLOB
#       (none = every name). Each removal is printed. Then removes the listing.
#   Why the scope globs: in an array the 24 tasks share results/meeting01, and a task's listing
#   says nothing about the others' files -- each task reconciles only its own fold's names.
#   Why "not newer than the listing": rsync -a keeps the run's mtimes, so every stale copy is older
#   than the listing; a file written at home after the run ended is not this job's and is kept.
#
# What it cannot fix: a job-nanny that is SIGKILLed never makes its final copy, so no listing
# arrives and nothing is reconciled (said loudly). Then WORK_DIR is still on /store with the newest
# state; recover from there (.wiki/Guides/GridUnesp-Deployment.md, "Stopping").

# job_nanny_write_listing RESULTS_DIR JOBID
job_nanny_write_listing() {
  local dir="$1" job="$2" out tmp
  [[ -d "$dir" ]] || { echo "[reconcile] no results dir '$dir' to list" >&2; return 1; }
  [[ -n "$job" ]] || { echo "[reconcile] job_nanny_write_listing: no job id" >&2; return 1; }
  out="$dir/.job-final-files-$job.txt"
  tmp="$out.tmp"
  {
    echo "# job $job: regular files under results/ after the run"
    (cd "$dir" && find . -type f ! -name ".job-final-files-*" -printf '%P\n' | LC_ALL=C sort)
  } > "$tmp"
  mv -f -- "$tmp" "$out"
}

# job_nanny_reconcile RESULTS_ROOT SCOPE_DIR JOBID [NAME_GLOB ...]
job_nanny_reconcile() {
  local root="$1" scope="$2" job="$3"
  shift 3
  local listing="$root/.job-final-files-$job.txt"
  if [[ "$scope" != "$root"/* ]]; then
    echo "[reconcile] REFUSED: scope '$scope' is not under '$root'" >&2
    return 1
  fi
  if [[ ! -s "$listing" ]]; then
    echo "[reconcile] NOT RECONCILED: $listing did not arrive (job-nanny made no final copy?)." \
         "Stale copies of files the run deleted -- finished GA searches' meeting01_ga_*" \
         "_checkpoint.json / _cache.jsonl -- may remain in $scope; a RESUME=1 would read them." >&2
    return 0
  fi
  if [[ "$(head -n 1 "$listing")" != "# job $job: "* ]]; then
    echo "[reconcile] REFUSED: $listing is not job $job's listing" >&2
    return 1
  fi
  [[ -d "$scope" ]] || { rm -f -- "$listing" "$listing.tmp"; return 0; }
  local -A keep=()
  local line
  while IFS= read -r line; do
    [[ "$line" == "#"* || -z "$line" ]] && continue
    keep["$line"]=1
  done < "$listing"
  local f rel name g match removed=0
  while IFS= read -r -d '' f; do
    rel="${f#"$root"/}"
    [[ -n "${keep[$rel]:-}" ]] && continue
    name="${f##*/}"
    [[ "$name" == .job-final-files-* ]] && continue
    match=$(( $# == 0 ))
    for g in "$@"; do
      # shellcheck disable=SC2053  # the glob is meant to match
      if [[ "$name" == $g ]]; then match=1; break; fi
    done
    (( match )) || continue
    rm -f -- "$f"
    echo "[reconcile] removed $f (deleted by the run; a periodic copy had brought it home)" >&2
    removed=$((removed + 1))
  done < <(find "$scope" -type f ! -newer "$listing" -print0)
  echo "[reconcile] job $job: $removed stale file(s) removed from $scope" >&2
  rm -f -- "$listing" "$listing.tmp"
}

_job_nanny_reconcile_self_test() {
  local tmp fails=0
  tmp="$(mktemp -d)"
  trap 'rm -rf -- "$tmp"' RETURN
  fail() { echo "[self-test] FAIL: $*" >&2; fails=$((fails + 1)); }

  # WORK_DIR side: what the run left (a stale checkpoint is NOT among them)
  local work="$tmp/work/results"
  mkdir -p "$work/meeting01_prevalidation/snn-ae" "$work/meeting01"
  touch "$work/meeting01_prevalidation/snn-ae/run1_manifest.json" \
        "$work/meeting01_prevalidation/snn-ae/a file with spaces.csv" \
        "$work/meeting01/meeting01_loso_fsdd_fold0_comparative_metrics.csv"
  job_nanny_write_listing "$work" 77 || fail "write_listing returned non-zero"
  [[ -f "$work/.job-final-files-77.txt" && ! -e "$work/.job-final-files-77.txt.tmp" ]] || fail "listing not written through rename"
  [[ "$(head -n 1 "$work/.job-final-files-77.txt")" == "# job 77: "* ]] || fail "listing header"
  grep -qx "meeting01_prevalidation/snn-ae/a file with spaces.csv" "$work/.job-final-files-77.txt" || fail "listing lacks a spaced name"

  # home side: the final copy (the run's files + the listing) on top of older periodic copies
  local home="$tmp/home/results"
  mkdir -p "$home/meeting01_prevalidation/snn-ae" "$home/meeting01"
  touch -d '2026-11-07 01:00' "$home/meeting01_prevalidation/snn-ae/meeting01_ga_x_run1_checkpoint.json" \
                              "$home/meeting01_prevalidation/snn-ae/meeting01_ga_x_run1_cache.jsonl" \
                              "$home/meeting01/meeting01_ga_meeting01_loso_fsdd_fold0_run1_checkpoint.json" \
                              "$home/meeting01/meeting01_ga_meeting01_loso_fsdd_fold1_run1_checkpoint.json"
  cp -a "$work/." "$home/"
  touch -d '+1 minute' "$home/meeting01_prevalidation/snn-ae/written_after_the_run.txt"

  # 1. whole-scope reconcile (the pre-validation): both stale artifacts go, the run's files and a
  #    file newer than the listing stay, the listing is removed
  job_nanny_reconcile "$home" "$home/meeting01_prevalidation" 77 2>/dev/null || fail "reconcile returned non-zero"
  [[ ! -e "$home/meeting01_prevalidation/snn-ae/meeting01_ga_x_run1_checkpoint.json" ]] || fail "stale checkpoint kept"
  [[ ! -e "$home/meeting01_prevalidation/snn-ae/meeting01_ga_x_run1_cache.jsonl" ]] || fail "stale cache kept"
  [[ -e "$home/meeting01_prevalidation/snn-ae/run1_manifest.json" ]] || fail "a file the run left was removed"
  [[ -e "$home/meeting01_prevalidation/snn-ae/a file with spaces.csv" ]] || fail "a spaced name the run left was removed"
  [[ -e "$home/meeting01_prevalidation/snn-ae/written_after_the_run.txt" ]] || fail "a file newer than the listing was removed"
  [[ ! -e "$home/.job-final-files-77.txt" ]] || fail "listing not removed after reconcile"
  # outside the scope: untouched
  [[ -e "$home/meeting01/meeting01_ga_meeting01_loso_fsdd_fold0_run1_checkpoint.json" ]] || fail "a file outside the scope was removed"

  # 2. array task (fsdd fold 0): only its own fold's names, another task's stale file stays
  cp -a "$work/.job-final-files-77.txt" "$home/"
  touch -d '+2 minutes' "$home/.job-final-files-77.txt"
  job_nanny_reconcile "$home" "$home/meeting01" 77 "meeting01_loso_fsdd_fold0_*" "meeting01_ga_meeting01_loso_fsdd_fold0_*" 2>/dev/null
  [[ ! -e "$home/meeting01/meeting01_ga_meeting01_loso_fsdd_fold0_run1_checkpoint.json" ]] || fail "the task's own stale checkpoint kept"
  [[ -e "$home/meeting01/meeting01_ga_meeting01_loso_fsdd_fold1_run1_checkpoint.json" ]] || fail "another task's file was removed"
  [[ -e "$home/meeting01/meeting01_loso_fsdd_fold0_comparative_metrics.csv" ]] || fail "the task's result was removed"

  # 3. no listing: nothing removed, a loud message, success (the run's own exit code stands)
  touch -d '2026-11-07 01:00' "$home/meeting01/meeting01_ga_meeting01_loso_fsdd_fold0_run2_cache.jsonl"
  local msg
  msg="$(job_nanny_reconcile "$home" "$home/meeting01" 78 2>&1)" || fail "missing listing must not fail the job"
  [[ "$msg" == *"NOT RECONCILED"* ]] || fail "missing listing not reported"
  [[ -e "$home/meeting01/meeting01_ga_meeting01_loso_fsdd_fold0_run2_cache.jsonl" ]] || fail "removed a file without a listing"

  # 4. refusals: another job's listing, a scope outside the root
  printf '# job 99: other\n' > "$home/.job-final-files-79.txt"
  job_nanny_reconcile "$home" "$home/meeting01" 79 2>/dev/null && fail "another job's listing accepted"
  job_nanny_reconcile "$home" "$tmp/elsewhere" 77 2>/dev/null && fail "a scope outside the root accepted"

  if ((fails)); then return 1; fi
  echo "[self-test] all job-nanny reconcile checks passed"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  case "${1:-}" in
    --self-test) _job_nanny_reconcile_self_test ;;
    *) echo "usage: bash $0 --self-test   (otherwise sourced by the GridUnesp sbatch files)" >&2; exit 2 ;;
  esac
fi
