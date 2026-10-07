#!/usr/bin/env bash
# _provenance.sh -- which code is this run made of? NOT run directly: sourced by
# 01_meeting01_run_loso.sh (and exercised by `source_revision.py --self-test`).
#
# The problem. A run lasts weeks, and the tree on the cluster has no git metadata, so the
# run could not say which sources its binary came from -- the events said git_commit
# "unknown". Worse, the binary and the sources can disagree: a deploy that syncs a newer tree
# and then fails (or skips) its build leaves new sources next to an OLD binary, and a job that
# waited days in the queue would start on it without a word.
#
# Two files carry the answer (format and meaning: source_revision.py):
#
#   SOURCE_REVISION                           in the tree; written by gridunesp_deploy.sh
#                                             BEFORE it syncs, so it describes what was shipped
#   out/build/<preset>/SOURCE_REVISION.built  a copy, written only AFTER a successful build
#
#   deployed tree, binary built from it       the two files are identical      -> run
#   sync done, build failed or skipped        .built is older / absent         -> REFUSED
#   git checkout (a local run)                no SOURCE_REVISION in the tree   -> computed live
#
# Loud, not silent: a refused start costs a rerun of the deploy; an unrefused one costs a
# multi-week run of the wrong binary.
#
# Functions (all `return 1` with a message on stderr instead of guessing):
#   provenance_label FILE                  one-line label of a SOURCE_REVISION block
#   provenance_resolve ROOT PY             sets + exports MEETING01_GIT_COMMIT
#   provenance_check_binary_is_current ROOT BUILD     the stale-binary refusal above
#   provenance_mark_built ROOT BUILD       stamp after a successful build (no-op in a checkout)
#   provenance_log_start ROOT RESUME       append one line to results/meeting01/source_revisions.log

_PROVENANCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_provenance_field() { sed -n "s/^$2=//p" "$1" | head -n 1; }

# 414e2422/3fa9c01b2d4e   or   414e2422+dirty/3fa9c01b2d4e
provenance_label() {
  local file="$1" commit dirty tree
  commit="$(_provenance_field "$file" commit)"
  dirty="$(_provenance_field "$file" dirty)"
  tree="$(_provenance_field "$file" tree_sha256)"
  if [[ -z "$commit" || -z "$tree" ]]; then
    echo "[provenance] $file lacks commit= / tree_sha256= (edited or truncated): rerun" \
         "scripts/pipeline/meeting01/gridunesp_deploy.sh to write it again" >&2
    return 1
  fi
  local mark=""
  [[ "$dirty" == "yes" ]] && mark="+dirty"
  printf '%s%s/%s' "$commit" "$mark" "${tree:0:12}"
}

provenance_resolve() {
  local root="$1" py="$2"
  if [[ -f "$root/SOURCE_REVISION" ]]; then
    MEETING01_GIT_COMMIT="$(provenance_label "$root/SOURCE_REVISION")" || return 1
  else
    local block
    block="$(mktemp)"
    if ! "$py" "$_PROVENANCE_DIR/source_revision.py" compute "$root" > "$block"; then
      rm -f "$block"
      echo "[provenance] no $root/SOURCE_REVISION and no usable git checkout: this tree cannot" \
           "say which code it is. Deploy it with gridunesp_deploy.sh, or run from the repository" >&2
      return 1
    fi
    MEETING01_GIT_COMMIT="$(provenance_label "$block")" || { rm -f "$block"; return 1; }
    rm -f "$block"
  fi
  export MEETING01_GIT_COMMIT
}

provenance_check_binary_is_current() {
  local root="$1" built="$1/out/build/$2/SOURCE_REVISION.built"
  [[ -f "$root/SOURCE_REVISION" ]] || return 0   # a git checkout: nothing deployed to compare
  if cmp -s "$root/SOURCE_REVISION" "$built"; then
    return 0
  fi
  local binary="no build stamp ($built)"
  [[ -f "$built" ]] && binary="$(provenance_label "$built")"
  cat >&2 <<EOF
[provenance] REFUSED: the meeting01 binary was not built from the sources now on disk.
  sources : $(provenance_label "$root/SOURCE_REVISION")
  binary  : ${binary}
A sync of a newer tree whose build then failed or was skipped leaves exactly this state; the
job would run the old binary and record the new revision. Rerun
scripts/pipeline/meeting01/gridunesp_deploy.sh (it rebuilds and re-stamps).
EOF
  return 1
}

provenance_mark_built() {
  if [[ -f "$1/SOURCE_REVISION" ]]; then
    cp "$1/SOURCE_REVISION" "$1/out/build/$2/SOURCE_REVISION.built"
  fi
  return 0
}

provenance_log_start() {
  mkdir -p "$1/results/meeting01"
  printf '%s start resume=%s revision=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" \
    "$MEETING01_GIT_COMMIT" >> "$1/results/meeting01/source_revisions.log"
}
