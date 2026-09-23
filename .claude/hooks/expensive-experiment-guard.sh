#!/usr/bin/env bash
# PreToolUse: block long-running experiment commands without explicit confirmation marker
tmpfile=$(mktemp)
cat > "$tmpfile"

tool=$(python3 - "$tmpfile" <<'EOF'
import json, sys
try:
    with open(sys.argv[1]) as f: d = json.load(f)
    print(d.get('tool_name', ''))
except Exception: print('')
EOF
)

if [[ "$tool" != "Bash" ]]; then rm -f "$tmpfile"; exit 0; fi

cmd=$(python3 - "$tmpfile" <<'EOF'
import json, sys
try:
    with open(sys.argv[1]) as f: d = json.load(f)
    print(d.get('tool_input', {}).get('command', ''))
except Exception: print('')
EOF
)
rm -f "$tmpfile"

# Patterns for expensive runs: full pipeline, long timeout experiments, all-profiles runs.
# 01_meeting01_run_article_profiles.sh (the old ~2.5h article pipeline this pattern was
# named after) was deleted 2026-09-23 along with the article-*.json profiles it ran (they
# never set cv_fold, so they ran the leakage-prone pooled/shuffled split -- see
# .wiki/Experiments/Meeting01.md). 01_meeting01_run_loso.sh is now the expensive one
# (multi-day/multi-week/multi-month grid; it also has its own internal
# EXPERIMENT_CONFIRMED=1 gate, so this is a second, redundant guard).
if echo "$cmd" | grep -qE '(01_meeting01_run_loso(_gridunesp)?\.(sh|sbatch)|timeout[[:space:]]+[0-9]{3,}|experiment0[0-9][[:space:]]--profile|ctest[[:space:]].*-j[0-9]+[[:space:]]*$)'; then
    # Allow if user added explicit "CONFIRMED:" prefix in comment or env
    if echo "$cmd" | grep -qiE '(#[[:space:]]*CONFIRMED|EXPERIMENT_CONFIRMED=1)'; then
        exit 0
    fi
    echo "BLOCKED: long-running experiment detected. Estimated time: minutes to hours."
    echo "Add '# CONFIRMED' comment or set EXPERIMENT_CONFIRMED=1 to proceed."
    echo "Command: $cmd"
    exit 2
fi

exit 0
