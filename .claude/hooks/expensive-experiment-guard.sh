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

# Patterns for expensive runs: full pipeline, long timeout experiments, all-profiles runs
if echo "$cmd" | grep -qE '(e04_run_article_profiles\.sh|timeout[[:space:]]+[0-9]{3,}|experiment0[0-9][[:space:]]--profile|ctest[[:space:]].*-j[0-9]+[[:space:]]*$)'; then
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
