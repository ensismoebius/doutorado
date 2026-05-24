#!/usr/bin/env bash
# PostToolUse: remind completion gates after meaningful coding activity
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

emit=0

if [[ "$tool" == "Edit" || "$tool" == "Write" ]]; then
    emit=1
elif [[ "$tool" == "Bash" ]]; then
    cmd=$(python3 - "$tmpfile" <<'EOF'
import json, sys
try:
    with open(sys.argv[1]) as f: d = json.load(f)
    print(d.get('tool_input', {}).get('command', ''))
except Exception: print('')
EOF
)
    if echo "$cmd" | grep -qiE '(^|[[:space:]])(git[[:space:]]+status|git[[:space:]]+diff|git[[:space:]]+commit|ctest|cmake[[:space:]]+--build|ninja|clang-tidy|cppcheck)'; then
        emit=1
    fi
fi

rm -f "$tmpfile"

[[ "$emit" -eq 1 ]] && echo "Completion gate: apply /agent-performance-enforcer and /validation-sequencing-enforcer before finishing."

exit 0
