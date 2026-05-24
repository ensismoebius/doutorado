#!/usr/bin/env bash
# PostToolUse: remind to update wiki after editing source files
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

[[ "$tool" != "Edit" && "$tool" != "Write" ]] && rm -f "$tmpfile" && exit 0

file=$(python3 - "$tmpfile" <<'EOF'
import json, sys
try:
    with open(sys.argv[1]) as f: d = json.load(f)
    print(d.get('tool_input', {}).get('file_path', ''))
except Exception: print('')
EOF
)
rm -f "$tmpfile"

# Trigger on source/header edits in nn project (not already-wiki files)
if echo "$file" | grep -qE '(include/|src/|software/nn/)' && ! echo "$file" | grep -qE '\.wiki/'; then
    echo "Wiki reminder: if layer/loss/optimizer/training changed → update .wiki/Core/ + .wiki/References.md + run /wiki if structure changed"
fi

exit 0
