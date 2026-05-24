#!/usr/bin/env bash
# Sync ML/C++/research skills from dotfiles OpenCode skill library into this project.
# Source of truth: ~/dotfiles/stow/config/.config/opencode/skills/<name>/SKILL.md
# Destinations:
#   .claude/commands/<name>.md   (Claude Code slash commands)
#   .github/skills/<name>/SKILL.md  (GitHub Copilot)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$HOME/dotfiles/stow/config/.config/opencode/skills"
CLAUDE_DIR="$ROOT/.claude/commands"
COPILOT_DIR="$ROOT/.github/skills"

# Skills relevant to this project (ML/C++/SNN/research — excludes dotfiles-specific)
SKILLS=(
    agent-performance-enforcer
    batch-metrics-aggregation
    bibliography-verifier
    build-test
    checkpoint-versioning-enforcer
    class-balance-and-stratification-enforcer
    cli-contract-enforcer
    cpp-performance-and-consistency-optimizer
    data-normalization-contract-enforcer
    dev-tooling-integrator
    experiment-config-schema-validator
    experiment-reproducibility
    filesystem-layout-enforcer
    fold-metadata-tracker
    gradient-flow-validator
    graphify
    hyperparameter-search-logger
    initialization-determinism-enforcer
    logging
    memory-constrained-design
    navigation
    nn-core-usage-enforcer
    numerical-stability-enforcer
    one-definition-per-file
    patching
    snn-efficiency-optimizer
    snn-parameter-bounds-enforcer
    state-of-the-art-code-reference-search
    static-analysis-enforcer
    thread-safety-contract-enforcer
    token-efficiency-enforcer
    validation-sequencing-enforcer
    wiki
)

mkdir -p "$CLAUDE_DIR" "$COPILOT_DIR"

synced=0
missing=0
for name in "${SKILLS[@]}"; do
    src="$SRC_DIR/$name/SKILL.md"
    if [[ ! -f "$src" ]]; then
        echo "MISSING: $src" >&2
        missing=$((missing + 1))
        continue
    fi

    # Claude command: strip YAML frontmatter (lines between first two --- delimiters)
    claude_dst="$CLAUDE_DIR/$name.md"
    awk 'BEGIN{fm=0; done=0} /^---$/{if(fm==0){fm=1;next} if(fm==1 && done==0){done=1;next}} done==1 || fm==0' "$src" > "$claude_dst"

    # Copilot: copy SKILL.md verbatim
    copilot_dst_dir="$COPILOT_DIR/$name"
    mkdir -p "$copilot_dst_dir"
    cp "$src" "$copilot_dst_dir/SKILL.md"

    synced=$((synced + 1))
done

printf 'synced %d skills (%d missing) → .claude/commands/ + .github/skills/\n' "$synced" "$missing"
