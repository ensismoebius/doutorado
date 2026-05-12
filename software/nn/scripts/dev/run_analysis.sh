#!/bin/bash
# Analyze completed grid search results and generate comprehensive tables

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/src/experiments/03/results"
ANALYSIS_DIR="$PROJECT_ROOT/analysis"

echo "════════════════════════════════════════════════════════════"
echo "  SNN Grid Search Result Analysis"
echo "════════════════════════════════════════════════════════════"

# Check for results
RESULT_COUNT=$(find "$RESULTS_DIR" -maxdepth 1 -name "*grid_*.json" 2>/dev/null | wc -l)

if [ "$RESULT_COUNT" -eq 0 ]; then
    echo "❌ No completed results found in $RESULTS_DIR"
    echo "   Grid search may still be running. Check:"
    echo "   tail -20 logs/grid_runs/*/joblog.tsv"
    exit 1
fi

echo "✓ Found $RESULT_COUNT completed profiles"

# Create analysis directory
mkdir -p "$ANALYSIS_DIR"
echo "✓ Output directory: $ANALYSIS_DIR"

# Run Python analysis
echo ""
echo "Generating analysis tables..."
cd "$PROJECT_ROOT"
python3 scripts/analyze_grid_results.py

# Report generated files
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Generated Files"
echo "════════════════════════════════════════════════════════════"

if [ -f "$ANALYSIS_DIR/master_comparison.csv" ]; then
    LINES=$(wc -l < "$ANALYSIS_DIR/master_comparison.csv")
    echo "✓ master_comparison.csv ($LINES lines)"
    echo "  → Full leaderboard of all $((LINES - 1)) profiles ranked by performance"
fi

if [ -f "$ANALYSIS_DIR/summary_statistics.csv" ]; then
    echo "✓ summary_statistics.csv"
    echo "  → Key metrics: success rate, best/worst/average losses"
    echo ""
    echo "  Summary Stats:"
    sed 's/^/    /' "$ANALYSIS_DIR/summary_statistics.csv" | head -10
fi

echo ""
echo "✓ Modality comparisons:"
for file in "$ANALYSIS_DIR"/comparison_*.csv; do
    if [ -f "$file" ]; then
        NAME=$(basename "$file")
        LINES=$(wc -l < "$file")
        echo "  - $NAME ($((LINES - 1)) configurations)"
    fi
done

echo ""
echo "✓ Hyperparameter sensitivity:"
for file in "$ANALYSIS_DIR"/sensitivity_*.csv; do
    if [ -f "$file" ]; then
        NAME=$(basename "$file")
        echo "  - $NAME"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Next Steps"
echo "════════════════════════════════════════════════════════════"
echo "1. View top 20 configurations:"
echo "   head -21 analysis/master_comparison.csv | column -t -s,"
echo ""
echo "2. Export to Excel/LibreOffice:"
echo "   libreoffice analysis/master_comparison.csv"
echo ""
echo "3. Quick modality comparison:"
echo "   for f in analysis/comparison_*.csv; do echo \"=== $(basename \$f) ===\"  head -6 \$f | tail -5; done"
echo ""
echo "4. Archive results:"
echo "   cp -r analysis/ results_\$(date +%Y%m%d_%H%M%S)_backup"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Analysis complete! ✓"
