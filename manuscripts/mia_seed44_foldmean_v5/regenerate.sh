#!/usr/bin/env bash
# Full local-data regeneration. Overleaf compilation only needs build.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
analysis_python="${MIA_ANALYSIS_PYTHON:-python3}"
"$analysis_python" scripts/extended_metrics.py
"$analysis_python" scripts/significance_analysis.py
"$analysis_python" scripts/subgroup_analysis.py
"$analysis_python" scripts/build_results.py
"$analysis_python" scripts/build_metric_intervals.py
"$analysis_python" scripts/draw_framework.py
"$analysis_python" scripts/plot_expanded.py
"$analysis_python" scripts/plot_mechanism.py
bash build.sh
