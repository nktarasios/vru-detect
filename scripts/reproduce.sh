#!/usr/bin/env bash
# Reproduce the VRU-Detect benchmark from a clean checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m pip install -r requirements.txt

python3 scripts/download_bdd_hf.py
python3 -m src.dataset
python3 -m src.baseline_eval --weights yolo26n.pt --conf 0.25 --max-images 300
python3 -m src.train --device auto
python3 -m src.evaluate --weights models/vru_best.pt --device auto
python3 scripts/build_portfolio_figures.py
python3 scripts/build_error_report.py

echo
echo "Optional demo (writes gitignored outputs/demo):"
echo "  python3 -m src.infer_demo --weights models/vru_best.pt --images data/processed/images/val --output-dir outputs/demo"
