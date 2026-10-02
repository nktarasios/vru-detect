# Contributing

Thanks for helping keep VRU-Detect reproducible. This repository is a
research/portfolio benchmark, so contributions should make the workflow easier
to rerun and audit without committing restricted datasets or generated model
artifacts.

## Environment setup

Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Use `python3` in documentation and scripts unless there is a project-specific
reason to use another interpreter command.

## Dataset access

Do not commit raw BDD100K data, generated YOLO datasets, checkpoints, demo
images, or result directories.

The documented local data path uses the Hugging Face mirror `dgural/bdd100k`
because the official ETH host (`dl.cv.ethz.ch`) failed DNS resolution in the
build environment:

```bash
python3 scripts/download_bdd_hf.py
python3 -m src.dataset
```

The downloader creates this supported raw layout:

```text
data/raw/bdd100k/images/100k/{train,val}/
data/raw/bdd100k/labels/det_20/det_train.json
data/raw/bdd100k/labels/det_20/det_val.json
```

The converter also supports `data/raw/images/...` and `data/raw/labels/...`
for manually prepared BDD-style inputs.

Dataset facts to preserve in docs and reports:

- BDD100K via Hugging Face mirror `dgural/bdd100k`
- BDD100K license scope: educational, research, and not-for-profit
- raw data is not redistributed
- local subsample: `1200` train / `300` validation images
- split source: VRU-preferring split of the validation mirror
- taxonomy aliases:
  - `pedestrian` -> `person`
  - `bicycle` -> `bike`

Weights trained on BDD100K may be encumbered for commercial redistribution.
Default to shipping code and instructions, not restricted data or weights.

## Reproduce the benchmark

1. Download the local mirror subset:
   `python3 scripts/download_bdd_hf.py`
2. Convert BDD-style labels to YOLO format:
   `python3 -m src.dataset`
3. Run the COCO-pretrained baseline:
   `python3 -m src.baseline_eval --weights yolo26n.pt --conf 0.25 --max-images 300`
4. Fine-tune:
   `python3 -m src.train --device auto`
5. Evaluate tuned weights:
   `python3 -m src.evaluate --weights <path-to-best.pt> --device auto`
6. Cross-reference the public NHTSA SGO archive with stratified recall:
   `python3 -m src.crossref_sgo`
   This writes `results/sgo_crossref/finding.md`. Downloaded CSVs cache under
   `data/downloads/sgo/` and stay gitignored.
7. Run a demo folder:
   `python3 -m src.infer_demo --weights <path-to-best.pt> --images <folder> --device auto`
8. Rebuild portfolio charts from committed CSVs:
   `python3 scripts/build_portfolio_figures.py`

Do not invent metrics. Only publish numbers from a completed reproducible run.
Phase 1–3 reference numbers already live under `results/`; update them only when
you re-run the corresponding evaluation.

## Tests

Run the focused test suite before opening a PR:

```bash
pytest
```

For changes that affect imports, CLI modules, or broad control flow, also run:

```bash
python3 -m compileall src tests
```

Add or update tests when changing:

- BDD-to-YOLO conversion behavior;
- taxonomy mappings;
- metric matching or stratification;
- threshold recommendation logic;
- demo threshold loading.

Documentation-only changes do not usually need new tests, but they should
still keep commands and paths aligned with the code.

## Pull request tips

- Keep each PR scoped to one phase, bug fix, or documentation update.
- State which commands you ran and whether data/model artifacts were required.
- Include before/after metric snippets only when they come from a completed
  reproducible run.
- Preserve the required disclaimers: VRU-Detect is not a safety-certified or
  production perception system and is a research/portfolio benchmark only.
- Avoid unrelated formatting churn in generated files or notebooks.
- Never commit datasets, checkpoints, generated demo images, or local
  environment files. Committed `results/*.csv` tables and summary charts are
  intentional so reviewers can audit reported numbers without re-training.
