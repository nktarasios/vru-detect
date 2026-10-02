# Scripts

## `download_bdd_hf.py`

`download_bdd_hf.py` downloads the local VRU-Detect BDD100K subset from the
Hugging Face mirror `dgural/bdd100k`.

The official ETH BDD100K host (`dl.cv.ethz.ch`) is preferred when reachable,
but DNS resolution failed in the build environment used for this project. This
script documents and automates the fallback mirror path.

### What it creates

Default command:

```bash
python3 scripts/download_bdd_hf.py
```

Default output layout:

```text
data/raw/bdd100k/images/100k/train/
data/raw/bdd100k/images/100k/val/
data/raw/bdd100k/labels/det_20/det_train.json
data/raw/bdd100k/labels/det_20/det_val.json
data/raw/bdd100k/SOURCE.txt
```

Default local split:

- `1200` train images
- `300` validation images
- VRU-preferring split of the Hugging Face validation mirror

The split is deterministic with `--seed 42` unless overridden.

### Taxonomy normalization

The Hugging Face mirror uses FiftyOne-style labels for some classes. The
downloader writes BDD-style frames and normalizes the aliases needed by
VRU-Detect:

- `pedestrian` -> `person`
- `bicycle` -> `bike`

The script also preserves non-VRU classes in the generated BDD-style JSON when
they are present, but downstream VRU-Detect training/evaluation uses only
`person`, `rider`, and `bike`.

### Common options

```bash
python3 scripts/download_bdd_hf.py \
  --raw-dir data/raw/bdd100k \
  --max-vru 1200 \
  --max-bg 300 \
  --seed 42 \
  --workers 16
```

- `--raw-dir`: destination raw-data root.
- `--max-vru`: maximum VRU-containing mirror frames to sample before the local
  train/validation split.
- `--max-bg`: maximum background/non-VRU mirror frames to sample before the
  local train/validation split.
- `--seed`: deterministic shuffle seed.
- `--workers`: parallel image download workers.

After download, convert the data with:

```bash
python3 -m src.dataset
```

### License and redistribution

BDD100K is licensed for educational, research, and not-for-profit use. Raw data
downloaded by this script must not be committed or redistributed through this
repository. Weights trained on BDD100K may also be encumbered for commercial
redistribution, so the default packaging policy is to ship code and
reproduction instructions rather than restricted data or trained weights.

## `build_portfolio_figures.py`

Rebuild the summary charts and findings note from committed CSVs:

```bash
python3 scripts/build_portfolio_figures.py
```

Writes:

- `results/portfolio/baseline_vs_finetuned.png`
- `results/portfolio/failure_modes_recall.png`
- `results/portfolio/threshold_tradeoff.png`
- `results/portfolio/FINDINGS.md`

No dataset download or GPU is required for this script.

## `build_error_report.py`

Build the miss-concentration / residual-risk markdown report:

```bash
python3 scripts/build_error_report.py
```

Writes `results/portfolio/ERROR_REPORT.md`.

## `reproduce.sh`

End-to-end reproduction helper (install deps, download, convert, baseline,
train, evaluate, rebuild portfolio artifacts):

```bash
bash scripts/reproduce.sh
```
