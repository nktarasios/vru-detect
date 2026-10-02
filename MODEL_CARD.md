# VRU-Detect Model Card

## Model details

Portfolio companions: [README.md](README.md),
[docs/PRODUCT_DECISIONS.md](docs/PRODUCT_DECISIONS.md),
[results/portfolio/FINDINGS.md](results/portfolio/FINDINGS.md).

VRU-Detect benchmarks a YOLO26 nano detector (`yolo26n`) on vulnerable road
user (VRU) classes derived from BDD100K Detection labels:

- `person`
- `rider`
- `bike`

The baseline is the COCO-pretrained `yolo26n` model evaluated at confidence
threshold `0.25`. The Phase 2-3 fine-tuned model is a CPU-trained `yolo26n`
run on the local BDD100K validation-mirror subsample. The weights path used by
scripts is `models/vru_best.pt`; the weights are not committed and should be
reproduced with `python3 -m src.train`.

## Intended use

This project is intended for:

- research and portfolio demonstration of a road-scene detection workflow;
- reproducible comparison between a COCO-pretrained detector and a
  BDD-focused fine-tuned detector;
- analysis of weak spots across class, time of day, weather, scene, and object
  size strata;
- pipeline validation for data conversion, metric computation, threshold
  sweeps, and demo visualization.

## Out-of-scope use

VRU-Detect is not appropriate for:

- safety-certified or production perception systems;
- autonomous-driving deployment decisions;
- regulatory, compliance, insurance, or safety-case claims;
- commercial redistribution of BDD100K raw data or potentially encumbered
  BDD-trained weights without reviewing and satisfying the upstream license
  terms.

## Training data provenance

Primary dataset:

- BDD100K via Hugging Face mirror `dgural/bdd100k`
- official ETH host `dl.cv.ethz.ch` was preferred but DNS failed in the build
  environment
- BDD100K license scope: educational, research, and not-for-profit use

Local subset:

- `1200` train images
- `300` validation images
- VRU-preferring train/validation split created from the validation mirror
- raw data is downloaded locally and is not redistributed in this repository

Taxonomy normalization:

- `pedestrian` -> `person`
- `bicycle` -> `bike`

The download step is:

```bash
python3 scripts/download_bdd_hf.py
```

The conversion step is:

```bash
python3 -m src.dataset
```

## Evaluation methodology

Evaluation uses processed YOLO-format labels from `data/processed` and reports
precision/recall by class at IoU threshold `0.5`.

Baseline command:

```bash
python3 -m src.baseline_eval --weights yolo26n.pt --conf 0.25 --max-images 300
```

Baseline results on the 300-image validation sample:

| class | precision | recall |
| --- | ---: | ---: |
| person | 0.754 | 0.257 |
| rider | 0.039 | 0.351 |
| bike | 0.714 | 0.075 |

Known stratified baseline weak spots:

- night `person` recall: approximately `0.145`
- small-object recall: approximately `0.003`

Fine-tuned evaluation command:

```bash
python3 -m src.evaluate --weights models/vru_best.pt
```

Fine-tuned and baseline results below use confidence threshold `0.25` on the
same 300-image validation sample:

| class | baseline precision | baseline recall | fine-tuned precision | fine-tuned recall |
| --- | ---: | ---: | ---: | ---: |
| person | 0.754 | 0.257 | 0.689 | 0.349 |
| rider | 0.039 | 0.351 | 0.632 | 0.324 |
| bike | 0.714 | 0.075 | 0.625 | 0.149 |
| all | 0.396 | 0.248 | 0.685 | 0.335 |

Observed changes:

- `person` recall improved from `0.257` to `0.349`.
- `rider` precision improved from `0.039` to `0.632`, with similar recall.
- `bike` recall roughly doubled from `0.075` to `0.149`, but remains low.
- Aggregate precision and recall both improved.

Completed training run:

- model: `yolo26n`
- epochs: `10`
- device: CPU
- VRU oversample factor: `3`
- completed run image size / batch: `640` / `4`
- best validation mAP50: approximately `0.330` at epoch 8
- training curves: `results/training/results.csv`

Recommended per-class thresholds from
`results/finetuned/threshold_justification.md`:

- `person`: `0.05` (P=`0.291`, R=`0.605`) - recall-prioritized
- `rider`: `0.05` (P=`0.219`, R=`0.432`) - recall-prioritized
- `bike`: `0.15` (P=`0.424`, R=`0.209`) - best F1

## Known limitations

- The BDD100K Hugging Face mirror used here is a local fallback because the
  official ETH host did not resolve in the build environment.
- The local dataset is a small VRU-preferring subsample, not the full BDD100K
  detection benchmark.
- The split is created from the validation mirror, so it is suitable for
  portfolio benchmarking and pipeline validation, not for publication-grade
  claims about full BDD100K performance.
- Class mapping is approximate: the mirror uses aliases such as `pedestrian`
  and `bicycle`, which are normalized to `person` and `bike`; rider/bike
  semantics can remain ambiguous.
- The baseline maps COCO detections into the project taxonomy, so rider
  behavior is especially approximate before fine-tuning.
- The completed fine-tuned model was trained on CPU with a 1200/300 subsample
  of the BDD100K validation mirror, not with a full official BDD100K train
  split GPU run.
- CPU-scale training and small local runs may not fully converge; `bike` recall
  remains low even after fine-tuning.
- Aggregate precision/recall can hide important failure modes; the baseline
  already shows poor night-person and small-object recall.

## Distribution and license notes

Raw BDD100K data must not be committed to this repository. Generated datasets,
checkpoints, demo images, and result directories should also remain out of
version control unless a specific artifact is cleared for publication.

Weights trained on BDD100K may be encumbered for commercial redistribution.
The default packaging stance is to ship code and reproduction instructions,
not restricted raw data or trained weights.

## Safety disclaimer

VRU-Detect is not a safety-certified or production perception system. It is a
research/portfolio benchmark only. Results from this repository must not be
used as evidence that a detector is safe for real-world driving.
