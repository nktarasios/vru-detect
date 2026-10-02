# VRU-Detect Findings

Generated from committed evaluation CSVs for portfolio review.

## Headline

Fine-tuning does not invent a perfect VRU detector. It changes the failure
shape: rider false positives collapse, person recall rises, and the remaining
risk concentrates in night scenes and small distant objects.

## Class-level change (conf=0.25)

| class | baseline P / R | fine-tuned P / R | product read |
| --- | --- | --- | --- |
| person | 0.754 / 0.257 | 0.689 / 0.349 | More pedestrians caught; precision dips slightly, acceptable if misses are costlier. |
| rider | 0.039 / 0.351 | 0.632 / 0.324 | Precision jumps; COCO person/bike confusion was the baseline pathology. |
| bike | 0.714 / 0.075 | 0.625 / 0.149 | Recall roughly doubles but stays low, still an open product risk. |
| all | 0.396 / 0.248 | 0.685 / 0.335 | Aggregate gains hide residual night/small-object weakness. |

## Hard-condition recall

| condition | baseline recall | fine-tuned recall | delta |
| --- | ---: | ---: | ---: |
| Night (all) | 0.146 | 0.241 | +0.095 |
| Night person | 0.145 | 0.262 | +0.116 |
| Small objects | 0.003 | 0.079 | +0.076 |
| Medium objects | 0.185 | 0.335 | +0.150 |
| City street | 0.242 | 0.342 | +0.100 |

## Product implications

1. **Do not ship on aggregate mAP alone.** Night person recall and small-object
   recall remain the safety-relevant bottlenecks after fine-tuning.
2. **Thresholds are product decisions.** Person/rider use recall-first
   operating points (`0.05`); bike uses best-F1 (`0.15`) because false bike
   triggers were noisier relative to the recall gain.
3. **The portfolio claim is methodological.** This repo shows how to expose
   condition-specific risk and choose an operating point deliberately, not
   that a CPU-trained nano model is deployment-ready.

## Charts

- `results/portfolio/baseline_vs_finetuned.png`
- `results/portfolio/failure_modes_recall.png`
- `results/portfolio/threshold_tradeoff.png`
